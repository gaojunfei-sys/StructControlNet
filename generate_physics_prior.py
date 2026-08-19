

import os
import csv
import pickle
import random
import sys
import time
from collections import deque, Counter, defaultdict

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from shapely.ops import unary_union

from project_config import (
    PHYSICS_DEBUG_DIR as CONFIG_PHYSICS_DEBUG_DIR,
    RANDOM_SEED,
    TESTSET_DIR as CONFIG_TESTSET_DIR,
    TESTSET_HEATMAP_DIR,
)

try:
    from shapely.ops import polylabel as _shapely_polylabel
except Exception:
    _shapely_polylabel = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


TESTSET_DIR = str(CONFIG_TESTSET_DIR)
OUT_DIR = str(TESTSET_HEATMAP_DIR)
DEBUG_DIR = str(CONFIG_PHYSICS_DEBUG_DIR)
SAMPLE_NAME = None   # None = 取 index.csv 第一行; 或填具体 name 字符串如 "399"

# Ground-truth spatial priors are opt-in for debugging and evaluation.
USE_ANNOTATION_LAYOUT_PRIOR = False
USE_ANNOTATION_REGION_MAP = False
ANNOTATION_ANCHOR_ALL_ROOMS = True
USE_TOPOLOGY_LAYOUT_PRIOR = False
USE_ROOM_ORDER_PRIOR = False
ENABLE_ADJACENCY_REPAIR = False

# RPLAN boundary coordinates use a nominal 256 grid.
COORD_SPACE = 256

# Pixel classification thresholds.
WHITE_THRESH       = 240
WALL_CHROMA_MAX    = 22
WALL_VALUE_MAX     = 220
WIN_R_MIN          = 220
WIN_G_MIN          = 180
WIN_B_MAX          = 90
WIN_RB_DIFF_MIN    = 100

# Outer-wall detection.
OUTER_WALL_BAND_MULT  = 1.4
OUTER_WALL_MIN_BAND   = 6
OUTER_CONTOUR_MIN_AREA = 200

REDRAW_INNER_WALL_MIN    = 3
REDRAW_INNER_WALL_MAX    = 6
REDRAW_INNER_WALL_SHRINK = 1

# Several bedroom types share a color but remain separate region IDs.
RTYPE_INFO = {
    0:  ("LivingRoom",  (244, 242, 229)),
    1:  ("MasterRoom",  (253, 244, 171)),
    2:  ("Kitchen",     (234, 216, 214)),
    3:  ("Bathroom",    (205, 233, 252)),
    4:  ("DiningRoom",  (244, 242, 229)),
    5:  ("ChildRoom",   (253, 244, 171)),
    6:  ("StudyRoom",   (253, 244, 171)),
    7:  ("SecondRoom",  (253, 244, 171)),
    8:  ("GuestRoom",   (253, 244, 171)),
    9:  ("Balcony",     (208, 216, 135)),
    10: ("Entrance",    (244, 242, 229)),
    11: ("Storage",     (245, 225, 195)),
    12: ("WallIn",      (180, 180, 180)),
    13: ("External",    (100, 100, 100)),
}

# Force refinement parameters.
SIM_ITERATIONS    = 80
SIM_DT            = 0.25
SIM_MAX_VEL       = 3.0
STRONG_REL        = 1.0
RADIUS_SCALE_SIM  = 0.80
SOFT_ANCHOR_GAIN  = 0.42
REPULSION_GAIN    = 1.2
REL_ATTRACTION_GAIN = 0.16
DAMPING           = 0.55
PRIOR_N_SAMPLES   = 3000

# Keep the living room central while nudging other rooms toward the shell.
EDGE_CROWDING_ENABLED = True
EDGE_ROOM_ANCHOR_ALPHA = 0.24
EDGE_ROOM_INSET_FRAC_DEFAULT = 0.64
EDGE_ROOM_INSET_FRAC_SERVICE = 0.48
EDGE_ROOM_INSET_FRAC_BALCONY = 0.36
EDGE_CORNER_INSET_SCALE = 0.82
EDGE_ATTRACTION_GAIN = 0.11

# Every RPLAN rEdge type denotes physical adjacency and receives equal weight.
EDGE_TYPE_REL = {0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}
DEFAULT_REL = 1.0

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
def classify_pixels(img):
    """
    返回 cmap (h,w) int32:
      -2 = 白背景, -1 = 灰墙(暂全标), -4 = 亮黄窗户, -3 = 其他/待填
    """
    h, w = img.shape[:2]
    r = img[:, :, 0].astype(np.int32)
    g = img[:, :, 1].astype(np.int32)
    b = img[:, :, 2].astype(np.int32)
    cmap = np.full((h, w), -3, dtype=np.int32)

    is_white = (r >= WHITE_THRESH) & (g >= WHITE_THRESH) & (b >= WHITE_THRESH)
    cmap[is_white] = -2

    is_window = (
        (r >= WIN_R_MIN) & (g >= WIN_G_MIN) & (b <= WIN_B_MAX)
        & ((r - b) >= WIN_RB_DIFF_MIN) & (cmap == -3)
    )
    cmap[is_window] = -4

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    is_wall = (cmap == -3) & (chroma <= WALL_CHROMA_MAX) & (mx <= WALL_VALUE_MAX)
    cmap[is_wall] = -1

    n_white  = int((cmap == -2).sum())
    n_wall   = int((cmap == -1).sum())
    n_window = int((cmap == -4).sum())
    n_rest   = int((cmap == -3).sum())
    print(f"  分类: 白={n_white}, 墙={n_wall}, 窗={n_window}, 其他={n_rest}")
    return cmap


def split_outer_inner_walls(cmap):
    """
    用建筑外轮廓划分内/外墙. 返回:
      cmap (内墙改为 -5),
      n_outer, n_inner, inner_wall_thickness (估算),
      inner_mask (bool, 用于采颜色)
    """
    wall_mask = (cmap == -1)
    if not np.any(wall_mask):
        return cmap, 0, 0, 4.0, np.zeros_like(wall_mask)

    inside = cv2.distanceTransform(wall_mask.astype(np.uint8), cv2.DIST_L2, 3)
    half_thickness = float(inside.max())
    wall_thickness = max(2.0, half_thickness * 2.0)
    band = max(OUTER_WALL_MIN_BAND, int(round(wall_thickness * OUTER_WALL_BAND_MULT)))

    foreground = (cmap != -2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    outline_band = np.zeros_like(foreground)
    for c in contours:
        if cv2.contourArea(c) >= OUTER_CONTOUR_MIN_AREA:
            cv2.drawContours(outline_band, [c], -1, 255, thickness=band)

    outer = wall_mask & (outline_band > 0)
    inner = wall_mask & (~outer)

    if inner.any():
        inner_dist = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
        inner_thickness = max(2.0, float(inner_dist.max()) * 2.0)
    else:
        inner_thickness = wall_thickness

    print(f"  墙厚: 全墙≈{wall_thickness:.1f}px, 内墙≈{inner_thickness:.1f}px, "
          f"外墙保留={int(outer.sum())}, 内墙吞掉={int(inner.sum())}")
    return cmap, int(outer.sum()), int(inner.sum()), inner_thickness, inner


def sample_inner_wall_color(img, inner_mask, fallback=(128, 128, 128)):
    if inner_mask is None or not np.any(inner_mask):
        return np.array(fallback, dtype=np.uint8)
    px = img[inner_mask]
    med = np.median(px, axis=0)
    return np.array([int(round(v)) for v in med], dtype=np.uint8)


def calibrate_affine(boundary, cmap):
    """
    源图像的 "彩色 + 内墙" 区域在像素空间, boundary 在 256-空间,
    用两边的 bbox 对齐求出 (sx, sy, ox, oy):
        img_x = bnd_x * sx + ox
        img_y = bnd_y * sy + oy
    返回 (sx, sy, ox, oy).
    """
    bx0 = float(boundary[:, 0].min())
    by0 = float(boundary[:, 1].min())
    bx1 = float(boundary[:, 0].max())
    by1 = float(boundary[:, 1].max())

    foreground = (cmap != -2)
    if not foreground.any():
        h, w = cmap.shape
        cx0, cy0, cx1, cy1 = 0, 0, w - 1, h - 1
    else:
        ys, xs = np.where(foreground)
        cx0, cy0 = int(xs.min()), int(ys.min())
        cx1, cy1 = int(xs.max()), int(ys.max())

    sx = (cx1 - cx0) / max(1.0, bx1 - bx0)
    sy = (cy1 - cy0) / max(1.0, by1 - by0)
    ox = cx0 - bx0 * sx
    oy = cy0 - by0 * sy
    print(f"  仿射: bnd bbox=({bx0:.0f},{by0:.0f}-{bx1:.0f},{by1:.0f}), "
          f"img bbox=({cx0},{cy0}-{cx1},{cy1}), s=({sx:.2f},{sy:.2f}), "
          f"o=({ox:.1f},{oy:.1f})")
    return sx, sy, ox, oy


def boundary_to_pixel(p, affine):
    """256-空间 (x,y) → 图像 (px,py)"""
    sx, sy, ox, oy = affine
    return (p[0] * sx + ox, p[1] * sy + oy)


def pixel_to_boundary(p, affine):
    """图像 (px,py) → 256-空间 (x,y)"""
    sx, sy, ox, oy = affine
    return ((p[0] - ox) / sx, (p[1] - oy) / sy)


def _select_entry_door_mask(cmap, door_seg_256=None, affine=None):
    h, w = cmap.shape
    yellow = (cmap == -4).astype(np.uint8)
    if int(yellow.sum()) == 0:
        return None

    n_comp, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        yellow, connectivity=8
    )
    if n_comp <= 1:
        return None

    target_label = 0
    if door_seg_256 is not None and affine is not None:
        a, b = door_seg_256[0], door_seg_256[1]
        ax, ay = boundary_to_pixel(a, affine)
        bx, by = boundary_to_pixel(b, affine)
        mid_x = int(round((ax + bx) / 2.0))
        mid_y = int(round((ay + by) / 2.0))
        mid_x = int(np.clip(mid_x, 0, w - 1))
        mid_y = int(np.clip(mid_y, 0, h - 1))
        target_label = int(labels[mid_y, mid_x])
        if target_label == 0:
            ys, xs = np.where(yellow == 1)
            if len(ys) > 0:
                d2 = (ys - mid_y) ** 2 + (xs - mid_x) ** 2
                k = int(np.argmin(d2))
                target_label = int(labels[ys[k], xs[k]])

    if target_label == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        target_label = int(np.argmax(areas)) + 1

    return (labels == target_label).astype(np.uint8)


def find_door_corridor_pixels(cmap, door_seg_256=None, affine=None, depth=8):
    """
    入口门的"客厅强制领地像素":
      1. 把 door_seg_256 端点映射到图像像素, 求中点.
      2. 在 cmap == -4 (黄色门窗) 的连通分量里, 找含/最近门中点的那块,
         作为"入口门像素 mask".
      3. 把门 mask 朝内膨胀 depth 像素, 与 cmap == -3 (可生长) 取交集.
      4. 这些像素在 region_grow 之前就硬分配给客厅, 其他房间的生长跨不
         进来 -> LR 必定连着门.
    返回 [(x, y), ...] 像素列表.
    """
    h, w = cmap.shape
    door_mask = _select_entry_door_mask(cmap, door_seg_256, affine)
    if door_mask is None:
        return []

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(door_mask, kernel, iterations=int(depth))
    growable = (cmap == -3).astype(np.uint8)
    corridor = (dilated == 1) & (growable == 1)
    ys, xs = np.where(corridor)
    return list(zip(xs.tolist(), ys.tolist()))


def find_image_entry_door(cmap, affine):
    """Detect the visible yellow entry door and return a boundary-space segment."""
    if affine is None:
        return None
    door_mask = _select_entry_door_mask(cmap)
    if door_mask is None:
        return None
    ys, xs = np.where(door_mask > 0)
    if len(ys) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if (x1 - x0) >= (y1 - y0):
        py = (y0 + y1) / 2.0
        a_px = (float(x0), py)
        b_px = (float(x1), py)
    else:
        px = (x0 + x1) / 2.0
        a_px = (px, float(y0))
        b_px = (px, float(y1))

    a = pixel_to_boundary(a_px, affine)
    b = pixel_to_boundary(b_px, affine)
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
    return (a, b, mid, length)


def extract_sim_polygon(cmap, affine, simplify_eps=1.5):
    """
    从 cmap 中抠出实际建筑 polygon (图像空间), 然后用 inverse affine
    映射回 256-空间, 作为力学模拟的边界多边形.

    这样 polygon 与源图像的外墙完全一致, 不会出现"polygon 没覆盖到的小翼"
    导致区域生长空洞的问题.
    """
    foreground = (cmap != -2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    eps = max(0.5, simplify_eps)
    pts = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2).astype(float)
    pts_b = np.array([pixel_to_boundary(p, affine) for p in pts])
    poly = ShapelyPolygon(pts_b)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def pick_sample(testset_dir, name=None):
    idx_path = os.path.join(testset_dir, "index.csv")
    rows = []
    with open(idx_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if name is None:
        chosen = rows[0]
    else:
        for r in rows:
            if r["name"] == str(name):
                chosen = r
                break
        else:
            raise ValueError(f"在 index.csv 中找不到 name={name}")
    print(f"[样本] name={chosen['name']}, n_rooms={chosen['n_rooms']}, "
          f"n_edges={chosen['n_edges']}")
    return chosen


def load_annotation(testset_dir, name):
    with open(os.path.join(testset_dir, "annotations", f"{name}.pkl"), "rb") as f:
        ann = pickle.load(f)
    return ann


def boundary_to_polygon(boundary):
    """
    boundary: (N,4) int32 [x, y, type1, type2]
    取 (x,y) 序列构成闭合多边形. 返回 shapely Polygon (在 256-空间).
    """
    pts = boundary[:, :2].astype(float).tolist()
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    poly = ShapelyPolygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def boundary_window_segments(boundary):
    """
    boundary 第 3/4 列是边类型. 经验上:
       type1=2 标记一段是窗户; type1=1 是普通墙;
    返回 list of ((x1,y1),(x2,y2)) 表示窗户线段.
    """
    segs = []
    n = len(boundary)
    for i in range(n):
        x1, y1, t1, t2 = boundary[i]
        x2, y2, _, _ = boundary[(i + 1) % n]
        if int(t1) == 2 or int(t2) == 2:
            segs.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return segs


def find_entry_door(boundary):
    """
    找入口门: boundary 上的 "黄色段" (t1/t2 == 2) 里, 取最长的一条.
    RPLAN 里入口门通常比窗户更宽, 所以最长段几乎一定是门.
    返回 ((x1,y1), (x2,y2), midpoint(x,y), length) 或 None.
    """
    segs = boundary_window_segments(boundary)
    if not segs:
        return None
    best = None; best_len = -1.0
    for (a, b) in segs:
        L = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        if L > best_len:
            best_len = L
            best = (a, b)
    if best is None:
        return None
    a, b = best
    mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
    return (a, b, mid, best_len)


class Pt:
    """简化的力导向点 (在 256-坐标空间运行)"""
    __slots__ = ("idx", "rtype", "color", "pos", "vel", "acc",
                 "mass", "radius", "radius_render", "fixed")

    def __init__(self, idx, rtype, color, pos, area):
        self.idx = idx
        self.rtype = rtype
        self.color = color
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(2)
        self.acc = np.zeros(2)
        self.mass = max(area, 50.0)
        self.radius = float(np.sqrt(self.mass / np.pi) * RADIUS_SCALE_SIM)
        self.radius_render = float(np.sqrt(self.mass / np.pi))
        self.fixed = False

    def step(self, dt, max_vel):
        if self.fixed:
            return
        self.vel += self.acc * dt
        v = float(np.linalg.norm(self.vel))
        if v > max_vel:
            self.vel = self.vel / v * max_vel
        self.pos += self.vel * dt
        self.acc[:] = 0.0


def build_relationship_matrix(rType, rEdge):
    n = len(rType)
    R = np.zeros((n, n), dtype=float)
    for a, b, t in rEdge:
        a, b, t = int(a), int(b), int(t)
        if a < 0 or b < 0 or a >= n or b >= n or a == b:
            continue
        rel = EDGE_TYPE_REL.get(t, DEFAULT_REL)
        R[a, b] = max(R[a, b], rel)
        R[b, a] = R[a, b]
    return R


def _bfs_layers(R, root, n):
    """从 root 出发的 BFS 层数 (graph distance). 不连通节点 = -1."""
    layer = [-1] * n
    layer[root] = 0
    queue = [root]
    head = 0
    while head < len(queue):
        u = queue[head]; head += 1
        for v in range(n):
            if R[u, v] > 0 and layer[v] == -1:
                layer[v] = layer[u] + 1
                queue.append(v)
    return layer


def _connected_groups(R, nodes):
    """在子集 nodes 内, 按 R 的连通关系分组 (并查集)."""
    if not nodes:
        return []
    idx = {n: i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for i, ni in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            nj = nodes[j]
            if R[ni, nj] > 0:
                union(i, j)
    groups = {}
    for i, ni in enumerate(nodes):
        groups.setdefault(find(i), []).append(ni)
    return list(groups.values())


def _polygon_anchor(polygon):
    """
    返回适合放枢纽节点的"内部最深点" (pole of inaccessibility).
    优先使用 shapely.ops.polylabel; 不可用时退回采样最大内切圆心;
    再不行用 centroid.
    """
    if _shapely_polylabel is not None:
        try:
            tol = max(polygon.length / 200.0, 1.0)
            p = _shapely_polylabel(polygon, tolerance=tol)
            if polygon.contains(p):
                return float(p.x), float(p.y)
        except Exception:
            pass
    minx, miny, maxx, maxy = polygon.bounds
    best = None; best_d = -1.0
    grid = 30
    for i in range(grid):
        for j in range(grid):
            x = minx + (i + 0.5) * (maxx - minx) / grid
            y = miny + (j + 0.5) * (maxy - miny) / grid
            sp = ShapelyPoint(x, y)
            if polygon.contains(sp):
                d = polygon.exterior.distance(sp)
                if d > best_d:
                    best_d = d; best = (x, y)
    if best is not None:
        return best
    c = polygon.centroid
    return float(c.x), float(c.y)


def split_main_protrusions(polygon, erosion_factor=0.42,
                            min_area_frac=0.015):
    """
    检测 polygon 的"突出部分" (例如阳台 / 小卫生间 / 储藏角落).
    思路: 把 polygon 用 buffer(-eps) 收缩; eps = 主体锚点离边界距离 × factor.
    如果收缩后分裂成多块, 不含主体锚点的那些块就是突出部分.
    返回 [(center_xy, area, radius_estimate), ...] (按面积降序).
    """
    main_anchor = _polygon_anchor(polygon)
    sp = ShapelyPoint(*main_anchor)
    main_r = polygon.exterior.distance(sp)
    if main_r <= 2.0:
        return []
    eps = main_r * erosion_factor
    if eps < 2.0:
        return []
    try:
        eroded = polygon.buffer(-eps, join_style=2)
    except Exception:
        return []
    if eroded.is_empty:
        return []

    if eroded.geom_type == "MultiPolygon":
        parts = list(eroded.geoms)
    else:
        parts = [eroded]
    if len(parts) < 2:
        return []

    main_eroded = None
    sub_eroded = []
    for p in parts:
        if p.contains(sp):
            main_eroded = p
        else:
            sub_eroded.append(p)
    if main_eroded is None:
        parts.sort(key=lambda p: -p.area)
        main_eroded = parts[0]
        sub_eroded = parts[1:]

    min_area = polygon.area * min_area_frac
    results = []
    for p in sub_eroded:
        if p.area < min_area:
            continue
        c = _polygon_anchor(p)
        try:
            restored = p.buffer(eps, join_style=2).intersection(polygon)
            if restored.is_empty:
                area_orig = p.area
            elif hasattr(restored, "geoms"):
                area_orig = max(g.area for g in restored.geoms)
            else:
                area_orig = restored.area
        except Exception:
            area_orig = p.area
        r_est = float(np.sqrt(max(area_orig, 1.0) / np.pi))
        results.append((c, float(area_orig), r_est))

    results.sort(key=lambda x: -x[1])
    return results


def _pick_living_idx(rType, areas, R):
    """
    选定客厅 (中心枢纽节点):
      - 优先 rType == 0 (LivingRoom). 多个取面积最大.
      - 没有时降级到 rType == 4 (DiningRoom).
      - 都没有时取度数最大节点.
    """
    candidates = [i for i, t in enumerate(rType) if int(t) == 0]
    if not candidates:
        candidates = [i for i, t in enumerate(rType) if int(t) == 4]
    if candidates:
        return max(candidates, key=lambda i: areas[i])
    degrees = [int((R[i] > 0).sum()) for i in range(len(rType))]
    return int(np.argmax(degrees))


def _arc_contains(s, arc_start, arc_end, perimeter):
    """1D 周长上 s 是否落在 [arc_start, arc_end] 弧内 (考虑 wrap-around)."""
    s = s % perimeter
    arc_start = arc_start % perimeter
    arc_end = arc_end % perimeter
    if arc_start <= arc_end:
        return arc_start <= s <= arc_end
    return s >= arc_start or s <= arc_end


def _circular_order(R, nodes):
    """
    给节点子集 nodes 生成一条 "周向访问顺序", 让 R[i,j]>0 的节点在序列里相邻.
    用贪心 DFS-like: 从子图度数最低的链端节点出发, 每步选最紧密的未访问邻居.
    """
    if not nodes:
        return []
    if len(nodes) == 1:
        return [nodes[0]]
    sub_deg = {n: int(sum(1 for m in nodes if m != n and R[n, m] > 0)) for n in nodes}
    start = sorted(nodes, key=lambda n: (sub_deg[n], n))[0]
    order = [start]
    visited = {start}
    while len(order) < len(nodes):
        cur = order[-1]
        cands = [m for m in nodes if m not in visited]
        adj = [m for m in cands if R[cur, m] > 0]
        if adj:
            # 优先 R 大、子图度数小 (避免提早把链中段塞死)
            nxt = sorted(adj, key=lambda m: (-R[cur, m], sub_deg[m], m))[0]
        else:
            nxt = sorted(cands, key=lambda m: (sub_deg[m], m))[0]
        order.append(nxt)
        visited.add(nxt)
    return order


PRIVATE_TYPES = {1, 5, 6, 7, 8}
SERVICE_TYPES = {2, 3, 10, 11}
BALCONY_TYPES = {9}


def _edge_inset_frac(rtype):
    t = int(rtype)
    if t in BALCONY_TYPES:
        return EDGE_ROOM_INSET_FRAC_BALCONY
    if t in SERVICE_TYPES:
        return EDGE_ROOM_INSET_FRAC_SERVICE
    return EDGE_ROOM_INSET_FRAC_DEFAULT


def _polygon_corner_candidates(polygon, center, door_mid=None, min_door_dist=0.0):
    coords = list(polygon.exterior.coords)
    if len(coords) <= 4:
        return []
    coords = coords[:-1]
    ccw = bool(polygon.exterior.is_ccw)
    turn_sign = 1.0 if ccw else -1.0
    c = np.asarray(center, dtype=float)
    diag = max(float(np.sqrt(polygon.area)), 1.0)
    out = []
    for idx, cur in enumerate(coords):
        prev = np.asarray(coords[idx - 1], dtype=float)
        p = np.asarray(cur, dtype=float)
        nxt = np.asarray(coords[(idx + 1) % len(coords)], dtype=float)
        e1 = p - prev
        e2 = nxt - p
        l1 = float(np.linalg.norm(e1))
        l2 = float(np.linalg.norm(e2))
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        if turn_sign * cross <= 1e-6:
            continue
        a = prev - p
        b = nxt - p
        cosv = float(np.clip(np.dot(a, b) / (l1 * l2), -1.0, 1.0))
        angle = float(np.arccos(cosv))
        if angle > np.deg2rad(150.0):
            continue
        if door_mid is not None and min_door_dist > 0:
            d_door = float(np.linalg.norm(p - np.asarray(door_mid, dtype=float)))
            if d_door < min_door_dist:
                continue
        radial = float(np.linalg.norm(p - c)) / diag
        sharpness = float(np.pi - angle)
        s = float(polygon.exterior.project(ShapelyPoint(float(p[0]), float(p[1]))))
        out.append({
            "idx": idx,
            "point": (float(p[0]), float(p[1])),
            "s": s,
            "score": sharpness + 0.35 * radial,
        })
    out.sort(key=lambda x: -x["score"])
    return out


def _arc_length(arc_s, arc_e, perimeter):
    length = (arc_e - arc_s) % perimeter
    return perimeter if length <= 1e-6 else length


def _cyclic_distance(a, b, perimeter):
    d = abs((a - b) % perimeter)
    return min(d, perimeter - d)


def _corner_for_arc(corners, arc_s, arc_e, perimeter, used_corners):
    if not corners:
        return None
    arc_len = _arc_length(arc_s, arc_e, perimeter)
    arc_mid = (arc_s + arc_len * 0.5) % perimeter
    best = None
    best_score = -1e9
    for c in corners:
        if c["idx"] in used_corners:
            continue
        if not _arc_contains(c["s"], arc_s, arc_e, perimeter):
            continue
        mid_penalty = _cyclic_distance(c["s"], arc_mid, perimeter) / max(arc_len, 1.0)
        score = float(c["score"]) - 0.25 * mid_penalty
        if score > best_score:
            best_score = score
            best = c
    return best


def _unit(v, fallback=(1.0, 0.0)):
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n < 1e-6:
        return np.asarray(fallback, dtype=float)
    return arr / n


def _door_axes(polygon, boundary, door_info_override=None):
    """
    Derive a local coordinate frame from the entry door:
      inward: from door into the apartment
      tangent: along the door segment
    If no door is available, fall back to a stable vertical frame.
    """
    anchor = np.array(_polygon_anchor(polygon), dtype=float)
    door_info = door_info_override
    if door_info is None and boundary is not None:
        door_info = find_entry_door(boundary)
    if door_info is None:
        return None, anchor, np.array([0.0, 1.0]), np.array([1.0, 0.0])

    a, b, mid, length = door_info
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    mid = np.array(mid, dtype=float)
    tangent = _unit(b - a, fallback=(0.0, 1.0))
    n1 = np.array([-tangent[1], tangent[0]], dtype=float)
    n2 = -n1

    eps = max(2.0, length * 0.35)
    p1 = ShapelyPoint(*(mid + n1 * eps))
    p2 = ShapelyPoint(*(mid + n2 * eps))
    in1 = polygon.contains(p1)
    in2 = polygon.contains(p2)
    if in1 and not in2:
        inward = n1
    elif in2 and not in1:
        inward = n2
    else:
        inward = _unit(anchor - mid, fallback=n1)
    if float(np.dot(inward, anchor - mid)) < 0:
        inward = -inward
    return door_info, mid, _unit(inward), tangent


def _inside_along(origin, direction, distance, polygon, min_frac=0.12):
    origin = np.asarray(origin, dtype=float)
    direction = _unit(direction)
    for frac in np.linspace(1.0, min_frac, 18):
        p = origin + direction * distance * float(frac)
        if polygon.contains(ShapelyPoint(float(p[0]), float(p[1]))):
            return p
    c = np.array(_polygon_anchor(polygon), dtype=float)
    return origin * 0.35 + c * 0.65


def _nudge_inside(point, polygon, toward=None):
    p = np.asarray(point, dtype=float)
    if polygon.contains(ShapelyPoint(float(p[0]), float(p[1]))):
        return p
    if toward is None:
        toward = np.array(_polygon_anchor(polygon), dtype=float)
    else:
        toward = np.asarray(toward, dtype=float)
    for frac in np.linspace(0.85, 0.05, 18):
        q = p * frac + toward * (1.0 - frac)
        if polygon.contains(ShapelyPoint(float(q[0]), float(q[1]))):
            return q
    return toward


def _group_kind(group, rType):
    kinds = [int(rType[i]) for i in group]
    if any(t in BALCONY_TYPES for t in kinds):
        return "balcony"
    service = sum(1 for t in kinds if t in SERVICE_TYPES)
    private = sum(1 for t in kinds if t in PRIVATE_TYPES)
    if service > private:
        return "service"
    if private > 0:
        return "private"
    return "neutral"


def _ordered_group_nodes(group, rType, R, kind):
    order = _circular_order(R, list(group))
    if kind == "service":
        pref = {2: 0, 3: 1, 10: 2, 11: 3}
        return sorted(order, key=lambda i: (pref.get(int(rType[i]), 9), i))
    if kind == "private":
        pref = {7: 0, 5: 1, 6: 2, 8: 3, 1: 4}
        return sorted(order, key=lambda i: (pref.get(int(rType[i]), 9), i))
    return order


def _orient_perimeter_order(order, rType):
    if len(order) < 2:
        return order
    bad_start_types = {3, 11}  # bathroom/storage rarely occupy the first door-side bay
    first_bad = int(rType[order[0]]) in bad_start_types
    last_bad = int(rType[order[-1]]) in bad_start_types
    if first_bad and not last_bad:
        order = list(reversed(order))
        print(f"  perimeter order flipped away from service-first: {order}")
    return order


def topology_layout_prior(R, areas, polygon, rType, boundary=None,
                          door_info=None):
    """
    Topology-only spatial prior.

    Uses only:
      - outer polygon
      - room type
      - adjacency graph
      - target areas

    Main heuristic:
      1. Living room is placed on the functional middle band. If the boundary
         labels expose a usable opening, it is softly pulled toward that opening.
      2. Removing living room splits the graph into functional groups. Private
         groups go farther from the door along the door tangent; service groups
         go nearer to the door on the opposite tangent side.
      3. Within each group, rooms spread along the local tangent and adjacency
         order. The force simulation then refines with relation attraction.
    """
    if not USE_TOPOLOGY_LAYOUT_PRIOR:
        return None
    n = len(areas)
    if n == 0:
        return None
    living_idx = _pick_living_idx(rType, areas, R)

    poly_anchor = np.array(_polygon_anchor(polygon), dtype=float)
    door_info, door_mid, inward, tangent = _door_axes(
        polygon, boundary, door_info_override=door_info
    )
    avg_r = float(np.mean([np.sqrt(max(a, 1.0) / np.pi) for a in areas]))
    living_r = float(np.sqrt(max(areas[living_idx], 1.0) / np.pi))

    others = [i for i in range(n) if i != living_idx]
    groups = _connected_groups(R, others)
    if not groups:
        groups = [[i] for i in others]

    minx, miny, maxx, maxy = polygon.bounds
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    group_kinds = [_group_kind(g, rType) for g in groups]
    has_private = "private" in group_kinds
    has_service = "service" in group_kinds

    if height >= width * 0.85 and has_private and has_service:
        living_pos = np.array([poly_anchor[0], miny + height * 0.58],
                              dtype=float)
        living_pos = _nudge_inside(living_pos, polygon, toward=poly_anchor)
    elif door_info is not None:
        door_in = _inside_along(door_mid, inward,
                                max(living_r * 0.90, avg_r * 1.15),
                                polygon)
        living_pos = poly_anchor * 0.45 + door_in * 0.55
        living_pos = _nudge_inside(living_pos, polygon, toward=poly_anchor)
    else:
        living_pos = poly_anchor

    positions = [None] * n
    positions[living_idx] = (float(living_pos[0]), float(living_pos[1]))

    step_probe = max(avg_r * 1.8, np.sqrt(max(polygon.area, 1.0)) * 0.32)

    # Use the dominant axis because boundary opening labels are not reliable.
    if height >= width * 0.85:
        private_dir = np.array([0.0, -1.0])
    else:
        cand_left = _inside_along(living_pos, np.array([-1.0, 0.0]),
                                  step_probe, polygon)
        cand_right = _inside_along(living_pos, np.array([1.0, 0.0]),
                                   step_probe, polygon)
        if door_info is not None:
            dl = float(np.linalg.norm(cand_left - door_mid))
            dr = float(np.linalg.norm(cand_right - door_mid))
            private_dir = np.array([-1.0, 0.0]) if dl >= dr else np.array([1.0, 0.0])
        else:
            private_dir = np.array([-1.0, 0.0])
    service_dir = -private_dir

    neutral_dirs = [
        inward,
        -inward,
        _unit(private_dir + inward, fallback=private_dir),
        _unit(service_dir + inward, fallback=service_dir),
    ]
    kind_counts = defaultdict(int)

    anchors = {}
    protrusions = split_main_protrusions(polygon)
    protrusion_rooms = []
    if protrusions:
        cands = [i for i in others if int(rType[i]) in (BALCONY_TYPES | {3, 11})]
        cands = sorted(cands, key=lambda i: (0 if int(rType[i]) in BALCONY_TYPES else 1,
                                             areas[i]))
        for room_idx, (anc, _p_area, _p_r) in zip(cands, protrusions):
            positions[room_idx] = anc
            anchors[room_idx] = anc
            protrusion_rooms.append(room_idx)

    group_summaries = []
    for group in groups:
        group = [i for i in group if i not in protrusion_rooms]
        if not group:
            continue
        kind = _group_kind(group, rType)
        seq = kind_counts[kind]
        kind_counts[kind] += 1

        if kind == "private":
            direction = private_dir
        elif kind == "service":
            direction = service_dir
        elif kind == "balcony":
            direction = inward
        else:
            direction = neutral_dirs[seq % len(neutral_dirs)]

        group_area = float(sum(areas[i] for i in group))
        group_r = float(np.sqrt(max(group_area, 1.0) / np.pi))
        distance = max(living_r * 0.80 + group_r * 0.75,
                       avg_r * (1.6 + 0.18 * seq))
        center = _inside_along(living_pos, direction, distance, polygon)

        axis = np.array([-direction[1], direction[0]], dtype=float)
        if abs(axis[0]) >= abs(axis[1]):
            if axis[0] < 0:
                axis = -axis
        elif axis[1] < 0:
            axis = -axis
        axis = _unit(axis)

        ordered = _ordered_group_nodes(group, rType, R, kind)
        spacing = max(avg_r * 0.95,
                      np.mean([np.sqrt(areas[i] / np.pi) for i in ordered]) * 1.25)
        mid = (len(ordered) - 1) / 2.0
        for k, room_idx in enumerate(ordered):
            raw = center + axis * ((k - mid) * spacing)
            q = _nudge_inside(raw, polygon, toward=center)
            positions[room_idx] = (float(q[0]), float(q[1]))
        group_summaries.append((kind, ordered))

    for i in range(n):
        if positions[i] is None:
            angle = 2.0 * np.pi * (i + 1) / max(n, 1)
            direction = np.array([np.cos(angle), np.sin(angle)])
            q = _inside_along(living_pos, direction, avg_r * 1.8, polygon)
            positions[i] = (float(q[0]), float(q[1]))

    print(f"  topology prior: living={living_idx}, groups={group_summaries}, "
          f"hard_anchors={list(anchors.keys())}")
    return positions, living_idx, anchors


def graph_layout_prior(R, areas, polygon, rType, boundary=None,
                       door_info=None, room_order=None):
    """
    位置先验 (基于"周长弧 -> 内部领地重心"的几何分割):

    设计目标 (按优先级):
      P1. 客厅 (LR) 必须挨着入口门 → LR 在周长上分到的弧 = 以门为中点的弧.
      P2. 突出的小矩形部分由非客厅小房间占据 (并锚定在突出 polylabel).
      P3. 其余非客厅"周长房间"沿剩余周长按邻接序排开, 各自分得一段弧.
      P4. 把 polygon 内部当成 1 个 2D 体积, 每个采样点根据"它最近的边界点
          落在哪段弧"分给对应房间 → 每个房间得到一块"领地".
          种子位置 = 这块领地的重心 (centroid).
          这样:
            - LR 的种子在"门正前方那块领地的中心" → 区域生长自然连门 + 居中
            - 每个周长房间的种子在自己那块楔形/扇形领地中心 → 不会
              出现长条形或断续的奇怪生长形状
            - 突出房间的种子直接用突出 polylabel (锚定不变)

    返回 (positions, living_idx, anchors_dict).
      anchors_dict 包含 LR + 所有突出房间 + 所有周长房间的最终位置. 力学
      模拟阶段把它们都作为"软锚点"使用 (允许微调防重叠, 但不会大幅偏离).
    """
    n = len(areas)
    if n == 0:
        return [], -1, {}
    cx, cy = _polygon_anchor(polygon)
    if n == 1:
        return [(cx, cy)], 0, {0: (cx, cy)}

    living_idx = _pick_living_idx(rType, areas, R)
    others = [i for i in range(n) if i != living_idx]
    perimeter = float(polygon.exterior.length)
    total_area = float(sum(areas))

    if door_info is None and boundary is not None:
        door_info = find_entry_door(boundary)
    if door_info is not None:
        door_mid = door_info[2]
        door_dlen = float(door_info[3])
        door_dist = float(polygon.exterior.project(ShapelyPoint(*door_mid)))
        print(f"  入口门: 中点={door_mid}, 长度={door_dlen:.1f}, "
              f"周长投影={door_dist:.1f}")
    else:
        door_mid = None
        door_dlen = 0.0
        door_dist = 0.0
        print("  [警告] 未检测到入口门, LR 弧默认置 0.")

    protrusions = split_main_protrusions(polygon)
    anchors = {}
    used_for_protrusion = []
    skipped_door_protrusion = 0
    protrusion_areas_by_room = {}
    if protrusions:
        small_pref = {9, 3, 11}
        cand_priority = sorted(
            others,
            key=lambda i: (
                0 if int(rType[i]) in small_pref else 1,
                areas[i],
            ),
        )
        protrusions_filtered = []
        for (anc, p_area, p_r) in protrusions:
            if door_mid is not None:
                d2door = float(np.hypot(anc[0] - door_mid[0],
                                         anc[1] - door_mid[1]))
                if d2door < p_r * 1.6:
                    skipped_door_protrusion += 1
                    continue
            protrusions_filtered.append((anc, p_area, p_r))
        for k, (anc, p_area, p_r) in enumerate(protrusions_filtered):
            if k >= len(cand_priority):
                break
            room_idx = cand_priority[k]
            anchors[room_idx] = anc
            used_for_protrusion.append(room_idx)
            protrusion_areas_by_room[room_idx] = p_area
        if used_for_protrusion or skipped_door_protrusion:
            print(f"  突出部分: 总 {len(protrusions)}, "
                  f"含门跳过 {skipped_door_protrusion}, "
                  f"锚定房间 {used_for_protrusion}")

    # The living room owns the center and door exclusion arc, not a perimeter arc.
    perimeter_rooms = [i for i in others if i not in anchors]

    avg_r = float(np.mean([np.sqrt(a / np.pi) for a in areas]))

    if door_info is not None:
        door_arc_half = door_dlen / 2.0 + avg_r * 1.0
        door_arc_half = min(door_arc_half, perimeter * 0.125)
    else:
        door_arc_half = 0.0
    door_arc_start = (door_dist - door_arc_half) % perimeter
    door_arc_end = (door_dist + door_arc_half) % perimeter
    excl_total = 2.0 * door_arc_half

    arc_assignments = {}
    if perimeter_rooms:
        order = None
        if USE_ROOM_ORDER_PRIOR and room_order is not None:
            parsed = []
            for v in list(room_order):
                idx = int(v) - 1
                if idx in perimeter_rooms and idx not in parsed:
                    parsed.append(idx)
            if len(parsed) == len(perimeter_rooms):
                order = parsed
                print(f"  room order prior: {order}")
        if order is None:
            order = _circular_order(R, perimeter_rooms)
        order = _orient_perimeter_order(order, rType)
        peri_areas = np.array([areas[i] for i in order], dtype=float)
        frac = peri_areas / max(peri_areas.sum(), 1e-6)
        avail = max(perimeter - excl_total, 1.0)
        cur = door_arc_end
        for k, i in enumerate(order):
            seg_len = avail * float(frac[k])
            arc_assignments[i] = (cur % perimeter,
                                   (cur + seg_len) % perimeter)
            cur += seg_len

    minx, miny, maxx, maxy = polygon.bounds
    rng = np.random.RandomState(42)
    samples = []
    tries = 0
    n_target = PRIOR_N_SAMPLES
    while len(samples) < n_target and tries < n_target * 12:
        tries += 1
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        if polygon.contains(ShapelyPoint(x, y)):
            samples.append((x, y))
    print(f"  内部采样: {len(samples)} 点 (目标 {n_target})")

    excluded_proto = []
    if protrusions:
        for room_idx, p_area in protrusion_areas_by_room.items():
            anc = anchors[room_idx]
            excluded_proto.append(
                (np.array(anc, dtype=float),
                 float(np.sqrt(p_area / np.pi)) * 1.10)
            )

    centroids = {}
    counts = {}
    rejected_proto = 0
    rejected_arc = 0
    rejected_door = 0
    ext = polygon.exterior
    for (x, y) in samples:
        skip = False
        if excluded_proto:
            for (anc_arr, r_lim) in excluded_proto:
                if (x - anc_arr[0]) ** 2 + (y - anc_arr[1]) ** 2 < r_lim * r_lim:
                    skip = True
                    break
        if skip:
            rejected_proto += 1
            continue
        sp = ShapelyPoint(x, y)
        proj = float(ext.project(sp))
        if door_arc_half > 0 and _arc_contains(proj, door_arc_start,
                                                  door_arc_end, perimeter):
            rejected_door += 1
            continue
        owner = None
        for room_idx, (arc_s, arc_e) in arc_assignments.items():
            if _arc_contains(proj, arc_s, arc_e, perimeter):
                owner = room_idx
                break
        if owner is None:
            rejected_arc += 1
            continue
        if owner not in centroids:
            centroids[owner] = np.zeros(2, dtype=float)
            counts[owner] = 0
        centroids[owner][0] += x
        centroids[owner][1] += y
        counts[owner] += 1
    print(f"  采样分组: 进入 perimeter 领地 {sum(counts.values())}, "
          f"突出排除 {rejected_proto}, 门弧排除 {rejected_door}, "
          f"弧未匹配 {rejected_arc}")

    positions = [None] * n

    positions[living_idx] = (cx, cy)

    hard_anchors = dict(anchors)
    for r_idx in list(hard_anchors.keys()):
        positions[r_idx] = hard_anchors[r_idx]

    ALPHA = EDGE_ROOM_ANCHOR_ALPHA if EDGE_CROWDING_ENABLED else 0.45
    corner_candidates = (
        _polygon_corner_candidates(
            polygon, (cx, cy), door_mid=door_mid, min_door_dist=avg_r * 1.25
        )
        if EDGE_CROWDING_ENABLED else []
    )
    used_corners = set()
    corner_hits = []
    for r_idx in perimeter_rooms:
        arc_s, arc_e = arc_assignments[r_idx]
        corner = _corner_for_arc(
            corner_candidates, arc_s, arc_e, perimeter, used_corners
        )
        if corner is not None:
            used_corners.add(corner["idx"])
            corner_hits.append((r_idx, corner["idx"]))
            bp_x, bp_y = corner["point"]
            bp = ShapelyPoint(bp_x, bp_y)
        else:
            if arc_s <= arc_e:
                mid = (arc_s + arc_e) / 2.0
            else:
                mid = ((arc_s + arc_e + perimeter) / 2.0) % perimeter
            bp = polygon.exterior.interpolate(mid)
        r_room = float(np.sqrt(areas[r_idx] / np.pi))
        vx, vy = cx - bp.x, cy - bp.y
        nrm = float(np.hypot(vx, vy)) + 1e-6
        inset_frac = _edge_inset_frac(rType[r_idx]) if EDGE_CROWDING_ENABLED else 1.0
        if corner is not None:
            inset_frac *= EDGE_CORNER_INSET_SCALE
        bound_pos = (
            float(bp.x + vx / nrm * r_room * inset_frac),
            float(bp.y + vy / nrm * r_room * inset_frac),
        )
        if r_idx in centroids and counts[r_idx] >= 5:
            cen = (
                float(centroids[r_idx][0] / counts[r_idx]),
                float(centroids[r_idx][1] / counts[r_idx]),
            )
            positions[r_idx] = (
                ALPHA * cen[0] + (1 - ALPHA) * bound_pos[0],
                ALPHA * cen[1] + (1 - ALPHA) * bound_pos[1],
            )
        else:
            positions[r_idx] = bound_pos
    if corner_hits:
        print(f"  edge corner seeds: {corner_hits}")

    final = []
    for i, p in enumerate(positions):
        if p is None or any(np.isnan(p)):
            p = (cx, cy)
        x, y = p
        sp = ShapelyPoint(x, y)
        if not polygon.contains(sp):
            t = ext.project(sp)
            bp = ext.interpolate(t)
            vx, vy = cx - bp.x, cy - bp.y
            nrm = float(np.hypot(vx, vy)) + 1e-6
            x = float(bp.x + vx / nrm * avg_r * 0.4)
            y = float(bp.y + vy / nrm * avg_r * 0.4)
            if i in hard_anchors:
                hard_anchors[i] = (x, y)
        final.append((float(x), float(y)))
    return final, living_idx, hard_anchors


def _room_polygon_from_boundary(poly):
    arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    shp = ShapelyPolygon(arr)
    if not shp.is_valid:
        shp = shp.buffer(0)
    if shp.is_empty or shp.area <= 1.0:
        return None
    if hasattr(shp, "geoms"):
        shp = max(shp.geoms, key=lambda g: g.area)
    return shp


def annotation_layout_prior(ann, polygon, R):
    """
    Use gtBoxNew/rBoundary as the first-class spatial prior.

    The graph-only prior can satisfy rough adjacency, but it has no knowledge of
    which side of the plan each room belongs to. RPLAN annotations already carry
    that spatial order, so use them to initialize the force points and keep them
    anchored unless a sample lacks usable room boundaries.
    """
    if not USE_ANNOTATION_LAYOUT_PRIOR:
        return None
    if "rType" not in ann or "gtBoxNew" not in ann:
        return None

    rType = ann["rType"]
    gt = np.asarray(ann["gtBoxNew"], dtype=np.float64)
    if gt.ndim != 2 or gt.shape[0] != len(rType) or gt.shape[1] < 4:
        return None

    room_polys = ann.get("rBoundary")
    positions = []
    areas = []
    used_polys = 0
    bcx, bcy = _polygon_anchor(polygon)

    for i in range(len(rType)):
        x0, y0, x1, y1 = [float(v) for v in gt[i, :4]]
        bbox_area = max((x1 - x0) * (y1 - y0), 50.0)
        pos = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        area = bbox_area

        shp = None
        if room_polys is not None and i < len(room_polys):
            shp = _room_polygon_from_boundary(room_polys[i])
        if shp is not None:
            used_polys += 1
            area = max(float(shp.area), 50.0)
            rp = shp.representative_point()
            pos = (float(rp.x), float(rp.y))

        sp = ShapelyPoint(*pos)
        if not polygon.contains(sp):
            t = polygon.exterior.project(sp)
            bp = polygon.exterior.interpolate(t)
            vx, vy = bcx - bp.x, bcy - bp.y
            nrm = float(np.hypot(vx, vy)) + 1e-6
            pos = (float(bp.x + vx / nrm * 2.0),
                   float(bp.y + vy / nrm * 2.0))

        positions.append(pos)
        areas.append(area)

    living_idx = _pick_living_idx(rType, areas, R)
    anchors = {}
    if ANNOTATION_ANCHOR_ALL_ROOMS:
        anchors = {i: positions[i] for i in range(len(positions))}
    print(f"  annotation spatial prior: rooms={len(positions)}, "
          f"rBoundary={used_polys}, anchored={len(anchors)}")
    return positions, living_idx, anchors, areas


def init_points(ann, polygon, R, rng=None, cmap=None, affine=None):
    """
    初始化力学点 + 选出客厅 (中心枢纽) + 锚点字典.
      - 面积/质量/半径 = gtBoxNew 矩形面积
      - 初始位置 = graph_layout_prior (客厅居中 + 突出部分小房间 + 其他贴边)
      - anchors[i] = (x, y) 力学模拟时该点冻结在此
    返回 (points, living_idx, anchors).
    """
    rType = ann["rType"]
    gt = ann["gtBoxNew"]
    boundary = ann.get("boundary")
    room_order = ann.get("order")
    image_door_info = find_image_entry_door(cmap, affine) if cmap is not None else None
    if image_door_info is not None:
        print(f"  image door: mid=({image_door_info[2][0]:.1f},"
              f"{image_door_info[2][1]:.1f}), len={image_door_info[3]:.1f}")

    ann_prior = annotation_layout_prior(ann, polygon, R)
    if ann_prior is not None:
        prior_positions, living_idx, anchors, areas = ann_prior
    else:
        areas = []
        for i in range(len(rType)):
            x0, y0, x1, y1 = [float(v) for v in gt[i]]
            areas.append(max((x1 - x0) * (y1 - y0), 50.0))

        topo_prior = topology_layout_prior(R, areas, polygon, rType,
                                            boundary=boundary,
                                            door_info=image_door_info)
        if topo_prior is not None:
            prior_positions, living_idx, anchors = topo_prior
        else:
            prior_positions, living_idx, anchors = graph_layout_prior(
                R, areas, polygon, rType, boundary=boundary,
                door_info=image_door_info, room_order=room_order
            )

    pts = []
    for i, t in enumerate(rType):
        rname, color = RTYPE_INFO.get(int(t), ("Unknown", (200, 200, 200)))
        pts.append(Pt(i, int(t), color, prior_positions[i], areas[i]))
    return pts, living_idx, anchors


def force_simulate(points, R, polygon, iters=SIM_ITERATIONS,
                   living_idx=-1, anchors=None, verbose=True):
    """
    轻量力学精炼:

      由 graph_layout_prior 给出的种子位置已经在每个房间几何领地的重心,
      力学只负责:
        1. 防重叠: 任意两个圆距离 < (r_i + r_j) × 0.95 时排斥;
        2. 软锚回归: 每个非硬锚的点被弹簧拉回它的"先验位置"
           (init_pos), 防止重叠排斥推得太远;
        3. 阻尼;
        4. 出 polygon 后强力拉回主体最深点.

    硬锚 (anchors[i]=(x,y)) = 突出部分被分配的小房间, 它们的位置完全
    冻结, 不参与积分. (LR 和其他周长房间是软锚, 允许微调.)
    """
    bounds = polygon.bounds
    cx, cy = _polygon_anchor(polygon)
    n = len(points)
    if n == 0:
        return []

    if anchors is None:
        anchors = {}

    init_pos = [p.pos.copy() for p in points]

    for idx, anc in anchors.items():
        points[idx].pos = np.array([anc[0], anc[1]], dtype=float)
        points[idx].vel = np.zeros(2, dtype=float)
        points[idx].acc = np.zeros(2, dtype=float)
        init_pos[idx] = points[idx].pos.copy()

    history = []
    diag_size = float(np.sqrt(polygon.area))
    ext = polygon.exterior

    for it in range(iters):
        for i in range(n):
            for j in range(i + 1, n):
                delta = points[j].pos - points[i].pos
                dist = float(np.linalg.norm(delta)) + 1e-6
                touch = points[i].radius + points[j].radius
                safe = touch * 0.95
                rel = float(R[i, j])
                if rel > 0:
                    desired = touch * 0.92
                    if dist > desired:
                        dirv = delta / dist
                        f = REL_ATTRACTION_GAIN * rel * (dist - desired)
                        points[i].acc += f * dirv
                        points[j].acc -= f * dirv
                if dist < safe:
                    dirv = delta / dist
                    f = REPULSION_GAIN * (safe - dist)
                    points[i].acc -= f * dirv
                    points[j].acc += f * dirv

        for idx, p in enumerate(points):
            if idx in anchors:
                p.acc = np.zeros(2, dtype=float)
                continue

            d_anc = np.array(init_pos[idx]) - p.pos
            p.acc += SOFT_ANCHOR_GAIN * d_anc

            if EDGE_CROWDING_ENABLED and idx != living_idx:
                sp_edge = ShapelyPoint(float(p.pos[0]), float(p.pos[1]))
                if polygon.contains(sp_edge):
                    t_edge = ext.project(sp_edge)
                    bp_edge = ext.interpolate(t_edge)
                    inward_vec = p.pos - np.array([bp_edge.x, bp_edge.y], dtype=float)
                    inward_len = float(np.linalg.norm(inward_vec))
                    if inward_len > 1e-6:
                        desired = max(1.0, p.radius * _edge_inset_frac(p.rtype))
                        p.acc += (
                            EDGE_ATTRACTION_GAIN
                            * (desired - inward_len)
                            * inward_vec / inward_len
                        )

            sp = ShapelyPoint(p.pos[0], p.pos[1])
            if not polygon.contains(sp):
                back = np.array([cx - p.pos[0], cy - p.pos[1]])
                dn = float(np.linalg.norm(back)) + 1e-6
                p.acc += 2.0 * back / dn * diag_size * 0.05

            p.acc -= DAMPING * p.vel

        for idx, p in enumerate(points):
            if idx in anchors:
                continue
            p.step(SIM_DT, SIM_MAX_VEL)
            p.pos[0] = np.clip(p.pos[0],
                               bounds[0] + p.radius * 0.2,
                               bounds[2] - p.radius * 0.2)
            p.pos[1] = np.clip(p.pos[1],
                               bounds[1] + p.radius * 0.2,
                               bounds[3] - p.radius * 0.2)

        if it in (0, iters // 4, iters // 2, 3 * iters // 4, iters - 1):
            history.append((it, [p.pos.copy() for p in points]))
            if verbose:
                print(f"  iter={it:3d}/{iters}")

    for idx, anc in anchors.items():
        points[idx].pos = np.array([anc[0], anc[1]], dtype=float)

    for idx, p in enumerate(points):
        sp = ShapelyPoint(p.pos[0], p.pos[1])
        if not polygon.contains(sp):
            t = polygon.exterior.project(sp)
            bp = polygon.exterior.interpolate(t)
            vx = cx - bp.x
            vy = cy - bp.y
            nrm = float(np.hypot(vx, vy)) + 1e-6
            p.pos = np.array([
                bp.x + vx / nrm * p.radius * 0.4,
                bp.y + vy / nrm * p.radius * 0.4,
            ], dtype=float)

    return history


def prepare_image_canvas(img, ann):
    """
    用源图像构造画布 + 像素分类:
      - 像素分类得到 cmap (-2 白, -1 墙, -4 窗, -3 其他)
      - 内/外墙拆分: 外墙保持 -1; 内墙改为 -5 (待填)
      - 把"其他像素 (-3, 包含彩色 + 抗锯齿)" 也并入待填 (覆盖原有色块)
      => 最终 cmap:
         -2 白 (固定), -1 外墙 (固定), -4 窗 (固定),
         -3 / -5 → 都是 "可生长", 后续合并成 -3
    返回: cmap, inner_wall_color, inner_wall_thickness
    """
    cmap = classify_pixels(img)
    cmap, n_outer, n_inner, inner_thick, inner_mask = split_outer_inner_walls(cmap)
    inner_color = sample_inner_wall_color(img, inner_mask)
    print(f"  内墙原色采样 = {tuple(int(v) for v in inner_color)}")

    growable = (cmap == -3) | (cmap == -5)
    if inner_mask is not None and inner_mask.any():
        growable |= inner_mask
    cmap_new = cmap.copy()
    cmap_new[growable] = -3
    growable_total = int(growable.sum())
    print(f"  可生长像素总数 = {growable_total}")
    return cmap_new, inner_color, inner_thick, growable_total


def _fill_unassigned_from_regions(region_map, unassigned):
    h, w = region_map.shape
    dirs4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    bfs = deque()

    ys, xs = np.where(region_map >= 0)
    for y, x in zip(ys, xs):
        rid = int(region_map[y, x])
        for dx, dy in dirs4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                bfs.append((x, y, rid))
                break

    while bfs:
        x, y, rid = bfs.popleft()
        for dx, dy in dirs4:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                region_map[ny, nx] = rid
                unassigned[ny, nx] = 0
                bfs.append((nx, ny, rid))


def build_annotation_region_map(ann, cmap, affine, points):
    """
    Rasterize rBoundary into the current image space, then fill swallowed inner
    walls from the nearest annotated room. This preserves the RPLAN topology
    instead of asking random growth to rediscover the original room order.
    """
    if not USE_ANNOTATION_REGION_MAP:
        return None
    room_polys = ann.get("rBoundary")
    if room_polys is None or len(room_polys) < len(points):
        return None

    h, w = cmap.shape
    fixed_mask = (cmap == -1) | (cmap == -2) | (cmap == -4)
    growable = (cmap == -3) & (~fixed_mask)

    region_map = np.full((h, w), -99, dtype=np.int32)
    region_map[cmap == -1] = -1
    region_map[cmap == -2] = -2
    region_map[cmap == -4] = -4

    n = len(points)
    filled = [0] * n
    for i in range(n):
        arr = np.asarray(room_polys[i], dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 3:
            continue
        pix = []
        for q in arr:
            px, py = boundary_to_pixel(q, affine)
            pix.append([int(round(px)), int(round(py))])
        poly_px = np.asarray(pix, dtype=np.int32)
        if poly_px.shape[0] < 3:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_px], 255)
        room_mask = (mask > 0) & growable
        filled[i] = int(room_mask.sum())
        region_map[room_mask] = i

    assigned_total = int(sum(filled))
    if assigned_total == 0:
        return None

    unassigned = growable & (region_map < 0)
    remaining = int(unassigned.sum())
    if remaining > 0:
        _fill_unassigned_from_regions(region_map, unassigned)

    still = int(unassigned.sum())
    if still > 0:
        seed_xy = []
        for p in points:
            px, py = boundary_to_pixel(p.pos, affine)
            seed_xy.append((float(px), float(py)))
        seed_xy = np.asarray(seed_xy, dtype=np.float32)
        ys, xs = np.where(unassigned)
        for y, x in zip(ys, xs):
            d = (seed_xy[:, 0] - x) ** 2 + (seed_xy[:, 1] - y) ** 2
            rid = int(np.argmin(d))
            region_map[y, x] = rid
            unassigned[y, x] = 0

    assigned = [int((region_map == i).sum()) for i in range(n)]
    print(f"  annotation region map: polygon_px={assigned_total}, "
          f"filled_extra={remaining - int(unassigned.sum())}, "
          f"still_unassigned={int(unassigned.sum())}")
    return region_map, assigned


def region_grow(points, cmap, areas_target, affine,
                living_idx=-1, door_corridor=None):
    """
    区域生长 (在图像像素空间):
      - 仅在 cmap == -3 上扩散
      - 不可越过外墙 -1 / 背景 -2 / 窗 -4
    种子位置由 256-空间通过 affine 映射到图像空间.

    door_corridor: list of (x, y) 像素, 这些像素在生长前就硬分配给客厅
                   (living_idx). 这样无论其他房间怎么长, 都跨不进门口
                   走廊 → LR 必定连着门.
    """
    h, w = cmap.shape
    n = len(points)

    fixed_mask = (cmap == -1) | (cmap == -2) | (cmap == -4)
    growable = (cmap == -3) & (~fixed_mask)

    region_map = np.full((h, w), -99, dtype=np.int32)
    region_map[cmap == -1] = -1
    region_map[cmap == -2] = -2
    region_map[cmap == -4] = -4

    unassigned = growable.astype(np.uint8)

    DIRS4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    fronts = [deque() for _ in range(n)]
    assigned = [0] * n

    door_pre = 0
    if door_corridor and 0 <= living_idx < n:
        for (x, y) in door_corridor:
            if not (0 <= x < w and 0 <= y < h):
                continue
            if unassigned[y, x] == 0:
                continue
            region_map[y, x] = living_idx
            unassigned[y, x] = 0
            assigned[living_idx] += 1
            door_pre += 1
            for dx, dy in DIRS4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                    fronts[living_idx].append((nx, ny))
        if door_pre > 0:
            print(f"  门口走廊预分配给客厅 (idx={living_idx}): {door_pre} px")

    seeds_xy = []
    for p in points:
        px, py = boundary_to_pixel(p.pos, affine)
        sx = int(round(px))
        sy = int(round(py))
        sx = int(np.clip(sx, 0, w - 1))
        sy = int(np.clip(sy, 0, h - 1))
        if not unassigned[sy, sx]:
            ys, xs = np.where(unassigned == 1)
            if len(ys) == 0:
                seeds_xy.append((sx, sy))
                continue
            d2 = (ys - sy) ** 2 + (xs - sx) ** 2
            k = int(np.argmin(d2))
            sx, sy = int(xs[k]), int(ys[k])
        seeds_xy.append((sx, sy))

    for i, (sx, sy) in enumerate(seeds_xy):
        if unassigned[sy, sx]:
            region_map[sy, sx] = i
            unassigned[sy, sx] = 0
            assigned[i] += 1
            for dx, dy in DIRS4:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                    fronts[i].append((nx, ny))

    active = set(range(n))
    while active:
        order = list(active)
        random.shuffle(order)
        for i in order:
            if assigned[i] >= areas_target[i] or not fronts[i]:
                if assigned[i] >= areas_target[i] or not fronts[i]:
                    active.discard(i)
                continue
            steps = min(random.randint(3, 12), areas_target[i] - assigned[i])
            for _ in range(steps):
                if not fronts[i]:
                    break
                if len(fronts[i]) > 1 and random.random() < 0.4:
                    pick = random.randint(0, len(fronts[i]) - 1)
                    x, y = fronts[i][pick]
                    del fronts[i][pick]
                else:
                    x, y = fronts[i].popleft()
                if not (0 <= x < w and 0 <= y < h):
                    continue
                if not unassigned[y, x]:
                    continue
                region_map[y, x] = i
                unassigned[y, x] = 0
                assigned[i] += 1
                dirs = list(DIRS4)
                random.shuffle(dirs)
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                        fronts[i].append((nx, ny))
                if assigned[i] >= areas_target[i]:
                    break

    # Multi-source BFS fills gaps while preserving contiguous room regions.
    remaining = int(unassigned.sum())
    if remaining > 0:
        print(f"  剩余未分配 {remaining} 像素 → BFS 连续填补")
        bfs = deque()
        for y in range(h):
            for x in range(w):
                if unassigned[y, x]:
                    continue
                rid = region_map[y, x]
                if rid < 0:
                    continue
                for dx, dy in DIRS4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                        bfs.append((x, y, rid))
                        break
        while bfs:
            x, y, rid = bfs.popleft()
            for dx, dy in DIRS4:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and unassigned[ny, nx]:
                    region_map[ny, nx] = rid
                    unassigned[ny, nx] = 0
                    assigned[rid] += 1
                    bfs.append((nx, ny, rid))

        still = int(unassigned.sum())
        if still > 0:
            # Isolated pockets fall back to their nearest seed.
            print(f"  仍有 {still} 像素孤立 → 按最近种子距离分配")
            coords = list(zip(*np.where(unassigned == 1)))
            seed_xy_arr = np.array(seeds_xy, dtype=np.float32)  # (n, 2) (x, y)
            for (y, x) in coords:
                d = (seed_xy_arr[:, 0] - x) ** 2 + (seed_xy_arr[:, 1] - y) ** 2
                rid = int(np.argmin(d))
                region_map[y, x] = rid
                unassigned[y, x] = 0
                assigned[rid] += 1

    return region_map, assigned


def compute_region_contacts(region_map, n_rooms):
    contacts = Counter()
    a = region_map[:, :-1]
    b = region_map[:, 1:]
    mask = (a >= 0) & (b >= 0) & (a != b)
    for u, v in zip(a[mask].tolist(), b[mask].tolist()):
        contacts[tuple(sorted((int(u), int(v))))] += 1

    a = region_map[:-1, :]
    b = region_map[1:, :]
    mask = (a >= 0) & (b >= 0) & (a != b)
    for u, v in zip(a[mask].tolist(), b[mask].tolist()):
        contacts[tuple(sorted((int(u), int(v))))] += 1
    return contacts


def door_living_contact_count(region_map, cmap, living_idx):
    if living_idx < 0:
        return 0
    living = (region_map == living_idx)
    door = (cmap == -4)
    count = 0
    count += int((living[:, :-1] & door[:, 1:]).sum())
    count += int((door[:, :-1] & living[:, 1:]).sum())
    count += int((living[:-1, :] & door[1:, :]).sum())
    count += int((door[:-1, :] & living[1:, :]).sum())
    return count


def ensure_living_touches_door(region_map, cmap, living_idx, depth=10):
    if living_idx < 0:
        return region_map
    before = door_living_contact_count(region_map, cmap, living_idx)
    if before > 0:
        return region_map
    door_mask = _select_entry_door_mask(cmap)
    if door_mask is None:
        return region_map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    near = cv2.dilate(door_mask, kernel, iterations=int(depth)) > 0
    editable = (region_map >= 0) & (cmap != -1) & (cmap != -2) & (cmap != -4)
    region_map[near & editable] = living_idx
    return region_map


def _expected_adjacencies(R):
    n = R.shape[0]
    expected = []
    for i in range(n):
        for j in range(i + 1, n):
            if R[i, j] > 0:
                expected.append((i, j))
    return expected


def _missing_adjacencies(region_map, n, expected, min_contact):
    contacts = compute_region_contacts(region_map, n)
    missing = [(i, j) for i, j in expected
               if contacts.get((i, j), 0) < min_contact]
    return missing, contacts


def _closest_region_pair(mask_i, mask_j):
    if not mask_i.any() or not mask_j.any():
        return None
    dist, nearest = ndimage.distance_transform_edt(
        ~mask_i, return_indices=True
    )
    ys, xs = np.where(mask_j)
    if len(ys) == 0:
        return None
    k = int(np.argmin(dist[ys, xs]))
    by, bx = int(ys[k]), int(xs[k])
    ay, ax = int(nearest[0, by, bx]), int(nearest[1, by, bx])
    return (ay, ax), (by, bx)


def _manhattan_path(a, b):
    ay, ax = a
    by, bx = b
    y, x = int(ay), int(ax)
    path = [(y, x)]
    step_x = 1 if bx >= x else -1
    while x != bx:
        x += step_x
        path.append((y, x))
    step_y = 1 if by >= y else -1
    while y != by:
        y += step_y
        path.append((y, x))
    return path


def _path_in_editable_space(mask_i, mask_j, passable, margin=96):
    if not mask_i.any() or not mask_j.any():
        return []

    h, w = passable.shape
    ys_i, xs_i = np.where(mask_i)
    ys_j, xs_j = np.where(mask_j)
    min_y = int(min(ys_i.min(), ys_j.min()))
    max_y = int(max(ys_i.max(), ys_j.max()))
    min_x = int(min(xs_i.min(), xs_j.min()))
    max_x = int(max(xs_i.max(), xs_j.max()))

    bounds = []
    y0 = max(0, min_y - margin)
    y1 = min(h, max_y + margin + 1)
    x0 = max(0, min_x - margin)
    x1 = min(w, max_x + margin + 1)
    bounds.append((y0, y1, x0, x1))
    if (y0, y1, x0, x1) != (0, h, 0, w):
        bounds.append((0, h, 0, w))

    kernel = STRUCT4.astype(np.uint8)
    neigh = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y0, y1, x0, x1 in bounds:
        start = mask_i[y0:y1, x0:x1]
        goal = mask_j[y0:y1, x0:x1]
        open_mask = passable[y0:y1, x0:x1] | start | goal
        if not start.any() or not goal.any():
            continue

        eroded = cv2.erode(start.astype(np.uint8), kernel, iterations=1) > 0
        starts = start & (~eroded)
        if not starts.any():
            starts = start

        hh, ww = start.shape
        visited = np.zeros((hh, ww), dtype=bool)
        prev_y = np.full((hh, ww), -1, dtype=np.int32)
        prev_x = np.full((hh, ww), -1, dtype=np.int32)
        q = deque()

        sy, sx = np.where(starts)
        for yy, xx in zip(sy.tolist(), sx.tolist()):
            visited[yy, xx] = True
            prev_y[yy, xx] = -2
            prev_x[yy, xx] = -2
            q.append((yy, xx))

        end = None
        while q:
            y, x = q.popleft()
            if goal[y, x]:
                end = (y, x)
                break
            for dy, dx in neigh:
                ny = y + dy
                nx = x + dx
                if ny < 0 or ny >= hh or nx < 0 or nx >= ww:
                    continue
                if visited[ny, nx] or not open_mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                prev_y[ny, nx] = y
                prev_x[ny, nx] = x
                q.append((ny, nx))

        if end is None:
            continue

        y, x = end
        path = []
        while True:
            path.append((y + y0, x + x0))
            py = int(prev_y[y, x])
            px = int(prev_x[y, x])
            if py == -2:
                break
            y, x = py, px
        path.reverse()
        return path

    return []


def _paint_two_room_path(region_map, path, i, j, editable, width):
    if len(path) < 2:
        return
    h, w = region_map.shape
    split = max(0, min(len(path) - 2, len(path) // 2 - 1))

    for k, (y, x) in enumerate(path):
        rid = i if k <= split else j
        for yy in range(y - width, y + width + 1):
            if yy < 0 or yy >= h:
                continue
            for xx in range(x - width, x + width + 1):
                if xx < 0 or xx >= w:
                    continue
                if (xx - x) ** 2 + (yy - y) ** 2 > width ** 2:
                    continue
                if editable[yy, xx]:
                    region_map[yy, xx] = rid

    # Ensure at least one explicit 4-neighbor contact survives disk overlap.
    for k in range(max(0, split - 2), min(len(path) - 1, split + 3)):
        y1, x1 = path[k]
        y2, x2 = path[k + 1]
        if abs(y1 - y2) + abs(x1 - x2) != 1:
            continue
        if editable[y1, x1]:
            region_map[y1, x1] = i
        if editable[y2, x2]:
            region_map[y2, x2] = j
        break


def enforce_required_adjacencies(region_map, points, R, affine, cmap,
                                 min_contact=1, width=2, max_passes=4,
                                 max_bridge_len=28):
    """
    Ensure the topology-prior heatmap exposes every required rEdge relation.
    Missing edges are repaired by carving thin two-room paths through editable
    interior pixels, keeping outer walls, background, and the yellow door fixed.
    """
    n = len(points)
    expected = _expected_adjacencies(R)
    missing, _contacts = _missing_adjacencies(
        region_map, n, expected, min_contact
    )
    missing_before = len(missing)
    if not missing:
        assigned = [int((region_map == i).sum()) for i in range(n)]
        return region_map, assigned, {"missing_before": 0,
                                      "missing_after": 0,
                                      "door_contact": 0}

    editable = ((cmap == -3) | (region_map >= 0))
    editable &= (cmap != -1) & (cmap != -2) & (cmap != -4)

    for pass_idx in range(max_passes):
        missing, _contacts = _missing_adjacencies(
            region_map, n, expected, min_contact
        )
        if not missing:
            break

        draw_width = width + pass_idx
        margin = 96 + pass_idx * 48
        for i, j in missing:
            mask_i = (region_map == i)
            mask_j = (region_map == j)
            path = _path_in_editable_space(
                mask_i, mask_j, editable, margin=margin
            )
            if not path:
                pair = _closest_region_pair(mask_i, mask_j)
                if pair is None:
                    ax, ay = boundary_to_pixel(points[i].pos, affine)
                    bx, by = boundary_to_pixel(points[j].pos, affine)
                    pair = ((int(round(ay)), int(round(ax))),
                            (int(round(by)), int(round(bx))))
                path = _manhattan_path(pair[0], pair[1])
            if max_bridge_len is not None and len(path) > max_bridge_len:
                continue
            _paint_two_room_path(region_map, path, i, j, editable, draw_width)

    missing_after, _contacts_after = _missing_adjacencies(
        region_map, n, expected, min_contact
    )
    assigned = [int((region_map == i).sum()) for i in range(n)]
    print(f"  topology contacts: missing {missing_before} -> {len(missing_after)}")
    return region_map, assigned, {"missing_before": missing_before,
                                  "missing_after": len(missing_after),
                                  "door_contact": 0}


def _contact_band(region_map, i, j):
    h, w = region_map.shape
    band = np.zeros((h, w), dtype=bool)

    a = region_map[:, :-1]
    b = region_map[:, 1:]
    mask = ((a == i) & (b == j)) | ((a == j) & (b == i))
    band[:, :-1] |= mask
    band[:, 1:] |= mask

    a = region_map[:-1, :]
    b = region_map[1:, :]
    mask = ((a == i) & (b == j)) | ((a == j) & (b == i))
    band[:-1, :] |= mask
    band[1:, :] |= mask
    return band


def _pick_separator_room(i, j, R, living_idx):
    n = R.shape[0]
    common = [k for k in range(n)
              if k not in (i, j) and R[i, k] > 0 and R[j, k] > 0]
    if living_idx in common:
        return living_idx
    if common:
        return sorted(common, key=lambda k: (-int((R[k] > 0).sum()), k))[0]
    return None


def suppress_extra_adjacencies(region_map, R, cmap, living_idx=-1,
                               min_contact=16, width=2, max_passes=2,
                               use_wall_separator=False):
    """
    Separate visible contacts that are not present in rEdge. A common neighbor
    is painted into the contact band, so the extra i-j edge becomes i-k and j-k
    when both of those are valid topology edges.
    """
    n = R.shape[0]
    expected = _expected_adjacencies(R)
    editable = ((cmap == -3) | (region_map >= 0))
    editable &= (cmap != -1) & (cmap != -2) & (cmap != -4)
    removed_total = 0

    for pass_idx in range(max_passes):
        contacts = compute_region_contacts(region_map, n)
        extras = [(edge, count) for edge, count in contacts.items()
                  if edge not in expected and count >= min_contact]
        if not extras:
            break

        changed = 0
        for (i, j), _count in sorted(extras, key=lambda x: -x[1]):
            sep = _pick_separator_room(i, j, R, living_idx)
            if sep is None:
                continue
            band = _contact_band(region_map, i, j)
            if width > 0:
                ksize = max(1, width * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (ksize, ksize))
                band = cv2.dilate(band.astype(np.uint8), kernel,
                                  iterations=1) > 0
            paint = band & editable & ((region_map == i) | (region_map == j))
            changed += int(paint.sum())
            region_map[paint] = sep

        if changed == 0:
            break
        removed_total += changed

    contacts_after = compute_region_contacts(region_map, n)
    extras_after = [(edge, count) for edge, count in contacts_after.items()
                    if edge not in expected and count >= min_contact]
    if removed_total > 0:
        print(f"  topology extras: painted={removed_total}, "
              f"remaining={len(extras_after)}")
    return region_map, {"extra_after": len(extras_after),
                        "painted": removed_total}


def redraw_inner_walls(heatmap, region_map, color, thickness):
    rm = region_map
    h, w = rm.shape
    boundary = np.zeros((h, w), dtype=bool)
    diff_r = (rm[:, :-1] != rm[:, 1:]) & (rm[:, :-1] >= 0) & (rm[:, 1:] >= 0)
    boundary[:, :-1] |= diff_r
    diff_d = (rm[:-1, :] != rm[1:, :]) & (rm[:-1, :] >= 0) & (rm[1:, :] >= 0)
    boundary[:-1, :] |= diff_d
    if thickness > 1:
        k = max(1, thickness - 1)
        b8 = boundary.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        b8 = cv2.dilate(b8, kernel, iterations=1)
        boundary = b8 > 0
    heatmap[boundary] = color
    return heatmap


def render_heatmap(src_img, region_map, points, cmap, inner_wall_color,
                   inner_wall_thickness):
    """
    输出与源图同分辨率的热度图:
      - 白色背景 (cmap == -2): 直接保留 (255,255,255)
      - 外墙 (cmap == -1): 沿用源图灰色像素 (保护抗锯齿)
      - 窗户 (cmap == -4): 沿用源图亮黄
      - 房间区域: 按 rType 颜色填充
      - 内墙: 在不同 region id 边界处按估算粗细重画
    """
    h, w = src_img.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    outer_wall_mask = (cmap == -1)
    window_mask = (cmap == -4)
    out[outer_wall_mask] = src_img[outer_wall_mask]
    out[window_mask] = src_img[window_mask]

    idx_to_color = {p.idx: np.array(p.color, dtype=np.uint8) for p in points}
    for idx, c in idx_to_color.items():
        out[region_map == idx] = c

    out = cv2.medianBlur(out, 3)
    out[outer_wall_mask] = src_img[outer_wall_mask]
    out[window_mask] = src_img[window_mask]

    inner_thickness = int(round(inner_wall_thickness)) - REDRAW_INNER_WALL_SHRINK
    inner_thickness = max(REDRAW_INNER_WALL_MIN,
                          min(REDRAW_INNER_WALL_MAX, inner_thickness))
    print(f"  内墙重画: 粗细={inner_thickness}px, "
          f"颜色={tuple(int(v) for v in inner_wall_color)}")
    out = redraw_inner_walls(out, region_map, inner_wall_color, inner_thickness)
    out[outer_wall_mask] = src_img[outer_wall_mask]
    out[window_mask] = src_img[window_mask]
    return out


def render_simulation_panel(src_img, polygon, points, R, affine,
                            title="Force Simulation"):
    """
    在源图像尺度下绘制力学模拟结果:
      底图 = 源图变浅 (淡化外墙) + 关系连线 + 房间圆 + 标签
    """
    h, w = src_img.shape[:2]
    img = (src_img.astype(np.float32) * 0.4 + 255.0 * 0.6).astype(np.uint8)

    coords = np.array(list(polygon.exterior.coords), dtype=np.float32)
    px = coords[:, 0] * affine[0] + affine[2]
    py = coords[:, 1] * affine[1] + affine[3]
    pts_int = np.round(np.stack([px, py], axis=1)).astype(np.int32)
    cv2.polylines(img, [pts_int], True, (110, 110, 110), 2)

    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            rel = R[i, j]
            if rel <= 0:
                continue
            ax, ay = boundary_to_pixel(points[i].pos, affine)
            bx, by = boundary_to_pixel(points[j].pos, affine)
            a = (int(round(ax)), int(round(ay)))
            b = (int(round(bx)), int(round(by)))
            if rel >= 1.0:
                cv2.line(img, a, b, (50, 50, 230), 2)
            else:
                steps = 10
                for s in range(steps):
                    if s % 2 == 0:
                        x1 = int(a[0] + (b[0] - a[0]) * s / steps)
                        y1 = int(a[1] + (b[1] - a[1]) * s / steps)
                        x2 = int(a[0] + (b[0] - a[0]) * (s + 1) / steps)
                        y2 = int(a[1] + (b[1] - a[1]) * (s + 1) / steps)
                        cv2.line(img, (x1, y1), (x2, y2), (90, 200, 90), 1)

    avg_scale = 0.5 * (abs(affine[0]) + abs(affine[1]))
    for p in points:
        px_, py_ = boundary_to_pixel(p.pos, affine)
        cx = int(round(px_)); cy = int(round(py_))
        rr = max(4, int(round(p.radius_render * avg_scale)))
        cv2.circle(img, (cx, cy), rr, p.color, -1)
        cv2.circle(img, (cx, cy), rr, (60, 60, 60), 1)
        rname, _ = RTYPE_INFO.get(p.rtype, ("?", None))
        cv2.putText(img, f"{p.idx}:{rname[:4]}", (cx - 22, cy - rr - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)

    cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 40), 2, cv2.LINE_AA)
    return img


def run_one(name=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    chosen = pick_sample(TESTSET_DIR, name=name)
    name = chosen["name"]
    ann = load_annotation(TESTSET_DIR, name)

    rType    = ann["rType"]
    rEdge    = ann["rEdge"]
    boundary = ann["boundary"]
    print(f"\n[ann] rType={rType.tolist()}, n_edges={len(rEdge)}")

    src_path = os.path.join(TESTSET_DIR, "images", f"{name}.png")
    src_img = np.array(Image.open(src_path).convert("RGB"))
    H, W = src_img.shape[:2]
    print(f"\n[源图] {src_path}, {W}x{H}")
    cmap, inner_color, inner_thick, growable_total = prepare_image_canvas(src_img, ann)

    print("\n[对齐] boundary ↔ 图像")
    affine = calibrate_affine(boundary, cmap)

    polygon = extract_sim_polygon(cmap, affine, simplify_eps=1.5)
    if polygon is None or polygon.area <= 0:
        print("[poly] 从 cmap 提取失败, 退回到 boundary 多边形")
        polygon = boundary_to_polygon(boundary)
    else:
        print(f"[poly] 来自 cmap 真实外形, 顶点数 = {len(polygon.exterior.coords)}")
    minx, miny, maxx, maxy = polygon.bounds
    print(f"[poly] bounds=({minx:.1f},{miny:.1f},{maxx:.1f},{maxy:.1f}), "
          f"area={polygon.area:.1f}")

    R = build_relationship_matrix(rType, rEdge)
    n = len(rType)
    n_strong = int((R >= 1.0).sum() / 2)
    n_weak   = int(((R > 0) & (R < 1.0)).sum() / 2)
    print(f"[rel] 强关系={n_strong}, 中/弱={n_weak}")

    print("\n[模拟] 初始化点 (客厅居中 + 突出部分锚定 + 其他贴边界)")
    rng = np.random.default_rng(RANDOM_SEED)
    points, living_idx, anchors = init_points(ann, polygon, R, rng=rng,
                                              cmap=cmap, affine=affine)
    print(f"  living_idx = {living_idx} "
          f"({RTYPE_INFO.get(int(rType[living_idx]), ('?', None))[0]})")
    print(f"  anchors    = {anchors}")
    for p in points:
        rn, _ = RTYPE_INFO.get(p.rtype, ("?", None))
        deg = int((R[p.idx] > 0).sum())
        if p.idx == living_idx:
            tag = "★"
        elif p.idx in anchors:
            tag = "▲"
        else:
            tag = " "
        print(f"  {tag} pt {p.idx}: type={p.rtype}({rn}), deg={deg}, "
              f"pos=({p.pos[0]:.1f},{p.pos[1]:.1f}) [prior], "
              f"area={p.mass:.1f}, r={p.radius:.1f}")

    print("\n[模拟] 力学迭代 ...")
    t0 = time.time()
    _ = force_simulate(points, R, polygon, iters=SIM_ITERATIONS,
                        living_idx=living_idx, anchors=anchors)
    print(f"[模拟] 用时 {time.time() - t0:.1f}s")

    rect_area = np.array([p.mass for p in points], dtype=np.float64)
    targets = (rect_area / rect_area.sum() * growable_total).astype(np.int64)
    diff = growable_total - targets.sum()
    if diff != 0:
        targets[int(np.argmax(rect_area))] += diff
    print(f"\n[生长] 目标像素总和 = {targets.sum()} / 可生长 {growable_total}")

    door_info = find_entry_door(boundary)
    door_corridor = []
    if door_info is not None:
        door_seg = (door_info[0], door_info[1])
        door_corridor = find_door_corridor_pixels(cmap, door_seg, affine, depth=8)
        print(f"  入口门走廊像素: {len(door_corridor)} (depth=8)")
    image_door_corridor = find_door_corridor_pixels(cmap, depth=8)
    if image_door_corridor:
        door_corridor = image_door_corridor
        print(f"  image-door living preassign pixels: {len(door_corridor)} (depth=8)")

    t0 = time.time()
    guided = build_annotation_region_map(ann, cmap, affine, points)
    if guided is not None:
        region_map, assigned = guided
        print(f"[生长] annotation 引导用时 {time.time() - t0:.1f}s")
    else:
        region_map, assigned = region_grow(points, cmap, targets.tolist(), affine,
                                            living_idx=living_idx,
                                            door_corridor=door_corridor)
        print(f"[生长] 区域生长用时 {time.time() - t0:.1f}s")
    region_map = ensure_living_touches_door(region_map, cmap, living_idx, depth=10)
    if ENABLE_ADJACENCY_REPAIR:
        region_map, assigned, topo_stats = enforce_required_adjacencies(
            region_map, points, R, affine, cmap
        )
    else:
        expected = _expected_adjacencies(R)
        missing, _contacts = _missing_adjacencies(
            region_map, len(points), expected, min_contact=1
        )
        assigned = [int((region_map == i).sum()) for i in range(len(points))]
        topo_stats = {"missing_before": len(missing),
                      "missing_after": len(missing),
                      "door_contact": 0}
    region_map = ensure_living_touches_door(region_map, cmap, living_idx, depth=10)
    assigned = [int((region_map == i).sum()) for i in range(len(points))]
    door_contact = door_living_contact_count(region_map, cmap, living_idx)
    print(f"  topology check: missing_edges={topo_stats['missing_after']}, "
          f"living_door_contact={door_contact}")
    for i, p in enumerate(points):
        rn, _ = RTYPE_INFO.get(p.rtype, ("?", None))
        print(f"  pt {i} ({rn}): 目标={targets[i]}, 实际={assigned[i]}, "
              f"差={assigned[i]-targets[i]}")

    print("\n[输出] 渲染热度图 (源图分辨率)")
    heatmap = render_heatmap(src_img, region_map, points, cmap,
                              inner_color, inner_thick)
    sim_panel = render_simulation_panel(src_img, polygon, points, R, affine,
                                         title=f"sim {name}")

    side = np.full((H, 8, 3), 200, dtype=np.uint8)
    src_label = src_img.copy()
    heatmap_label = heatmap.copy()
    cv2.putText(src_label, f"src {name}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(heatmap_label, f"heatmap", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (40, 40, 40), 2, cv2.LINE_AA)
    combo = np.concatenate([src_label, side, sim_panel, side, heatmap_label], axis=1)

    out_heatmap = os.path.join(OUT_DIR, f"{name}.png")
    out_sim = os.path.join(DEBUG_DIR, f"{name}_simulation.png")
    out_combo = os.path.join(DEBUG_DIR, f"{name}_combo.png")
    Image.fromarray(heatmap).save(out_heatmap)
    Image.fromarray(sim_panel).save(out_sim)
    Image.fromarray(combo).save(out_combo)
    print(f"  -> {out_heatmap}")
    print(f"  -> {out_sim}")
    print(f"  -> {out_combo}")
    return name


def main():
    print("=" * 60)
    print("外轮廓 + 邻接 → 力学模拟 → 区域生长热度图")
    print("=" * 60)
    name = run_one(SAMPLE_NAME)
    print("\n完成. 样本:", name)


if __name__ == "__main__":
    main()
