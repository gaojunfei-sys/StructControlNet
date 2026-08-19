
import os
import time
import random
from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from project_config import TRAIN_HEATMAP_DIR, TRAIN_LAYOUT_DIR


SRC_DIR = str(TRAIN_LAYOUT_DIR)
DST_DIR = str(TRAIN_HEATMAP_DIR)
START_IDX = 1
END_IDX = 3000

# 像素分类阈值
WHITE_THRESH = 240          # R,G,B 同时 >= 即视为白色
WALL_CHROMA_MAX = 22        # 灰度判定: max-min <= 该值视为无彩
WALL_VALUE_MAX = 220        # 灰度且亮度 <= 该值视为墙 (放宽以吃掉浅灰抗锯齿)
COLOR_MERGE_DIST = 28       # 色块聚类合并距离
COLOR_ASSIGN_MAX_DIST = 40  # 像素到聚类中心最大距离 (收紧, 避免抗锯齿混入)
MIN_REGION_PIXELS = 400     # 过滤过小的色块连通块 (碎片重新标为待填)
COLOR_FREQ_MIN = 500        # 进入聚类候选的颜色最小出现次数 (只保留主色)
COLOR_OPEN_KERNEL = 5       # 色块掩膜形态学开运算核大小, 去除细小噪点

# 亮黄窗户检测 (示例像素值约 R=255 G=225 B=25)
WIN_R_MIN = 220
WIN_G_MIN = 180
WIN_B_MAX = 90
WIN_RB_DIFF_MIN = 100  # R - B 至少差这么多, 排除米色/淡黄房间色

# 外墙判定: 用建筑外轮廓 cv2.findContours, 沿轮廓向内扩 OUTER_WALL_BAND_MULT 倍墙厚
OUTER_WALL_BAND_MULT = 1.4  # 外墙带宽 = 估计墙厚 × 该倍数
OUTER_WALL_MIN_BAND = 6     # 最小带宽 (防止墙厚估计偏小)
OUTER_CONTOUR_MIN_AREA = 200  # 过滤微小噪点轮廓

# 内墙重画 (粗细自适应: 取 split_outer_inner_walls 估算的"内墙"墙厚, 钳制范围)
REDRAW_INNER_WALL_MIN = 3     # 最小粗细
REDRAW_INNER_WALL_MAX = 6     # 最大粗细
REDRAW_INNER_WALL_SHRINK = 1  # 在自动估算上再额外减薄的像素数

STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# cmap 取值约定:
#   -2 = 白色背景        (固定)
#   -1 = 外墙灰色        (固定, 输出沿用原图灰)
#   -4 = 亮黄窗户        (固定, 输出沿用原图黄)
#   -5 = 内墙 (灰)       (待填: 由区域生长吞掉)
#   -3 = 未分类杂散      (待填)
#    0..K-1 = 色块类别


def classify_pixels(img):
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

    rest_mask = (cmap == -3)
    color_rgb = {}
    if np.any(rest_mask):
        rest_pixels = img[rest_mask]
        cnt = Counter(map(tuple, rest_pixels.tolist()))
        candidates = [
            (np.array(c, dtype=np.float64), n)
            for c, n in cnt.most_common()
            if n >= COLOR_FREQ_MIN
        ]
        clusters = []  # [ [centroid, total_count] ]
        for col, n in candidates:
            placed = False
            for cl in clusters:
                if np.linalg.norm(col - cl[0]) < COLOR_MERGE_DIST:
                    new_tot = cl[1] + n
                    cl[0] = (cl[0] * cl[1] + col * n) / new_tot
                    cl[1] = new_tot
                    placed = True
                    break
            if not placed:
                clusters.append([col.copy(), n])

        if clusters:
            centers = np.array([cl[0] for cl in clusters], dtype=np.float32)
            ys, xs = np.where(rest_mask)
            px = img[ys, xs].astype(np.float32)
            d = np.linalg.norm(px[:, None, :] - centers[None, :, :], axis=2)
            nearest = np.argmin(d, axis=1)
            mind = d[np.arange(len(nearest)), nearest]
            valid = mind < COLOR_ASSIGN_MAX_DIST
            cmap[ys[valid], xs[valid]] = nearest[valid]
            for i, cl in enumerate(clusters):
                color_rgb[i] = tuple(int(round(v)) for v in cl[0])

    print(
        f"  白={int(np.sum(cmap == -2))}, 墙(总)={int(np.sum(cmap == -1))}, "
        f"窗={int(np.sum(cmap == -4))}, 色块={int(np.sum(cmap >= 0))}, "
        f"未分类={int(np.sum(cmap == -3))}, 颜色类别数={len(color_rgb)}"
    )
    return cmap, color_rgb


def split_outer_inner_walls(cmap):
    """
    用建筑的外轮廓判别外墙/内墙:
      1. 估计墙厚
      2. foreground = 非白像素 (整栋楼: 墙+色块+窗+杂散)
      3. cv2.findContours(RETR_EXTERNAL) 拿到建筑外轮廓
      4. 沿轮廓画一条粗细 = 墙厚 × OUTER_WALL_BAND_MULT 的带状区域
      5. 落在带内的墙像素 → 外墙 (-1 保留)
      6. 其余墙像素 → 内墙 (-5, 后续被生长吞掉)
    """
    wall_mask = (cmap == -1)
    if not np.any(wall_mask):
        return cmap, 0, 0, 6.0

    h, w = cmap.shape

    inside = cv2.distanceTransform(wall_mask.astype(np.uint8), cv2.DIST_L2, 3)
    half_thickness = float(inside.max())
    wall_thickness = max(2.0, half_thickness * 2.0)
    band = max(OUTER_WALL_MIN_BAND, int(round(wall_thickness * OUTER_WALL_BAND_MULT)))

    foreground = (cmap != -2).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    outline_band = np.zeros_like(foreground)
    kept = 0
    for c in contours:
        if cv2.contourArea(c) >= OUTER_CONTOUR_MIN_AREA:
            cv2.drawContours(outline_band, [c], -1, 255, thickness=band)
            kept += 1

    outer = wall_mask & (outline_band > 0)
    inner = wall_mask & (~outer)

    n_outer = int(outer.sum())
    n_inner = int(inner.sum())

    # Estimate inner-wall thickness separately from the thicker outer wall.
    if inner.any():
        inner_dist = cv2.distanceTransform(inner.astype(np.uint8), cv2.DIST_L2, 3)
        inner_thickness = max(2.0, float(inner_dist.max()) * 2.0)
    else:
        inner_thickness = wall_thickness

    print(f"  墙厚估计: 全墙≈{wall_thickness:.1f}px, "
          f"内墙≈{inner_thickness:.1f}px, "
          f"外墙带宽={band}px, 外轮廓数={kept} → "
          f"外墙保留={n_outer}, 内墙吞掉={n_inner}")
    return cmap, n_outer, n_inner, inner_thickness, inner


def label_color_regions(cmap, color_rgb):
    """
    每个色块连通块 = 一个独立 region (拥有唯一 id), 即使颜色相同。
    流程:
      1. 对每个颜色掩膜先做形态学开运算 → 干掉细小噪点, 这些像素重置为 -3
      2. 连通组件标记
      3. 过小连通块 (< MIN_REGION_PIXELS) 也重置为 -3
    被重置的像素后续由生长 / 邻域填补阶段吸收, 不会形成"小色块"。
    """
    regions = []
    cur = 0
    dropped_open = 0
    dropped_small = 0
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (COLOR_OPEN_KERNEL, COLOR_OPEN_KERNEL)
    )

    for cls_val in sorted(color_rgb.keys()):
        mask = (cmap == cls_val)
        if not np.any(mask):
            continue

        m8 = mask.astype(np.uint8)
        opened = cv2.morphologyEx(m8, cv2.MORPH_OPEN, open_kernel)
        removed_by_open = mask & (opened == 0)
        if removed_by_open.any():
            cmap[removed_by_open] = -3
            dropped_open += int(removed_by_open.sum())
        clean_mask = opened.astype(bool)

        labeled, nf = ndimage.label(clean_mask, structure=STRUCT4)
        for i in range(1, nf + 1):
            rm = (labeled == i)
            cnt = int(rm.sum())
            if cnt < MIN_REGION_PIXELS:
                cmap[rm] = -3
                dropped_small += cnt
                continue
            ys, xs = np.where(rm)
            cy, cx = int(ys.mean()), int(xs.mean())
            regions.append({
                'id': cur,
                'color_id': int(cls_val),
                'rgb': color_rgb[cls_val],
                'pixel_count': cnt,
                'centroid': (cx, cy),
                'mask': rm,
            })
            cur += 1

    print(f"  彩色连通区域: {cur} 个 "
          f"(开运算丢弃={dropped_open}, 小块丢弃={dropped_small})")
    for r in regions:
        print(f"    区域{r['id']}: rgb={r['rgb']}, "
              f"像素={r['pixel_count']}, 形心=({r['centroid'][0]},{r['centroid'][1]})")
    return regions


def detect_partitions(cmap):
    """
    固定像素 (外墙-1, 白-2, 窗-4) 作为绝对分隔。
    其他像素 (色块, 内墙-5, 杂散-3) 形成可生长前景, 用 4 邻域连通成分划分房间分区。
    返回 partition_map: 像素所属分区编号, 固定像素处为 -1。
    """
    fixed = (cmap == -1) | (cmap == -2) | (cmap == -4)
    foreground = ~fixed
    part_labels, n_parts = ndimage.label(foreground, structure=STRUCT4)
    partition_map = np.full(cmap.shape, -1, dtype=np.int32)
    partition_map[foreground] = part_labels[foreground] - 1
    print(f"  外墙+窗户分隔形成 {n_parts} 个房间分区")
    return partition_map


def compute_targets(regions, cmap, partition_map):
    """
    每个分区内的可生长像素 = 该分区色块 + 该分区内墙 + 杂散
    每个种子的目标 = 原始 pixel_count + 同分区内墙按比例分配的额外像素
    """
    h, w = cmap.shape
    n_seeds = len(regions)
    targets = [r['pixel_count'] for r in regions]

    seed_partition = []
    for r in regions:
        cx, cy = r['centroid']
        cx = min(max(cx, 0), w - 1)
        cy = min(max(cy, 0), h - 1)
        p = int(partition_map[cy, cx])
        if p < 0:  # 形心落在固定像素上 → 用 mask 中点
            ys, xs = np.where(r['mask'])
            mid = len(ys) // 2
            cy, cx = int(ys[mid]), int(xs[mid])
            p = int(partition_map[cy, cx])
        seed_partition.append(p)

    part_total = {}
    n_parts = int(partition_map.max()) + 1 if partition_map.max() >= 0 else 0
    for p in range(n_parts):
        part_total[p] = int((partition_map == p).sum())

    part_seed_orig = defaultdict(list)  # part -> list[(seed_idx, original_count)]
    for i, p in enumerate(seed_partition):
        if p >= 0:
            part_seed_orig[p].append((i, targets[i]))

    # Preserve original area ratios within each disconnected partition.
    new_targets = list(targets)
    for p, items in part_seed_orig.items():
        total_avail = part_total.get(p, 0)
        sum_orig = sum(c for _, c in items)
        if sum_orig <= 0 or total_avail <= 0:
            continue
        allocated = 0
        for idx, (seed_i, orig_c) in enumerate(items):
            if idx == len(items) - 1:
                t = total_avail - allocated
            else:
                t = int(round(total_avail * orig_c / sum_orig))
                allocated += t
            new_targets[seed_i] = max(t, orig_c)  # 不少于原始值
        diff = total_avail - sum(new_targets[i] for i, _ in items)
        if diff != 0:
            big_seed = max(items, key=lambda x: x[1])[0]
            new_targets[big_seed] += diff

    return new_targets, seed_partition


def region_growing(regions, cmap, partition_map, target_pixels, seed_partition):
    h, w = cmap.shape
    n_seeds = len(regions)

    region_map = np.full((h, w), -99, dtype=np.int32)
    region_map[cmap == -1] = -1
    region_map[cmap == -2] = -2
    region_map[cmap == -4] = -4

    fixed_mask = np.zeros((h, w), dtype=np.uint8)
    fixed_mask[cmap == -1] = 1
    fixed_mask[cmap == -2] = 1
    fixed_mask[cmap == -4] = 1

    unassigned = np.ones((h, w), dtype=np.uint8)
    unassigned[fixed_mask == 1] = 0

    if n_seeds == 0:
        return region_map, unassigned, [], target_pixels, fixed_mask

    assigned = [0] * n_seeds
    DIRS4 = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    fronts = [deque() for _ in range(n_seeds)]

    order = list(range(n_seeds))
    random.shuffle(order)
    for idx in order:
        r = regions[idx]
        sp = seed_partition[idx]
        cx, cy = r['centroid']
        if not (0 <= cx < w and 0 <= cy < h
                and unassigned[cy, cx] and partition_map[cy, cx] == sp):
            ys, xs = np.where(r['mask'] & (unassigned == 1))
            if len(ys) == 0:
                continue
            mid = len(ys) // 2
            cy, cx = int(ys[mid]), int(xs[mid])
            sp = int(partition_map[cy, cx])
            seed_partition[idx] = sp

        for _ in range(5):
            ox = random.randint(-2, 2)
            oy = random.randint(-2, 2)
            tx, ty = cx + ox, cy + oy
            if (0 <= tx < w and 0 <= ty < h
                    and unassigned[ty, tx]
                    and partition_map[ty, tx] == sp):
                cx, cy = tx, ty
                break

        region_map[cy, cx] = idx
        unassigned[cy, cx] = 0
        assigned[idx] += 1
        for dx, dy in DIRS4:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < w and 0 <= ny < h
                    and unassigned[ny, nx]
                    and partition_map[ny, nx] == sp):
                fronts[idx].append((nx, ny))

    active = set(range(n_seeds))
    while active:
        active_list = list(active)
        random.shuffle(active_list)
        for i in active_list:
            if assigned[i] >= target_pixels[i] or not fronts[i]:
                if assigned[i] >= target_pixels[i] or not fronts[i]:
                    active.discard(i)
                continue
            steps = min(random.randint(3, 12), target_pixels[i] - assigned[i])
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
                sp = seed_partition[i]
                if (unassigned[y, x]
                        and partition_map[y, x] == sp
                        and fixed_mask[y, x] == 0):
                    region_map[y, x] = i
                    unassigned[y, x] = 0
                    assigned[i] += 1
                    dirs = list(DIRS4)
                    random.shuffle(dirs)
                    for dx, dy in dirs:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < w and 0 <= ny < h
                                and unassigned[ny, nx]
                                and partition_map[ny, nx] == sp
                                and fixed_mask[ny, nx] == 0):
                            fronts[i].append((nx, ny))
                if assigned[i] >= target_pixels[i]:
                    break

    return region_map, unassigned, assigned, target_pixels, fixed_mask


def neighbor_fill(region_map, unassigned, assigned, target_pixels,
                  partition_map, fixed_mask, h, w):
    dx = [0, 1, 0, -1]
    dy = [-1, 0, 1, 0]
    for _ in range(3):
        changed = False
        for y in range(h):
            for x in range(w):
                if (unassigned[y, x]
                        and partition_map[y, x] >= 0
                        and fixed_mask[y, x] == 0):
                    neighbors = []
                    for k in range(4):
                        nx, ny = x + dx[k], y + dy[k]
                        if (0 <= nx < w and 0 <= ny < h
                                and not unassigned[ny, nx]):
                            idx = region_map[ny, nx]
                            if idx >= 0 and partition_map[ny, nx] == partition_map[y, x]:
                                neighbors.append(idx)
                    if neighbors:
                        cnt = Counter(neighbors)
                        candidates = [i for i, _ in cnt.most_common()
                                      if assigned[i] < target_pixels[i]]
                        chosen = candidates[0] if candidates else cnt.most_common(1)[0][0]
                        region_map[y, x] = chosen
                        unassigned[y, x] = 0
                        assigned[chosen] += 1
                        changed = True
        if not changed:
            break
    return region_map, unassigned, assigned


def fair_allocate(region_map, unassigned, assigned, target_pixels,
                  partition_map, fixed_mask, seed_partition, h, w, n_seeds):
    remaining = int(unassigned.sum())
    if remaining == 0:
        return region_map, assigned

    print(f"  公平分配阶段: 剩余 {remaining} 个未分配像素")

    by_part = defaultdict(list)
    for y in range(h):
        for x in range(w):
            if (unassigned[y, x]
                    and partition_map[y, x] >= 0
                    and fixed_mask[y, x] == 0):
                by_part[int(partition_map[y, x])].append((x, y))

    for part_idx, coords in by_part.items():
        seeds_here = [i for i in range(n_seeds) if seed_partition[i] == part_idx]
        if not seeds_here:
            continue
        perm = np.random.permutation(len(coords))
        coords = [coords[i] for i in perm]
        for (x, y) in coords:
            best_idx = None
            best_def = -1.0
            for s in seeds_here:
                if assigned[s] < target_pixels[s]:
                    deficit = (target_pixels[s] - assigned[s]) / max(target_pixels[s], 1)
                    if deficit > best_def:
                        best_def = deficit
                        best_idx = s
            if best_idx is None:
                neighbors = []
                for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < w and 0 <= ny < h
                            and not unassigned[ny, nx]
                            and region_map[ny, nx] >= 0):
                        neighbors.append(region_map[ny, nx])
                best_idx = (Counter(neighbors).most_common(1)[0][0]
                            if neighbors else seeds_here[0])
            region_map[y, x] = best_idx
            unassigned[y, x] = 0
            assigned[best_idx] += 1

    return region_map, assigned


def sample_inner_wall_color(img, inner_mask, fallback=(128, 128, 128)):
    """
    从原图的内墙像素中取中位 RGB, 作为重画内墙的颜色。
    原图内墙通常比外墙浅 (e.g. (128,128,128) vs (79,79,79)),
    必须用内墙自己的颜色, 不能用外墙的颜色, 否则线条会偏深。
    """
    if inner_mask is None or not np.any(inner_mask):
        return np.array(fallback, dtype=np.uint8)
    px = img[inner_mask]
    med = np.median(px, axis=0)
    return np.array([int(round(v)) for v in med], dtype=np.uint8)


def redraw_inner_walls(heatmap, region_map, wall_color, thickness):
    """
    扫描相邻像素 region_map 不同 (且都 >= 0) 的位置,
    用 wall_color 画一条粗 thickness 的线, 模拟原始内墙。
    同色两个房间因 region_map id 不同, 也会被分隔开。

    注意:
      - 边界只标 "一侧" (左侧 / 上侧) 的那个像素 → 1 px 起线;
      - 然后用一次膨胀把宽度扩到目标 thickness。
      - 之前两侧都标会让总宽 ≈ thickness + 1, 偏粗。
    """
    rm = region_map
    h, w = rm.shape
    boundary = np.zeros((h, w), dtype=bool)

    diff_r = (rm[:, :-1] != rm[:, 1:]) & (rm[:, :-1] >= 0) & (rm[:, 1:] >= 0)
    boundary[:, :-1] |= diff_r
    diff_d = (rm[:-1, :] != rm[1:, :]) & (rm[:-1, :] >= 0) & (rm[1:, :] >= 0)
    boundary[:-1, :] |= diff_d

    # Dilate the one-pixel boundary to approximate the requested wall width.
    if thickness > 1:
        k = max(1, thickness - 1)
        b8 = boundary.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        b8 = cv2.dilate(b8, kernel, iterations=1)
        boundary = b8 > 0

    heatmap[boundary] = wall_color
    return heatmap


def generate_heatmap(img, region_map, regions, cmap, h, w,
                     inner_wall_thickness, inner_wall_color):
    """
    输出:
      - 白色背景: 沿用白色 (255,255,255)
      - 外墙 (-1): 沿用原图灰色 (保护抗锯齿)
      - 窗户 (-4): 沿用原图亮黄
      - 色块: 按聚类得到的纯色填充
      - 内墙: 在不同 region id 边界处按真实墙厚重画 (颜色 = 原图内墙浅灰)
    """
    heatmap = np.full((h, w, 3), 255, dtype=np.uint8)

    outer_wall_mask = (cmap == -1)
    window_mask = (cmap == -4)
    heatmap[outer_wall_mask] = img[outer_wall_mask]
    heatmap[window_mask] = img[window_mask]

    idx_to_rgb = {r['id']: np.array(r['rgb'], dtype=np.uint8) for r in regions}
    for idx, rgb in idx_to_rgb.items():
        sel = (region_map == idx)
        heatmap[sel] = rgb

    smoothed = cv2.medianBlur(heatmap, 3)
    smoothed[outer_wall_mask] = img[outer_wall_mask]
    smoothed[window_mask] = img[window_mask]

    inner_thickness = int(round(inner_wall_thickness)) - REDRAW_INNER_WALL_SHRINK
    inner_thickness = max(REDRAW_INNER_WALL_MIN,
                          min(REDRAW_INNER_WALL_MAX, inner_thickness))
    print(f"  内墙重画: 粗细={inner_thickness}px, "
          f"颜色={tuple(int(v) for v in inner_wall_color)}")
    smoothed = redraw_inner_walls(
        smoothed, region_map, inner_wall_color, thickness=inner_thickness
    )
    smoothed[outer_wall_mask] = img[outer_wall_mask]
    smoothed[window_mask] = img[window_mask]
    return smoothed


def process_image(src_path, dst_dir):
    if not os.path.isfile(src_path):
        print(f"[跳过] 不存在: {src_path}")
        return False

    seed = int(time.time_ns()) ^ os.getpid() ^ hash(src_path)
    random.seed(seed)
    np.random.seed(seed % (2**31))

    basename = os.path.splitext(os.path.basename(src_path))[0]
    print("=" * 60)
    print(f"处理: {src_path}")

    img = np.array(Image.open(src_path).convert("RGB"))
    h, w = img.shape[:2]
    print(f"  尺寸: {w}x{h}")

    print("1. 像素分类 + 颜色识别")
    cmap, color_rgb = classify_pixels(img)
    if not color_rgb:
        print("[警告] 未检测到色块")
        return False

    print("1b. 拆分外墙/内墙")
    cmap, n_outer, n_inner, inner_wall_thickness, inner_mask = \
        split_outer_inner_walls(cmap)
    inner_wall_color = sample_inner_wall_color(img, inner_mask)
    print(f"  内墙原色采样 = {tuple(int(v) for v in inner_wall_color)}")
    if inner_mask is not None and inner_mask.any():
        cmap[inner_mask] = -5

    print("2. 连通区域标记")
    regions = label_color_regions(cmap, color_rgb)
    if not regions:
        print("[警告] 无有效色块区域")
        return False

    print("3. 分区检测 (外墙+窗 分隔)")
    partition_map = detect_partitions(cmap)

    print("3b. 计算每种子目标像素 (含分得的内墙)")
    target_pixels, seed_partition = compute_targets(regions, cmap, partition_map)
    for i, r in enumerate(regions):
        print(f"    种子{i} rgb={r['rgb']}: 原始={r['pixel_count']} → "
              f"目标={target_pixels[i]} (分区={seed_partition[i]})")

    print("4. 区域生长")
    (region_map, unassigned, assigned, target_pixels,
     fixed_mask) = region_growing(regions, cmap, partition_map,
                                   target_pixels, seed_partition)

    print("5. 邻域填补")
    region_map, unassigned, assigned = neighbor_fill(
        region_map, unassigned, assigned, target_pixels,
        partition_map, fixed_mask, h, w
    )

    print("6. 公平分配剩余像素")
    region_map, assigned = fair_allocate(
        region_map, unassigned, assigned, target_pixels,
        partition_map, fixed_mask, seed_partition, h, w, len(regions)
    )

    print("  最终统计:")
    for i, r in enumerate(regions):
        diff = assigned[i] - target_pixels[i]
        print(f"    区域{i} rgb={r['rgb']}: 目标={target_pixels[i]}, "
              f"实际={assigned[i]}, 差={diff}")

    print("7. 生成热度图")
    heatmap = generate_heatmap(
        img, region_map, regions, cmap, h, w,
        inner_wall_thickness, inner_wall_color
    )

    dst_path = os.path.join(dst_dir, f"{basename}.png")
    Image.fromarray(heatmap).save(dst_path)
    print(f"  已保存: {dst_path}")
    return True


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    print("=" * 60)
    print(f"批量处理: {SRC_DIR} 中 {START_IDX}~{END_IDX}.png")
    print(f"输出目录: {DST_DIR}")
    print("=" * 60)

    ok_n = fail_n = skip_n = 0
    for num in range(START_IDX, END_IDX + 1):
        src_path = os.path.join(SRC_DIR, f"{num}.png")
        if not os.path.isfile(src_path):
            skip_n += 1
            continue
        try:
            if process_image(src_path, DST_DIR):
                ok_n += 1
            else:
                fail_n += 1
        except Exception as e:
            fail_n += 1
            import traceback
            print(f"[异常] {num}.png: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"完成! 成功={ok_n}, 失败={fail_n}, 跳过={skip_n}")
    print("=" * 60)


if __name__ == "__main__":
    main()
