

from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from project_config import GENERATED_LAYOUT_DIR, GROUND_TRUTH_LAYOUT_DIR, TESTSET_DIR


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


LAYOUT_DIR = str(GENERATED_LAYOUT_DIR)
GT_DIR = str(GROUND_TRUTH_LAYOUT_DIR)
OUT_CSV = str(TESTSET_DIR / "layout_ged.csv")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# RPLAN room types sharing a rendered color are evaluated as one coarse class.
CLASS_NAMES = {
    0: "living_dining_entrance",
    1: "bedroom_study_guest",
    2: "kitchen",
    3: "bathroom",
    4: "balcony",
    5: "storage",
}

CLASS_COLORS = np.array(
    [
        (244, 242, 229),  # living / dining / entrance
        (253, 244, 171),  # bedroom-like rooms
        (234, 216, 214),  # kitchen
        (205, 233, 252),  # bathroom
        (208, 216, 135),  # balcony
        (245, 225, 195),  # storage
    ],
    dtype=np.float32,
)

IGNORE_LABEL = -1
MIN_COMPONENT_AREA = 64
ADJACENCY_DILATE_ITER = 3


def list_image_names(folder: str) -> set[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Directory does not exist: {folder}")
    names: set[str] = set()
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            names.add(name)
    return names


def paired_paths() -> list[tuple[str, str, str]]:
    layout_names = list_image_names(LAYOUT_DIR)
    gt_names = list_image_names(GT_DIR)
    common = sorted(layout_names & gt_names)

    print(f"layout images: {len(layout_names)}")
    print(f"ground-truth images: {len(gt_names)}")
    print(f"paired images: {len(common)}")
    if layout_names - gt_names:
        print(f"layout-only images: {len(layout_names - gt_names)}")
    if gt_names - layout_names:
        print(f"ground-truth-only images: {len(gt_names - layout_names)}")

    return [
        (name, os.path.join(LAYOUT_DIR, name), os.path.join(GT_DIR, name))
        for name in common
    ]


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_to(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.NEAREST), dtype=np.uint8)


def semantic_label_map(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(np.float32)
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    chroma = maxc - minc
    mean = arr.mean(axis=2)

    background = (mean > 247.0) & (chroma < 18.0)
    wall = (chroma < 32.0) & (mean < 205.0)
    door = (arr[:, :, 0] > 215.0) & (arr[:, :, 1] > 165.0) & (arr[:, :, 2] < 95.0)

    flat = arr.reshape(-1, 3)
    d2 = ((flat[:, None, :] - CLASS_COLORS[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(d2, axis=1).astype(np.int16).reshape(rgb.shape[:2])
    dist = np.sqrt(np.min(d2, axis=1)).reshape(rgb.shape[:2])

    ignore = background | wall | door | (dist > 95.0)
    labels[ignore] = IGNORE_LABEL
    return labels


def connected_components(label_map: np.ndarray) -> tuple[np.ndarray, list[int]]:
    comp_map = np.full(label_map.shape, -1, dtype=np.int32)
    comp_classes: list[int] = []
    next_id = 0

    for cls_id in CLASS_NAMES:
        mask = (label_map == cls_id).astype(np.uint8)
        n_labels, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        for local_id in range(1, n_labels):
            area = int(stats[local_id, cv2.CC_STAT_AREA])
            if area < MIN_COMPONENT_AREA:
                continue
            comp_map[cc == local_id] = next_id
            comp_classes.append(cls_id)
            next_id += 1

    return comp_map, comp_classes


def graph_signature(label_map: np.ndarray) -> tuple[Counter[int], Counter[tuple[int, int]], int, int]:
    comp_map, comp_classes = connected_components(label_map)
    node_counts = Counter(comp_classes)

    n = len(comp_classes)
    edge_counts: Counter[tuple[int, int]] = Counter()
    if n <= 1:
        return node_counts, edge_counts, n, 0

    radius = ADJACENCY_DILATE_ITER
    h, w = comp_map.shape
    comp_edges: set[tuple[int, int]] = set()

    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue

            y0_a = max(0, -dy)
            y1_a = h - max(0, dy)
            x0_a = max(0, -dx)
            x1_a = w - max(0, dx)

            y0_b = max(0, dy)
            y1_b = h - max(0, -dy)
            x0_b = max(0, dx)
            x1_b = w - max(0, -dx)

            a = comp_map[y0_a:y1_a, x0_a:x1_a]
            b = comp_map[y0_b:y1_b, x0_b:x1_b]
            contact = (a >= 0) & (b >= 0) & (a != b)
            if not np.any(contact):
                continue

            aa = a[contact]
            bb = b[contact]
            pairs = np.stack((np.minimum(aa, bb), np.maximum(aa, bb)), axis=1)
            unique_pairs = np.unique(pairs, axis=0)
            for i, j in unique_pairs:
                comp_edges.add((int(i), int(j)))

    for i, j in comp_edges:
        a, b = sorted((comp_classes[i], comp_classes[j]))
        edge_counts[(a, b)] += 1

    return node_counts, edge_counts, n, sum(edge_counts.values())


def counter_l1(a: Counter, b: Counter) -> int:
    keys = set(a) | set(b)
    return int(sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys))


def compute_pair(name: str, layout_path: str, gt_path: str) -> dict[str, object]:
    gt_rgb = load_rgb(gt_path)
    pred_rgb = load_rgb(layout_path)
    if pred_rgb.shape[:2] != gt_rgb.shape[:2]:
        pred_rgb = resize_to(pred_rgb, gt_rgb.shape[:2])

    gt_labels = semantic_label_map(gt_rgb)
    pred_labels = semantic_label_map(pred_rgb)

    pred_nodes, pred_edges, pred_n, pred_e = graph_signature(pred_labels)
    gt_nodes, gt_edges, gt_n, gt_e = graph_signature(gt_labels)

    node_ged = counter_l1(pred_nodes, gt_nodes)
    edge_ged = counter_l1(pred_edges, gt_edges)
    ged = node_ged + edge_ged
    norm = ged / max(gt_n + gt_e, 1)

    return {
        "name": name,
        "ged": ged,
        "normalized_ged": norm,
        "node_ged": node_ged,
        "edge_ged": edge_ged,
        "pred_nodes": pred_n,
        "gt_nodes": gt_n,
        "pred_edges": pred_e,
        "gt_edges": gt_e,
    }


def compute_ged() -> tuple[float, float, int]:
    pairs = paired_paths()
    if not pairs:
        raise RuntimeError(f"No paired images found.\nlayout: {LAYOUT_DIR}\nGT: {GT_DIR}")

    rows = []
    for name, layout_path, gt_path in pairs:
        row = compute_pair(name, layout_path, gt_path)
        rows.append(row)
        print(
            f"  {name}: GED={row['ged']}, normalized={row['normalized_ged']:.6f}, "
            f"node={row['node_ged']}, edge={row['edge_ged']}"
        )

    mean_ged = float(np.mean([float(r["ged"]) for r in rows]))
    mean_norm = float(np.mean([float(r["normalized_ged"]) for r in rows]))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nmean GED: {mean_ged:.6f}")
    print(f"mean normalized GED: {mean_norm:.6f}")
    print(f"evaluated images: {len(rows)}")
    print(f"CSV: {OUT_CSV}")
    return mean_ged, mean_norm, len(rows)


if __name__ == "__main__":
    compute_ged()
