
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn

from project_config import GENERATED_LAYOUT_DIR, GROUND_TRUTH_LAYOUT_DIR

LAYOUT_DIR = str(GENERATED_LAYOUT_DIR)
GT_DIR = str(GROUND_TRUTH_LAYOUT_DIR)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_basenames(folder: str) -> set[str]:
    names = set()
    if not os.path.isdir(folder):
        return names
    for f in os.listdir(folder):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            names.add(f)
    return names


def paired_paths() -> list[tuple[str, str]]:
    layout_names = list_basenames(LAYOUT_DIR)
    gt_names = list_basenames(GT_DIR)
    common = sorted(layout_names & gt_names)
    return [
        (os.path.join(LAYOUT_DIR, n), os.path.join(GT_DIR, n))
        for n in common
    ]


def load_rgb_np(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_to(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    pil = Image.fromarray(arr)
    pil = pil.resize((w, h), Image.LANCZOS)
    return np.asarray(pil, dtype=np.uint8)


def compute_ssim() -> tuple[float, int]:
    pairs = paired_paths()
    if not pairs:
        raise RuntimeError(
            f"未找到配对图片（两边需同名）。\nlayout: {LAYOUT_DIR}\nGT: {GT_DIR}"
        )

    scores: list[float] = []
    for layout_p, gt_p in pairs:
        gt = load_rgb_np(gt_p)
        pred = load_rgb_np(layout_p)
        if pred.shape[:2] != gt.shape[:2]:
            pred = resize_to(pred, (gt.shape[0], gt.shape[1]))
        try:
            s = float(ssim_fn(gt, pred, channel_axis=2, data_range=255))
        except TypeError:
            # scikit-image < 0.19
            s = float(ssim_fn(gt, pred, multichannel=True, data_range=255))
        scores.append(s)
        print(f"  {os.path.basename(layout_p)}: SSIM = {s:.6f}")

    mean_ssim = float(np.mean(scores))
    print(f"\n平均 SSIM ({len(scores)} 张): {mean_ssim:.6f}")
    return mean_ssim, len(scores)


if __name__ == "__main__":
    compute_ssim()
