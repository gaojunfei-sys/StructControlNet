

from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np
from PIL import Image

from project_config import GENERATED_LAYOUT_DIR, GROUND_TRUTH_LAYOUT_DIR, TESTSET_DIR


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


LAYOUT_DIR = str(GENERATED_LAYOUT_DIR)
GT_DIR = str(GROUND_TRUTH_LAYOUT_DIR)
OUT_CSV = str(TESTSET_DIR / "layout_psnr.csv")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
UINT8_DATA_RANGE = 255.0
LEGACY_FLOAT01_RANGE255_OFFSET = 20.0 * math.log10(255.0)


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


def psnr_from_mse(mse: float, data_range: float = UINT8_DATA_RANGE) -> float:
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def compute_pair(name: str, layout_path: str, gt_path: str) -> dict[str, object]:
    gt = load_rgb(gt_path)
    pred = load_rgb(layout_path)
    if pred.shape[:2] != gt.shape[:2]:
        pred = resize_to(pred, gt.shape[:2])

    diff = gt.astype(np.float32) - pred.astype(np.float32)
    mse_8bit = float(np.mean(diff * diff))
    psnr_8bit = psnr_from_mse(mse_8bit)

    if math.isinf(psnr_8bit):
        psnr_legacy = float("inf")
    else:
        psnr_legacy = psnr_8bit + LEGACY_FLOAT01_RANGE255_OFFSET

    return {
        "name": name,
        "mse_8bit": mse_8bit,
        "psnr_8bit": psnr_8bit,
        "psnr_legacy_float01_range255": psnr_legacy,
    }


def finite_mean(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    if finite:
        return float(np.mean(finite))
    return float("inf")


def compute_psnr() -> tuple[float, float, int]:
    pairs = paired_paths()
    if not pairs:
        raise RuntimeError(f"No paired images found.\nlayout: {LAYOUT_DIR}\nGT: {GT_DIR}")

    rows: list[dict[str, object]] = []
    for name, layout_path, gt_path in pairs:
        row = compute_pair(name, layout_path, gt_path)
        rows.append(row)
        strict = float(row["psnr_8bit"])
        legacy = float(row["psnr_legacy_float01_range255"])
        if math.isinf(strict):
            print(f"  {name}: PSNR_8bit=inf, legacy=inf")
        else:
            print(f"  {name}: PSNR_8bit={strict:.4f} dB, legacy={legacy:.4f} dB")

    mean_psnr = finite_mean([float(r["psnr_8bit"]) for r in rows])
    mean_legacy = finite_mean([float(r["psnr_legacy_float01_range255"]) for r in rows])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nmean PSNR_8bit: {mean_psnr:.4f} dB")
    print(f"mean legacy PSNR: {mean_legacy:.4f} dB")
    print(f"legacy offset: +{LEGACY_FLOAT01_RANGE255_OFFSET:.4f} dB")
    print(f"evaluated images: {len(rows)}")
    print(f"CSV: {OUT_CSV}")

    return mean_psnr, mean_legacy, len(rows)


if __name__ == "__main__":
    compute_psnr()
