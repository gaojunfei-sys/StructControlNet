
from __future__ import annotations

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

from project_config import GENERATED_LAYOUT_DIR, GROUND_TRUTH_LAYOUT_DIR

LAYOUT_DIR = str(GENERATED_LAYOUT_DIR)
GT_DIR = str(GROUND_TRUTH_LAYOUT_DIR)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def load_batch(paths: list[str], tfm: transforms.Compose, device: torch.device) -> torch.Tensor:
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(tfm(img))
    return torch.stack(tensors, dim=0).to(device)


class InceptionFeaturizer(nn.Module):
    """Inception v3，提取 avgpool 后 2048 维特征（常用 FID 设定）。"""

    def __init__(self) -> None:
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        net = inception_v3(weights=weights, transform_input=False)
        net.aux_logits = False
        net.eval()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: list[torch.Tensor] = []

        def hook(_m, _inp, out: torch.Tensor) -> None:
            feats.append(torch.flatten(out, 1))

        h = self.net.avgpool.register_forward_hook(hook)
        try:
            with torch.no_grad():
                self.net(x)
        finally:
            h.remove()
        return feats[0]


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
        sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def collect_features(paths: list[str], model: InceptionFeaturizer, tfm, device: torch.device) -> np.ndarray:
    all_feats: list[np.ndarray] = []
    model.eval()
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i : i + BATCH_SIZE]
        x = load_batch(batch_paths, tfm, device)
        with torch.no_grad():
            feat = model(x)
        all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def compute_fid() -> float:
    pairs = paired_paths()
    if len(pairs) < 2:
        raise RuntimeError(
            f"FID 至少需要两侧各 2 张配对图片，当前有效配对: {len(pairs)}。"
            f"\nlayout: {LAYOUT_DIR}\nGT: {GT_DIR}"
        )

    layout_paths = [a for a, _ in pairs]
    gt_paths = [b for _, b in pairs]

    tfm = transforms.Compose(
        [
            transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    model = InceptionFeaturizer().to(DEVICE)

    print(f"设备: {DEVICE}，配对样本数: {len(pairs)}")
    print("提取 layout 特征…")
    f_fake = collect_features(layout_paths, model, tfm, DEVICE)
    print("提取 ground-truth 特征…")
    f_real = collect_features(gt_paths, model, tfm, DEVICE)

    mu_fake = np.mean(f_fake, axis=0)
    sigma_fake = np.cov(f_fake, rowvar=False)
    mu_real = np.mean(f_real, axis=0)
    sigma_real = np.cov(f_real, rowvar=False)

    fid = frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real)
    print(f"FID (layout vs ground-truth): {fid:.4f}")
    return fid


if __name__ == "__main__":
    compute_fid()
