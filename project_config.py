"""Shared, portable path configuration for the StructControlNet scripts.

All defaults live under this repository. Set the documented environment
variables when the dataset or model weights are stored elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _configured_path(env_name: str, default: Path) -> Path:
    value = os.environ.get(env_name)
    path = Path(value).expanduser() if value else default
    return path.resolve()


DATA_ROOT = _configured_path(
    "STRUCTCONTROLNET_DATA_ROOT",
    PROJECT_ROOT / "data" / "rplan",
)
MODEL_ROOT = _configured_path(
    "STRUCTCONTROLNET_MODEL_ROOT",
    PROJECT_ROOT / "models",
)

TRAIN_LAYOUT_DIR = _configured_path(
    "STRUCTCONTROLNET_TRAIN_LAYOUT_DIR",
    DATA_ROOT / "layout",
)
TRAIN_HEATMAP_DIR = _configured_path(
    "STRUCTCONTROLNET_TRAIN_HEATMAP_DIR",
    DATA_ROOT / "heatmap",
)

TESTSET_DIR = _configured_path(
    "STRUCTCONTROLNET_TESTSET_DIR",
    DATA_ROOT / "testset",
)
TESTSET_HEATMAP_DIR = _configured_path(
    "STRUCTCONTROLNET_TESTSET_HEATMAP_DIR",
    TESTSET_DIR / "heatmap",
)
GENERATED_LAYOUT_DIR = _configured_path(
    "STRUCTCONTROLNET_GENERATED_LAYOUT_DIR",
    TESTSET_DIR / "layout",
)
GROUND_TRUTH_LAYOUT_DIR = _configured_path(
    "STRUCTCONTROLNET_GROUND_TRUTH_LAYOUT_DIR",
    TESTSET_DIR / "images",
)
PHYSICS_DEBUG_DIR = _configured_path(
    "STRUCTCONTROLNET_PHYSICS_DEBUG_DIR",
    TESTSET_DIR / "physics_debug",
)

BASE_MODEL_PATH = _configured_path(
    "STRUCTCONTROLNET_BASE_MODEL_PATH",
    MODEL_ROOT / "stable-diffusion",
)
CONTROLNET_PATH = _configured_path(
    "STRUCTCONTROLNET_CONTROLNET_PATH",
    MODEL_ROOT / "controlnet",
)

RANDOM_SEED = int(os.environ.get("STRUCTCONTROLNET_RANDOM_SEED", "42"))
