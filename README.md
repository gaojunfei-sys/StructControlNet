# StructControlNet

This repository contains the RPLAN heatmap, ControlNet training, layout
generation, and evaluation scripts.

## Open resources

- Base model: [Stable Diffusion v1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- Dataset: [RPLAN dataset](https://zenodo.org/records/18874946)

## Script pipeline

1. `reconstruct_heatmaps.py` reconstructs training heatmaps from layout images.
2. `train_controlnet.py` trains an SD 1.5 or SDXL ControlNet.
3. `generate_physics_prior.py` writes test conditions to
   `testset/heatmap/{name}.png`.
4. `generate_layouts.py` reads those conditions and writes final layouts to
   `testset/layout/{name}.png`.
5. `evaluate_fid.py`, `evaluate_ssim.py`, `evaluate_psnr.py`, and
   `evaluate_ged.py` compare `testset/layout` with `testset/images`.

Physics diagnostic images are written to `testset/physics_debug`, outside the
condition-image directory.

## Portable paths

Defaults are defined in `project_config.py` and live under the repository:

- dataset root: `data/rplan`
- model root: `models`

Override them without editing source code by setting environment variables:

```powershell
$env:STRUCTCONTROLNET_DATA_ROOT = 'D:\path\to\rplan'
$env:STRUCTCONTROLNET_MODEL_ROOT = 'E:\path\to\models'
$env:STRUCTCONTROLNET_BASE_MODEL_PATH = 'E:\path\to\stable-diffusion'
$env:STRUCTCONTROLNET_CONTROLNET_PATH = 'E:\path\to\checkpoint\controlnet'
$env:STRUCTCONTROLNET_RANDOM_SEED = '42'
```

More specific overrides are available for each directory; their names are
listed in `project_config.py`.
