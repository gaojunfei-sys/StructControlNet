import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import torch
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image

from project_config import (
    BASE_MODEL_PATH as CONFIG_BASE_MODEL_PATH,
    CONTROLNET_PATH as CONFIG_CONTROLNET_PATH,
    GENERATED_LAYOUT_DIR,
    RANDOM_SEED,
    TESTSET_HEATMAP_DIR,
)


CONTROLNET_PATH = str(CONFIG_CONTROLNET_PATH)
BASE_MODEL_PATH = str(CONFIG_BASE_MODEL_PATH)
HEATMAP_DIR = str(TESTSET_HEATMAP_DIR)

OUTPUT_DIR = str(GENERATED_LAYOUT_DIR)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

MAX_IMAGES_TRIAL = 500


RESIZE_W = 512
RESIZE_H = 512

PROMPT = "white background, simple background, solo, no humans"

NUM_INFERENCE_STEPS = 100
GUIDANCE_SCALE = 5.0
CONTROLNET_CONDITIONING_SCALE = 3.0
IMG2IMG_STRENGTH = 0.7

USE_SEED = True
BASE_SEED = RANDOM_SEED


def list_heatmap_images(directory):
    if not os.path.isdir(directory):
        return []

    names = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            names.append(os.path.join(directory, name))

    return names


def save_image_rgb(img: Image.Image, path: str) -> None:
    img = img.convert("RGB")
    ext = os.path.splitext(path)[1].lower()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if ext in (".jpg", ".jpeg"):
        img.save(path, quality=95)
    else:
        img.save(path)


def generate_one_layout(
    heatmap_path: str,
    output_path: str,
    pipe,
    image_index: int,
) -> None:
    original_img = Image.open(heatmap_path).convert("RGB")
    orig_w, orig_h = original_img.size
    print(f"  原始尺寸: {orig_w} × {orig_h}")

    # The heatmap provides both the initial image and ControlNet condition.
    init_image = original_img.resize((RESIZE_W, RESIZE_H))
    control_image = original_img.resize((RESIZE_W, RESIZE_H))

    generator = None
    if USE_SEED:
        generator = torch.Generator(device="cuda").manual_seed(BASE_SEED + image_index)

    with torch.inference_mode():
        image = pipe(
            prompt=PROMPT,
            image=init_image,
            control_image=control_image,
            strength=IMG2IMG_STRENGTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            controlnet_conditioning_scale=CONTROLNET_CONDITIONING_SCALE,
            generator=generator,
        ).images[0]

    if image.size != (orig_w, orig_h):
        image = image.resize((orig_w, orig_h), Image.LANCZOS)
        print(f"  缩放回原始尺寸: {orig_w} × {orig_h}")

    save_image_rgb(image, output_path)
    print(f"  ✅ layout 已保存: {output_path}")


def generate_layout_batch():
    paths = list_heatmap_images(HEATMAP_DIR)

    if not paths:
        raise FileNotFoundError(
            f"在热度图目录未找到图片（支持 {sorted(IMAGE_EXTENSIONS)}）: {HEATMAP_DIR}"
        )

    if MAX_IMAGES_TRIAL is not None:
        paths = paths[:MAX_IMAGES_TRIAL]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"🚀 开始生成 layout，共 {len(paths)} 张…")
    print(f"输出目录: {OUTPUT_DIR}")

    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        torch_dtype=torch.float32,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        BASE_MODEL_PATH,
        controlnet=controlnet,
        torch_dtype=torch.float32,
        safety_checker=None,
    )

    pipe = pipe.to("cuda")

    for idx, heatmap_path in enumerate(paths):
        base_name = os.path.basename(heatmap_path)
        output_path = os.path.join(OUTPUT_DIR, base_name)

        print(f"\n── {base_name} ──")
        generate_one_layout(
            heatmap_path=heatmap_path,
            output_path=output_path,
            pipe=pipe,
            image_index=idx,
        )

    print(f"\n✅ layout 批量生成完成，输出目录: {OUTPUT_DIR}")
    return paths


if __name__ == "__main__":
    generate_layout_batch()
