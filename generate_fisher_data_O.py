import os
import argparse
from typing import List, Tuple

from PIL import Image, ImageDraw
import numpy as np
import torch
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from ultralytics import YOLOWorld


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sd_model", type=str, default="stabilityai/stable-diffusion-2-inpainting"
    )
    parser.add_argument(
        "--sd_txt2img_model", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument(
        "--yolo_world_weights", type=str, default="models/yolov8x-worldv2.pt"
    )
    parser.add_argument("--device", type=str, default="0,1")
    return parser.parse_args()


def create_directories(root_dir: str):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    original_dir = os.path.join(root_dir, "tench_original")
    inpainted_dir = os.path.join(root_dir, "tench_inpainted")
    comparison_dir = os.path.join(root_dir, "tench_comparisons")

    for d in [original_dir, inpainted_dir, comparison_dir]:
        if not os.path.exists(d):
            os.mkdir(d)

    return original_dir, inpainted_dir, comparison_dir


def setup_pipelines(
    sd_txt2img_model: str, sd_inpaint_model: str, device_img: str, device_inpaint: str
):
    txt2img = StableDiffusionPipeline.from_pretrained(
        sd_txt2img_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(f"cuda:{device_img}")

    inpaint = StableDiffusionInpaintPipeline.from_pretrained(
        sd_inpaint_model,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(f"cuda:{device_inpaint}")

    return txt2img, inpaint


def setup_yolo_world(weights: str):
    model = YOLOWorld(weights)
    model.set_classes(["tench", "fish"])
    return model


def generate_tench_image(
    pipe: StableDiffusionPipeline,
    generator: torch.Generator,
    steps: int,
    guidance: float,
) -> Image.Image:
    prompt = "a photorealistic image of a tench fish in water, natural background, high detail, high quality"
    negative = "low quality, blurry, distorted, cartoon, painting, illustration, text, watermark"
    image = pipe(
        prompt=prompt,
        negative_prompt=negative,
        generator=generator,
        guidance_scale=guidance,
        num_inference_steps=steps,
    ).images[0]
    return image


def detect_tench_boxes_yolo_world(
    model,
    image: Image.Image,
    conf: float,
    min_box_area: int,
) -> List[Tuple[int, int, int, int]]:
    img_np = np.array(image)
    results = model.predict(source=img_np, conf=conf, verbose=False)
    boxes: List[Tuple[int, int, int, int]] = []
    if not results:
        return boxes
    r = results[0]
    if getattr(r, "boxes", None) is None or r.boxes is None:
        return boxes
    xyxy = r.boxes.xyxy
    if xyxy is None:
        return boxes
    xyxy = xyxy.cpu().numpy().astype(int)
    for x1, y1, x2, y2 in xyxy:
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w * h >= min_box_area:
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def build_box_mask(
    image_size: Tuple[int, int], boxes: List[Tuple[int, int, int, int]], dilate_px: int
) -> Image.Image:
    w, h = image_size
    mask = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        x1p = max(0, x1 - dilate_px)
        y1p = max(0, y1 - dilate_px)
        x2p = min(w - 1, x2 + dilate_px)
        y2p = min(h - 1, y2 + dilate_px)
        draw.rectangle([x1p, y1p, x2p, y2p], fill=255)
    return mask


def inpaint_remove_tench(
    pipe: StableDiffusionInpaintPipeline,
    source_image: Image.Image,
    mask_image: Image.Image,
    generator: torch.Generator,
    steps: int,
    guidance: float,
    strength: float,
) -> Image.Image:
    prompt = "natural realistic water background, ripples, reflections, seamless, high quality"
    negative = "fish, tench, animal, artifact, blur, low quality, text, watermark"
    out = pipe(
        prompt=prompt,
        image=source_image,
        mask_image=mask_image,
        negative_prompt=negative,
        generator=generator,
        guidance_scale=guidance,
        num_inference_steps=steps,
        strength=strength,
    ).images[0]
    return out


def draw_boxes(
    image: Image.Image,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (255, 0, 0),
    width: int = 4,
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return annotated


def create_comparison(left: Image.Image, right: Image.Image) -> Image.Image:
    lw, lh = left.size
    rw, rh = right.size
    target_h = min(lh, rh)
    lrs = left.resize((int(lw * target_h / lh), target_h))
    rrs = right.resize((int(rw * target_h / rh), target_h))
    comp = Image.new("RGB", (lrs.width + rrs.width, target_h))
    comp.paste(lrs, (0, 0))
    comp.paste(rrs, (lrs.width, 0))
    return comp


def main():
    args = parse_args()
    device_img, device_inpaint = args.device.split(",")

    txt2img_pipe, inpaint_pipe = setup_pipelines(
        sd_txt2img_model=args.sd_txt2img_model,
        sd_inpaint_model=args.sd_model,
        device_img=device_img,
        device_inpaint=device_inpaint,
    )
    yolo_model = setup_yolo_world(args.yolo_world_weights)

    output_path = "./data/"
    original_dir, inpainted_dir, comparison_dir = create_directories(output_path)

    gen_img = torch.Generator(f"cuda:{device_img}")
    gen_inp = torch.Generator(f"cuda:{device_inpaint}")

    numbers_per_class = 4
    seed_base = 0
    txt2img_steps = 40
    txt2img_guidance = 7.0
    for idx in tqdm(range(numbers_per_class), desc="Generating tench removal pairs"):
        seed = seed_base + idx
        gen_img.manual_seed(seed)
        gen_inp.manual_seed(seed)
        torch.manual_seed(seed)

        tench_img = generate_tench_image(
            pipe=txt2img_pipe,
            generator=gen_img,
            steps=txt2img_steps,
            guidance=txt2img_guidance,
        )

        detect_conf = 0.25
        min_box_area = 20 * 20
        mask_dilate_px = 16
        boxes = detect_tench_boxes_yolo_world(
            model=yolo_model,
            image=tench_img,
            conf=detect_conf,
            min_box_area=min_box_area,
        )

        if len(boxes) == 0:
            mask_img = Image.new("L", tench_img.size, color=0)
        else:
            mask_img = build_box_mask(tench_img.size, boxes, mask_dilate_px)

        inpaint_steps = 40
        inpaint_guidance = 7.0
        inpaint_strength = 0.85
        inpainted_img = inpaint_remove_tench(
            pipe=inpaint_pipe,
            source_image=tench_img,
            mask_image=mask_img,
            generator=gen_inp,
            steps=inpaint_steps,
            guidance=inpaint_guidance,
            strength=inpaint_strength,
        )

        base = f"tench_{idx:04d}.png"
        tench_path = os.path.join(original_dir, base)
        inpaint_path = os.path.join(inpainted_dir, base)
        tench_img.save(tench_path)
        inpainted_img.save(inpaint_path)

        annotated_left = draw_boxes(tench_img, boxes)
        comp = create_comparison(annotated_left, inpainted_img)
        comp.save(os.path.join(comparison_dir, f"comparison_{idx:04d}.png"))

    print(f"Original data saved to: {original_dir}")
    print(f"Inpainted data saved to: {inpainted_dir}")
    print(f"Comparison data saved to: {comparison_dir}")


if __name__ == "__main__":
    main()
