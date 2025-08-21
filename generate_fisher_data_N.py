import os
import argparse
from PIL import Image
from tqdm.auto import tqdm

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionImg2ImgPipeline

from prompts.nudity_prompts import nudity_prompts_100
from prompts.mma_data import mma_nudity_prompts


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument("--numbers_per_class", type=int, default=1)
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()


def create_directories(root_dir, numbers_per_class):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    forget_save_dir = os.path.join(root_dir, f"forget_data{numbers_per_class*100}")
    retain_save_dir = os.path.join(root_dir, f"retain_data{numbers_per_class*100}")

    if not os.path.exists(forget_save_dir):
        os.mkdir(forget_save_dir)
    if not os.path.exists(retain_save_dir):
        os.mkdir(retain_save_dir)

    return forget_save_dir, retain_save_dir


def setup_pipelines(model_path, device):
    """Setup text-to-image and image-to-image pipelines."""
    # Text-to-image pipeline for initial generation
    txt2img_pipeline = AutoPipelineForText2Image.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
    ).to(f"cuda:{device}")

    # Image-to-image pipeline for retain data generation
    img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        local_files_only=True,
        safety_checker=None,
    ).to(f"cuda:{device}")

    return txt2img_pipeline, img2img_pipeline


def generate_forget_image(
    pipeline,
    prompt,
    generator,
):
    image = pipeline(
        prompt=prompt,
        generator=generator,
    ).images[0]
    return image


def generate_retain_image(
    pipeline,
    prompt,
    image,
    negative_prompt,
    strength=0.8,
):
    retain_image = pipeline(
        prompt=prompt,
        image=image,
        strength=strength,
        negative_prompt=negative_prompt,
    ).images[0]
    return retain_image


def main():
    args = parse_args()

    txt2img_pipeline, img2img_pipeline = setup_pipelines(args.model_path, args.device)

    gen = torch.Generator(f"cuda:{args.device}")

    # Setup directories
    root_dir = "/home/dataset/generation_dataset_v1_5/"
    root_dir = "test/"
    forget_save_dir, retain_save_dir = create_directories(
        root_dir, args.numbers_per_class
    )

    # Setup data
    nude_prompts = mma_nudity_prompts[:100]
    numbers_per_class = args.numbers_per_class

    # Define negative prompts
    negative_prompt_img2img = "nudity, nude, naked, nsfw, explicit, sexual content, adult content, pornographic, erotic, revealing clothing, underwear, lingerie, topless, bottomless, bare chest, exposed, intimate, suggestive"

    # Generate paired forget and retain data
    for prompt_idx, nude_prompt in enumerate(tqdm(nude_prompts)):
        for idx in range(numbers_per_class):
            gen.manual_seed(idx)
            torch.manual_seed(idx)

            # Step 1: Generate forget image using text-to-image
            forget_image = generate_forget_image(
                pipeline=txt2img_pipeline,
                prompt=nude_prompt,
                generator=gen,
            )

            # Save forget image
            forget_filename = "_".join(nude_prompt.split(" "))
            forget_image_path = os.path.join(
                forget_save_dir, f"{forget_filename[:10]}{idx:04d}.png"
            )
            forget_image.save(forget_image_path, "png")

            # Step 2: Generate retain image using forget image as initial image
            retain_image = generate_retain_image(
                pipeline=img2img_pipeline,
                prompt=nude_prompt,
                image=forget_image,
                negative_prompt=negative_prompt_img2img,
                strength=0.8,
            )

            # Save retain image
            retain_image_path = os.path.join(
                retain_save_dir, f"{forget_filename[:10]}{idx:04d}.png"
            )
            retain_image.save(retain_image_path, "png")

    print(f"Forget data saved to: {forget_save_dir}")
    print(f"Retain data saved to: {retain_save_dir}")


if __name__ == "__main__":
    main()
