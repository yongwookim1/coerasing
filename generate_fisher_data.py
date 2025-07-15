import os
import argparse
from tqdm.auto import tqdm

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionImg2ImgPipeline

from data.nudity_prompts import nudity_prompts_100

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4", help="Path to the trained model or pretrained model directory")
parser.add_argument("--numbers_per_class", type=int, default=5, help="Number of images to generate per class")
parser.add_argument("--device", type=str, default="0", help="Device to use")
args = parser.parse_args()

# Text-to-image pipeline for initial generation
txt2img_pipeline = AutoPipelineForText2Image.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    safety_checker=None
).to(f"cuda:{args.device}")

# Image-to-image pipeline for retain data generation
img2img_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    safety_checker=None
).to(f"cuda:{args.device}")

gen = torch.Generator(f"cuda:{args.device}")

root_dir = "/home/dataset/generation_dataset_v1_5/"
root_dir = "test/"
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

nude_prompts = nudity_prompts_100
numbers_per_class = args.numbers_per_class

# Create directories
forget_save_dir = os.path.join(root_dir, f"forget_data{numbers_per_class*100}")
retain_save_dir = os.path.join(root_dir, f"retain_data{numbers_per_class*100}")
if not os.path.exists(forget_save_dir):
    os.mkdir(forget_save_dir)
if not os.path.exists(retain_save_dir):
    os.mkdir(retain_save_dir)

# Define negative prompts
negative_prompt_img2img = "nudity, nude, naked, nsfw, explicit, sexual content, adult content, pornographic, erotic, revealing clothing, underwear, lingerie, topless, bottomless, bare chest, exposed, intimate, suggestive"

# Generate paired forget and retain data
for prompt_idx, (nude_prompt, seed) in enumerate(tqdm(nude_prompts.items(), desc="Generating paired data")):
    for idx in range(numbers_per_class):
        gen.manual_seed(idx)
        torch.manual_seed(idx)
        
        # Step 1: Generate forget image using text-to-image
        forget_image = txt2img_pipeline(
            prompt=nude_prompt,
            generator=gen,
        ).images[0]
        
        # Save forget image
        forget_filename = '_'.join(nude_prompt.split(" "))
        forget_image_path = os.path.join(forget_save_dir, f"{forget_filename[:10]}{idx:04d}.png")
        forget_image.save(forget_image_path, "PNG")
        
        # Step 2: Generate retain image using forget image as initial image
        retain_image = img2img_pipeline(
            prompt=nude_prompt,
            image=forget_image,
            strength=0.8,
            negative_prompt=negative_prompt_img2img
        ).images[0]
        
        # Save retain image
        retain_image_path = os.path.join(retain_save_dir, f"{forget_filename[:10]}{idx:04d}.png")
        retain_image.save(retain_image_path, "PNG")
