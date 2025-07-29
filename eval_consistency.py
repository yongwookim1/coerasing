import argparse
import os
import csv
import shutil
import numpy as np
import torch
import random
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDIMScheduler
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from data.nudity_prompts import nudity_prompts_100


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_checkpoint", type=str, default=None, help="Path to the UNet checkpoint file.")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save evaluation results.")
    parser.add_argument("--device", type=str, default="0", help="GPU device to use.")
    
    args = parser.parse_args()
    return args


def setup_pipeline(unet_checkpoint, device):
    pipe_original = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    
    # Load UNet checkpoint if provided
    if unet_checkpoint is not None:
        print(f"Loading UNet from: {unet_checkpoint}")
        pipe.unet.load_state_dict(
            torch.load(unet_checkpoint, map_location="cpu"), strict=False)
    pipe = pipe.to(device)
    
    return pipe_original, pipe


def generate_images(pipe, prompts, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)
    
    generator = torch.Generator(device=device)
    
    for i, prompt in enumerate(tqdm(prompts, desc=f"Generating images in {os.path.basename(output_dir)}")):
        generator.manual_seed(i) 
        
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                generator=generator,
                num_inference_steps=25,
                guidance_scale=7.5,
            )
        image = result.images[0]
        image_path = os.path.join(output_dir, f"{i:04d}.png")
        image.save(image_path)


def calculate_clip_score_image_to_image(image_dir1, image_dir2, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()

    clip_scores = []

    image_files = sorted([f for f in os.listdir(image_dir1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for image_file in tqdm(image_files, desc="Calculating Image-to-Image CLIP scores"):
        path1 = os.path.join(image_dir1, image_file)
        path2 = os.path.join(image_dir2, image_file)

        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")
        
        inputs = processor(images=[image1, image2], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)
            similarity = (image_features[0] @ image_features[1]).item()
            clip_scores.append(similarity * 100.0)
        
    mean_score = np.mean(clip_scores)
    std_score = np.std(clip_scores)

    return mean_score, std_score


def main():
    set_seed()
    
    args = parse_args()
    
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    
    if args.output_path is None:
        if args.unet_checkpoint is not None:
            path_parts = [p for p in args.unet_checkpoint.split('/') if p]
            output_path = os.path.join("eval", "_".join(path_parts[-5:]))
        else:
            output_path = os.path.join("eval/SD_v1-4_eval")
    else:
        output_path = args.output_path
    
    prompts = nudity_prompts_100
    
    # Setup output directories
    generated_images_dir_original = os.path.join(output_path, "generated_images_original")
    generated_images_dir_erased = os.path.join(output_path, "generated_images_erased")
    os.makedirs(generated_images_dir_original, exist_ok=True)
    os.makedirs(generated_images_dir_erased, exist_ok=True)
    
    # Setup pipeline
    pipe_original, pipe_erased = setup_pipeline(args.unet_checkpoint, device)
    
    # Generate images
    print("Generating original images...")
    generate_images(pipe_original, prompts, generated_images_dir_original, device)
    
    print("Generating erased/modified images...")
    generate_images(pipe_erased, prompts, generated_images_dir_erased, device)

    # Calculate Image-to-Image CLIP scores
    print("\nCalculating Image-to-Image CLIP scores between original and erased sets...")
    clip_mean, clip_std = calculate_clip_score_image_to_image(
        generated_images_dir_original,
        generated_images_dir_erased,
        device
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Number of samples evaluated: {len(prompts)}")
    print(f"Image-Image CLIP Score: {clip_mean:.3f} ± {clip_std:.3f}")
    print("="*60)
    
    # Save results to file
    results_file = os.path.join(output_path, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Number of samples evaluated: {len(prompts)}\n")
        f.write(f"Image-Image CLIP Score: {clip_mean:.3f} ± {clip_std:.3f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Clean up GPU memory
    del pipe_original, pipe_erased
    torch.cuda.empty_cache()
    
    # Clean up generated images
    print("Cleaning up generated image directories...")
    shutil.rmtree(generated_images_dir_original)
    shutil.rmtree(generated_images_dir_erased)


if __name__ == "__main__":
    main()