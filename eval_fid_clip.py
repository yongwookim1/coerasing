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
from cleanfid import fid
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_checkpoint", type=str)
    parser.add_argument("--csv_path", type=str, default="prompts/coco_10k.csv")
    parser.add_argument(
        "--coco10k_path", type=str, default="/home/dataset/coco2014/val2014"
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
    )
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    return args


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path):
    prompts = []
    seeds = []
    case_numbers = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            prompts.append(row["prompt"])
            seeds.append(int(row["evaluation_seed"]))
            case_numbers.append(row["case_number"])

    return prompts, seeds, case_numbers


def setup_pipeline(unet_checkpoint, device):
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # DDIM scheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # Load UNet checkpoint
    if unet_checkpoint is not None:
        pipe.unet.load_state_dict(
            torch.load(unet_checkpoint, map_location="cpu"), strict=False
        )
    pipe = pipe.to(device)

    return pipe


def generate_images(pipe, prompts, seeds, output_dir, device, batch_size=1):
    os.makedirs(output_dir, exist_ok=True)

    if batch_size == 1:
        # Original single image generation
        generator = torch.Generator(device=device)

        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            generator.manual_seed(seed)
            torch.manual_seed(seed)

            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                )

            image = result.images[0]
            image_path = os.path.join(output_dir, f"{i:04d}.png")
            image.save(image_path)
    else:
        # Batch processing
        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(prompts))

            batch_prompts = prompts[start_idx:end_idx]
            batch_seeds = seeds[start_idx:end_idx]

            # Create generators for each seed in batch
            generators = [
                torch.Generator(device=device).manual_seed(seed) for seed in batch_seeds
            ]
            torch.manual_seed(batch_seeds[0])

            with torch.no_grad():
                results = pipe(
                    prompt=batch_prompts,
                    generator=generators,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    height=512,
                    width=512,
                    num_images_per_prompt=1,
                )

            # Save batch results
            for i, image in enumerate(results.images):
                global_idx = start_idx + i
                image_path = os.path.join(output_dir, f"{global_idx:04d}.png")
                image.save(image_path)


def create_matched_coco_subset(coco10k_path, subset_dir, case_numbers):
    os.makedirs(subset_dir, exist_ok=True)

    matched_count = 0
    missing_files = []

    for i, case_number in enumerate(tqdm(case_numbers)):
        # Try different COCO filename formats
        possible_filenames = [f"COCO_val2014_{int(case_number):012d}.jpg"]

        src_path = None
        for filename in possible_filenames:
            potential_path = os.path.join(coco10k_path, filename)
            if os.path.exists(potential_path):
                src_path = potential_path
                break

        if src_path:
            # Name the file with the same index as generated image for proper matching
            dst_path = os.path.join(subset_dir, f"{i:04d}.jpg")
            shutil.copy(src_path, dst_path)
            matched_count += 1

    return matched_count


def calculate_fid(coco10k_path, generated_images_path, device, case_numbers):
    # Create temporary subset directory
    subset_dir = os.path.join(os.path.dirname(generated_images_path), "coco_subset")

    # 1:1 matching
    num_used = create_matched_coco_subset(coco10k_path, subset_dir, case_numbers)

    # Calculate FID with subset
    fid_value = fid.compute_fid(subset_dir, generated_images_path)

    return fid_value, num_used


def calculate_clip_score(prompts, image_dir, device):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    clip_scores = []

    for i, prompt in enumerate(tqdm(prompts)):
        image_path = os.path.join(image_dir, f"{i:04d}.png")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")

        # Process image and text separately
        image_inputs = processor(images=image, return_tensors="pt").to(device)
        text_inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(
            device
        )

        # Extract features
        with torch.no_grad():
            image_features = clip_model.get_image_features(**image_inputs)
            text_features = clip_model.get_text_features(**text_inputs)

            # L2 normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Calculate cosine similarity and scale to 0-100
            clip_score = (image_features * text_features).sum().item() * 100
            clip_scores.append(clip_score)

    mean_score = np.mean(clip_scores)
    std_score = np.std(clip_scores)

    return mean_score, std_score


def main():
    set_seed()

    args = parse_args()

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    if args.output_path is None:
        if args.unet_checkpoint is not None:
            save_path_instances = [i for i in args.unet_checkpoint.split("/")]
            save_path_instances = save_path_instances[2:]
            output_path = os.path.join(
                f"eval/{save_path_instances[0]}_{save_path_instances[1]}_{save_path_instances[2]}_{save_path_instances[3]}_{save_path_instances[4]}_{save_path_instances[5]}_{save_path_instances[6]}"
            )
        else:
            output_path = os.path.join("eval/SD")
    else:
        output_path = args.output_path

    # Load prompts and case numbers
    all_prompts, all_seeds, all_case_numbers = load_data(args.csv_path)

    # Take first N prompts
    prompts = all_prompts[: args.num_samples]
    seeds = all_seeds[: args.num_samples]
    case_numbers = all_case_numbers[: args.num_samples]

    # Setup output directories
    generated_images_dir = os.path.join(output_path, "generated_images")
    os.makedirs(output_path, exist_ok=True)

    # Setup pipeline
    pipe = setup_pipeline(args.unet_checkpoint, device)

    # Generate images with specified batch size
    generate_images(pipe, prompts, seeds, generated_images_dir, device, args.batch_size)

    fid_value, _ = calculate_fid(
        args.coco10k_path, generated_images_dir, device, case_numbers=case_numbers
    )

    clip_mean, clip_std = calculate_clip_score(prompts, generated_images_dir, device)

    # Print results
    print(f"FID Score: {fid_value:.2f}")
    print(f"CLIP Score: {clip_mean:.3f} ± {clip_std:.3f}")

    # Save results to file
    results_file = os.path.join(output_path, "evaluation_results.txt")
    with open(results_file, "a") as f:
        f.write("Retainability Results\n")
        f.write(f"FID Score: {fid_value:.2f}\n")
        f.write(f"CLIP Score: {clip_mean:.3f} ± {clip_std:.3f}\n")
        f.write("")

    # Clean up GPU memory
    del pipe
    torch.cuda.empty_cache()

    # Clean up temporary folders
    os.system(f"rm -rf {generated_images_dir}")
    coco_subset_dir = os.path.join(output_path, "coco_subset")
    if os.path.exists(coco_subset_dir):
        shutil.rmtree(coco_subset_dir)


if __name__ == "__main__":
    main()
