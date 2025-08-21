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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from prompts.nudity_prompts import nudity_prompts_100
from prompts.tench_prompts import tench_prompts


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--unet_checkpoint",
        type=str,
        default=None,
    )
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--target_concept", choices=["nudity", "tench"], default="tench", type=str)
    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    return args


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_pipeline(unet_checkpoint, device):
    pipe_original = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    if unet_checkpoint is not None:
        pipe.unet.load_state_dict(
            torch.load(unet_checkpoint, map_location="cpu"), strict=False
        )
    pipe = pipe.to(device)

    return pipe_original, pipe


def generate_images(pipe, prompts, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    generator = torch.Generator(device=device)

    for i, prompt in enumerate(tqdm(prompts)):
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


def calculate_dino_score_image_to_image(image_dir1, image_dir2, device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device)
    model.eval()

    transform = Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dino_scores = []

    # For pairing images
    image_files = sorted(
        [
            f
            for f in os.listdir(image_dir1)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    for image_file in tqdm(image_files):
        path1 = os.path.join(image_dir1, image_file)
        path2 = os.path.join(image_dir2, image_file)

        image1 = Image.open(path1).convert("RGB")
        image2 = Image.open(path2).convert("RGB")

        # Preprocess images
        tensor1 = transform(image1).unsqueeze(0).to(device)
        tensor2 = transform(image2).unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract features using DINO
            feat1 = model(tensor1)
            feat2 = model(tensor2)

            # Normalize features
            feat1 = feat1 / feat1.norm(p=2, dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(p=2, dim=-1, keepdim=True)

            # Calculate cosine similarity
            similarity = (feat1 @ feat2.T).item()
            # Scale from [-1, 1] to [0, 100]
            scaled_similarity = (similarity + 1) * 50.0
            dino_scores.append(scaled_similarity)

    mean_score = np.mean(dino_scores)
    std_score = np.std(dino_scores)

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

    if args.target_concept == "nudity":
        prompts = nudity_prompts_100
    elif args.target_concept == "tench":
        prompts = tench_prompts

    # Setup output directories
    generated_images_dir_original = os.path.join(
        output_path, "generated_images_original"
    )
    generated_images_dir_erased = os.path.join(output_path, "generated_images_erased")
    os.makedirs(generated_images_dir_original, exist_ok=True)
    os.makedirs(generated_images_dir_erased, exist_ok=True)

    # Setup pipeline
    pipe_original, pipe_erased = setup_pipeline(args.unet_checkpoint, device)

    generate_images(pipe_original, prompts, generated_images_dir_original, device)

    generate_images(pipe_erased, prompts, generated_images_dir_erased, device)

    # Calculate Image-to-Image DINO scores
    dino_mean, dino_std = calculate_dino_score_image_to_image(
        generated_images_dir_original, generated_images_dir_erased, device
    )

    # Print results
    print(f"Consistency Score: {dino_mean:.2f} ± {dino_std:.2f}")

    # Save results to file
    results_file = os.path.join(output_path, "evaluation_results.txt")
    with open(results_file, "a") as f:
        f.write(f"Consistency Evaluation Results\n")
        f.write(f"Consistency Score: {dino_mean:.2f} ± {dino_std:.2f}\n")
        f.write("")

    # Clean up GPU memory
    del pipe_original, pipe_erased
    torch.cuda.empty_cache()

    # Clean up generated images
    shutil.rmtree(generated_images_dir_original)
    shutil.rmtree(generated_images_dir_erased)


if __name__ == "__main__":
    main()
