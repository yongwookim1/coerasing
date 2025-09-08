import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image, ImageFilter
import numpy as np
import random

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image_path", type=str, default="./input_images")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--prompt", type=str, default="A high-quality photo, following given edge")
    parser.add_argument("--edge_threshold", type=int, default=175)
    parser.add_argument("--edge_intensity", type=float, default=1.0)
    parser.add_argument("--blur_radius", type=int, default=0)
    parser.add_argument("--save_edges", action="store_true", help="Save extracted edges")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--controlnet_id",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
    )

    return parser.parse_args()


def set_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_canny_edges(image_path, edge_threshold=100, edge_intensity=1.0, blur_radius=0):
    """Extract simple Canny edges without noise processing"""
    import cv2
    
    # Load image with OpenCV
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple Canny edge detection
    edges = cv2.Canny(gray, edge_threshold, edge_threshold * 2)
    
    # Convert to PIL and RGB
    edges_pil = Image.fromarray(edges).convert("RGB")
    return edges_pil


def setup_pipeline(device, model_path, controlnet_id, torch_dtype):
    # ControlNet pipeline for edge guidance - same as AS file
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch_dtype
    )
    controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(f"cuda:{device}")

    return controlnet_pipeline


def create_edge_guided_prompt(original_prompt, edges_description="with clear edges and structure"):
    """Create a prompt that incorporates edge information"""
    return f"{original_prompt}, {edges_description}, high quality, detailed"


def generate_image_from_edges(args, pipe):
    image_paths = glob.glob(os.path.join(args.image_path, "*.*"), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for _, input_path in enumerate(tqdm(sorted(image_paths), desc="Processing images")):
        # Extract Canny edges using OpenCV
        edge_image = extract_canny_edges(
            input_path, 
            args.edge_threshold, 
            args.edge_intensity,
            args.blur_radius
        )
        edge_image = edge_image.resize((512, 512))
        
        base = os.path.splitext(os.path.basename(input_path))[0]
        save_dir = os.path.join(args.output_dir, base)
        os.makedirs(save_dir, exist_ok=True)

        # Save Canny edges if requested
        if args.save_edges:
            edge_image.save(os.path.join(save_dir, f"{base}_canny.png"))

        # Use the prompt directly - let ControlNet handle the edge guidance
        edge_guided_prompt = args.prompt

        # Generate images using ControlNet pipeline - same as AS file
        for i in range(5):
            generator = torch.Generator(f"cuda:{args.device}").manual_seed(i)
            with torch.no_grad():
                image = pipe(
                    prompt=edge_guided_prompt,
                    image=edge_image,
                    negative_prompt="blurry, low quality",
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                ).images[0]
            image.save(os.path.join(save_dir, f"output_{i}.png"))


if __name__ == "__main__":
    set_seed()

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32
    pipe = setup_pipeline(
        device=args.device,
        model_path=args.model_path,
        controlnet_id=args.controlnet_id,
        torch_dtype=torch_dtype,
    )

    generate_image_from_edges(args=args, pipe=pipe)
