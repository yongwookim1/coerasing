import sys
import os
import argparse
from tqdm.auto import tqdm
from PIL import Image, ImageFilter
import random
import numpy as np

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--numbers_per_class", type=int, default=1)
    parser.add_argument("--device", type=str, default="4,0")
    
    return parser.parse_args()


def load_vangogh_prompts():
    vangogh_prompts = [
        "a beautiful landscape in the style of Van Gogh, swirling brushstrokes, vibrant colors",
        "a starry night scene in Van Gogh style, thick impasto texture, dynamic brushwork",
        "a sunflower field painted in Van Gogh style, bold yellow colors, expressive strokes",
        "a village scene in Van Gogh style, post-impressionist painting, textured brushstrokes",
        "a self-portrait in Van Gogh style, intense colors, emotional brushwork",
        "a cafe terrace at night in Van Gogh style, vivid colors, swirling patterns",
        "a wheat field with cypresses in Van Gogh style, dramatic sky, thick paint texture",
        "a bedroom interior in Van Gogh style, simple furniture, bold outlines",
        "a bridge over water in Van Gogh style, flowing brushstrokes, impressionist technique",
        "a garden scene in Van Gogh style, colorful flowers, energetic painting style",
        "a street scene in Van Gogh style, vibrant atmosphere, post-impressionist technique",
        "a portrait of a peasant in Van Gogh style, earthy colors, expressive face",
        "a still life with fruit in Van Gogh style, bold colors, thick brushstrokes",
        "a church in a village in Van Gogh style, dramatic lighting, swirling sky",
        "a Japanese bridge in Van Gogh style, water lilies, impressionist brushwork"
    ]
    return vangogh_prompts


def setup_pipelines(model_path, device_0, device_1):
    txt2img_pipeline = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(f"cuda:{device_0}")
    
    # ControlNet pipeline for style removal using edge guidance
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16
    )
    controlnet_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(f"cuda:{device_1}")
    
    return txt2img_pipeline, controlnet_pipeline


def extract_canny_edges(image, edge_threshold=10, edge_intensity=0.1, blur_radius=1):
    if image.mode != 'L':
        gray_image = image.convert('L')
    else:
        gray_image = image
    
    # Apply Gaussian blur before edge detection for smoother edges
    if blur_radius > 0:
        gray_image = gray_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Edge detection using PIL
    edges = gray_image.filter(ImageFilter.FIND_EDGES)
    
    # Apply threshold to reduce noise and make edges more subtle
    edges_np = np.array(edges)
    edges_np = np.where(edges_np > edge_threshold, edges_np, 0)
    
    # Reduce edge intensity to make them more subtle
    edges_np = (edges_np * edge_intensity).astype(np.uint8)
    
    # Convert back to PIL and apply additional blur for softer edges
    edges_pil = Image.fromarray(edges_np)
    if blur_radius > 0:
        edges_pil = edges_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius * 0.5))
    
    # Convert to RGB
    edges_pil = edges_pil.convert("RGB")
    return edges_pil


def create_directories(root_dir, numbers_per_class, num_prompts):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    
    # Create subdirectories
    total_images = numbers_per_class * num_prompts
    forget_save_dir = os.path.join(root_dir, f"AS_forget_data")
    retain_save_dir = os.path.join(root_dir, f"AS_retain_data")
    comparison_save_dir = os.path.join(root_dir, f"AS_comparison_data")
    canny_save_dir = os.path.join(root_dir, f"AS_canny_edges")
    
    for directory in [forget_save_dir, retain_save_dir, comparison_save_dir, canny_save_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    
    return forget_save_dir, retain_save_dir, comparison_save_dir, canny_save_dir


def clean_prompt_for_content(vangogh_prompt):
    content_prompt = vangogh_prompt.replace("in the style of Van Gogh", "")
    content_prompt = content_prompt.replace("in Van Gogh style", "")
    content_prompt = content_prompt.replace("swirling brushstrokes", "")
    content_prompt = content_prompt.replace("thick impasto texture", "")
    content_prompt = content_prompt.replace("dynamic brushwork", "")
    content_prompt = content_prompt.replace("textured brushstrokes", "")
    content_prompt = content_prompt.replace("expressive strokes", "")
    content_prompt = content_prompt.replace("post-impressionist painting", "")
    content_prompt = content_prompt.replace("emotional brushwork", "")
    content_prompt = content_prompt.replace("swirling patterns", "")
    content_prompt = content_prompt.replace("thick paint texture", "")
    content_prompt = content_prompt.replace("flowing brushstrokes", "")
    content_prompt = content_prompt.replace("impressionist technique", "")
    content_prompt = content_prompt.replace("energetic painting style", "")
    content_prompt = content_prompt.replace("post-impressionist technique", "")
    content_prompt = content_prompt.replace("thick brushstrokes", "")
    content_prompt = content_prompt.replace("impressionist brushwork", "")
    content_prompt = content_prompt.strip().replace("  ", " ")
    
    return content_prompt


def generate_forget_image(txt2img_pipeline, prompt, generator):
    forget_image = txt2img_pipeline(
        prompt=prompt,
        generator=generator,
        negative_prompt="blurry, low quality, distorted",
        guidance_scale=7.5,
        num_inference_steps=50
    ).images[0]
    return forget_image


def generate_retain_image(controlnet_pipeline, canny_edges, vangogh_prompt, seed, device_1):
    content_prompt = clean_prompt_for_content(vangogh_prompt)
    retain_prompt = f"painting of {content_prompt}, artistic, high quality, detailed"
    
    retain_image = controlnet_pipeline(
        prompt=retain_prompt,
        image=canny_edges,
        negative_prompt="Van Gogh style, artistic, painting, brushstrokes, impressionist, blurry, low quality",
        generator=torch.Generator(f"cuda:{device_1}").manual_seed(seed),
        guidance_scale=7.5,
        num_inference_steps=20,
        controlnet_conditioning_scale=1.0
    ).images[0]
    return retain_image


def create_comparison_image(forget_image, retain_image):
    forget_width, forget_height = forget_image.size
    retain_width, retain_height = retain_image.size
    
    # Ensure both images have the same height for proper alignment
    target_height = min(forget_height, retain_height)
    forget_resized = forget_image.resize((int(forget_width * target_height / forget_height), target_height))
    retain_resized = retain_image.resize((int(retain_width * target_height / retain_height), target_height))
    
    # Create combined image
    total_width = forget_resized.width + retain_resized.width
    comparison_image = Image.new("RGB", (total_width, target_height))
    comparison_image.paste(forget_resized, (0, 0))
    comparison_image.paste(retain_resized, (forget_resized.width, 0))
    
    return comparison_image


def generate_paired_data(prompts, txt2img_pipeline, controlnet_pipeline, forget_save_dir, retain_save_dir, comparison_save_dir, canny_save_dir, numbers_per_class, device_0, device_1):
    gen = torch.Generator(f"cuda:{device_0}")
    
    for prompt_idx, vangogh_prompt in enumerate(tqdm(prompts, desc="Generating Van Gogh style removal data")):
        for idx in range(numbers_per_class):
            gen.manual_seed(idx)
            torch.manual_seed(idx)
            
            print(f"Processing: {vangogh_prompt}")
            
            # Generate Van Gogh style image (forget data)
            forget_image = generate_forget_image(txt2img_pipeline, vangogh_prompt, gen)
            forget_filename = f"vangogh_{prompt_idx:02d}_{idx:04d}.png"
            forget_image.save(os.path.join(forget_save_dir, forget_filename))
            
            # Extract very subtle Canny edges from Van Gogh image
            canny_edges = extract_canny_edges(
                forget_image, 
                edge_threshold=20,
                edge_intensity=0.2,
                blur_radius=2
            )
            
            # Save Canny edges for inspection ###################
            canny_filename = f"canny_{prompt_idx:02d}_{idx:04d}.png"
            canny_edges.save(os.path.join(canny_save_dir, canny_filename))
            print(f"Canny edges saved: {canny_filename}")
            
            # Generate style-removed image using ControlNet
            retain_image = generate_retain_image(controlnet_pipeline, canny_edges, vangogh_prompt, idx, device_1)
            retain_filename = f"realistic_{prompt_idx:02d}_{idx:04d}.png"
            retain_image.save(os.path.join(retain_save_dir, retain_filename))
            
            # Create and save comparison image
            comparison_image = create_comparison_image(forget_image, retain_image)
            comparison_filename = f"comparison_{prompt_idx:02d}_{idx:04d}.png"
            comparison_image.save(os.path.join(comparison_save_dir, comparison_filename))
            
            print(f"Generated: {forget_filename} -> {retain_filename}")


def main():
    args = parse_args()
    device_0, device_1 = args.device.split(',')
    
    vangogh_prompts = load_vangogh_prompts()
    
    txt2img_pipeline, controlnet_pipeline = setup_pipelines(args.model_path, device_0, device_1)
    
    root_dir = "./data/"
    forget_save_dir, retain_save_dir, comparison_save_dir, canny_save_dir = create_directories(
        root_dir, args.numbers_per_class, len(vangogh_prompts)
    )
    
    # Generate paired data
    generate_paired_data(
        vangogh_prompts,
        txt2img_pipeline,
        controlnet_pipeline,
        forget_save_dir,
        retain_save_dir,
        comparison_save_dir,
        canny_save_dir,
        args.numbers_per_class,
        device_0,
        device_1
    )
    
    print("Generation completed!")
    print(f"Van Gogh style images saved in: {forget_save_dir}")
    print(f"Style-removed images saved in: {retain_save_dir}")
    print(f"Comparison images saved in: {comparison_save_dir}")
    print(f"Canny edges saved in: {canny_save_dir}")


if __name__ == "__main__":
    main()
