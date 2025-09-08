import os
import glob
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image_path", type=str, default="./edges")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--prompt", type=str, default="A photo of cat")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.5)
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
    )
    parser.add_argument(
        "--controlnet_id",
        type=str,
        default="lllyasviel/control_v11p_sd15_canny",
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


def setup_pipeline(device, model_id, controlnet_id, torch_dtype):
    controlnet = ControlNetModel.from_pretrained(
        controlnet_id,
        torch_dtype=torch_dtype,
    ).to(f"cuda:{device}")

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(f"cuda:{device}")
    pipe.set_progress_bar_config(disable=True)

    return pipe


def generate_image(args, pipe):
    image_paths = glob.glob(os.path.join(args.image_path, "*.*"), recursive=True)
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for _, edge_path in enumerate(sorted(image_paths)):
        control_image = load_image(edge_path)
        control_image = control_image.resize((512, 512))
        base = os.path.splitext(os.path.basename(edge_path))[0]

        save_dir = os.path.join(args.output_dir, base)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(0, 5):
            with torch.no_grad():
                image = pipe(
                    prompt=args.prompt,
                    image=control_image,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                ).images[0]
            image.save(os.path.join(save_dir, f"output_{int(i)}.png"))


if __name__ == "__main__":
    set_seed()

    args = parse_args()

    use_cuda = torch.cuda.is_available()
    torch_dtype = torch.float16 if use_cuda else torch.float32
    pipe = setup_pipeline(
        device=args.device,
        model_id=args.model_id,
        controlnet_id=args.controlnet_id,
        torch_dtype=torch_dtype,
    )

    generate_image(args=args, pipe=pipe)
