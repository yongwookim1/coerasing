import torch
from diffusers import StableDiffusionPipeline
from utils.model_utils import add_lora_to_unet

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--num_inference_steps", type=str, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_dir", type=str, default="eval/")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=None,
    )

    if args.unet_checkpoint is not None:
        pipe.unet.load_state_dict(torch.load(args.unet_checkpoint), strict=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    gen = torch.Generator(device)
    pipe = pipe.to(device)

    os.makedirs(args.output_dir, exist_ok=True)
    gen.manual_seed(0)
    torch.manual_seed(0)
    
    save_path_instances = [i for i in args.unet_checkpoint.split('/')]
    save_path_instances = save_path_instances[2:7]
    
    save_path = os.path.join(f"eval/{save_path_instances[0]}_{save_path_instances[1]}_{save_path_instances[2]}_{save_path_instances[3]}_{save_path_instances[4]}")
    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for i in range(5):
            gen.manual_seed(i)
            torch.manual_seed(i)
            out = pipe(prompt=[args.prompt], generator=gen,
                       num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale)
            image = out.images[0]
            # Save image
            filename = '_'.join(args.prompt.split(" "))
            image.save(os.path.join(save_path, f"{filename}_{i}.png"))
    
    print(f"[Save] Saved images at {save_path}")


if __name__ == "__main__":
    main()
