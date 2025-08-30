import torch
from diffusers import StableDiffusionPipeline

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=str, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="0")

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
        pipe.unet.load_state_dict(torch.load(args.unet_checkpoint, map_location="cpu"), strict=False)

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    gen = torch.Generator(device)
    pipe = pipe.to(device)

    gen.manual_seed(0)
    torch.manual_seed(0)
    
    if args.output_path is None:
        if args.unet_checkpoint is not None:
            save_path_instances = [i for i in args.unet_checkpoint.split('/')]
            save_path_instances = save_path_instances[2:]
            output_path = os.path.join(f"eval/{save_path_instances[0]}_{save_path_instances[1]}_{save_path_instances[2]}_{save_path_instances[3]}_{save_path_instances[4]}_{save_path_instances[5]}_{save_path_instances[6]}")
        else:
            output_path = os.path.join("eval/SD")
    else:
        output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        for i in range(5):
            gen.manual_seed(i)
            torch.manual_seed(i)
            out = pipe(prompt=[args.prompt],
                       generator=gen,
                       num_inference_steps=args.num_inference_steps,
                       guidance_scale=args.guidance_scale,
                       )
            image = out.images[0]
            # Save image
            filename = '_'.join(args.prompt.split(" "))
            image.save(os.path.join(output_path, f"{filename}_{i}.png"))
    
    print(f"[Save] Saved images at {output_path}")


if __name__ == "__main__":
    main()
