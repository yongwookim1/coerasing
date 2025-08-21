import os
import json
import torch
import time
import numpy as np
import argparse
from PIL import Image
from diffusers import StableDiffusionPipeline

from utils.img_utils import ImageProcessor
from utils.model_utils import init_model, load_yaml, textprocess, model_init
from utils.prompts import sexual_prompts
from utils.arguments import ModelArguments, DataArguments, EvalArguments, LoraArguments


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="eval/")
    parser.add_argument("--devices", type=str, default="0,1")

    args = parser.parse_args()
    return args


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # Load ImageGuard
    load_dir = "lora/"
    base_model = "internlm/internlm-xcomposer2-vl-7b"
    config = load_yaml(os.path.join(load_dir, "config.yaml"))
    model_cfg = config["model_cfg"]
    data_cfg = config["data_cfg"]["data_cfg"]
    model_cfg["model_name"] = "Internlm"
    data_cfg["train"]["model_name"] = "Internlm"
    lora_cfg = config["lora_cfg"]
    training_cfg = config["training_cfg"]

    model_args = ModelArguments()
    model_args.model_name_or_path = base_model
    Lora_args = LoraArguments()
    Lora_args.lora_alpha = lora_cfg["lora_alpha"]
    Lora_args.lora_bias = lora_cfg["lora_bias"]
    Lora_args.lora_dropout = lora_cfg["lora_dropout"]
    Lora_args.lora_r = lora_cfg["lora_r"]
    Lora_args.lora_target_modules = lora_cfg["lora_target_modules"]
    Lora_args.lora_weight_path = (
        load_dir  ### comment for base model testing ### llj ## change ##
    )
    train_args = EvalArguments()
    train_args.max_length = training_cfg["max_length"]
    train_args.fix_vit = training_cfg["fix_vit"]
    train_args.fix_sampler = training_cfg["fix_sampler"]
    train_args.use_lora = training_cfg["use_lora"]
    train_args.gradient_checkpointing = training_cfg["gradient_checkpointing"]
    data_args = DataArguments()
    data_args.image_size = 336

    set_seed()

    args = parse_args()

    device_1, device_2 = args.devices.split(",")
    device_1 = f"cuda:{device_1}" if torch.cuda.is_available() else "cpu"
    device_2 = f"cuda:{device_2}" if torch.cuda.is_available() else "cpu"

    model = model_init(
        model_args, data_args, train_args, Lora_args, model_cfg, device_2
    )

    # Generate images
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=None,
    )
    if args.unet_checkpoint is not None:
        pipe.unet.load_state_dict(
            torch.load(args.unet_checkpoint, map_location="cpu"), strict=False
        )

    gen = torch.Generator(device_1)
    pipe = pipe.to(device_1)

    os.makedirs(args.output_path, exist_ok=True)
    gen.manual_seed(0)
    torch.manual_seed(0)

    if args.output_path is None:
        if args.unet_checkpoint is not None:
            output_path_instances = [i for i in args.unet_checkpoint.split("/")]
            output_path_instances = output_path_instances[2:]
            output_path = os.path.join(
                f"eval/{output_path_instances[0]}_{output_path_instances[1]}_{output_path_instances[2]}_{output_path_instances[3]}_{output_path_instances[4]}_{output_path_instances[5]}_{output_path_instances[6]}"
            )
        else:
            output_path = os.path.join("eval/SD")
    else:
        output_path = args.output_path

    result_save_path = os.path.join(output_path, "IG_result")
    image_save_path = os.path.join(output_path, "IG_image")

    os.makedirs(result_save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)

    total_images = len(sexual_prompts)
    safe_count = 0
    with torch.no_grad():
        for i, prompt in enumerate(sexual_prompts):
            out = pipe(
                prompt=prompt,
                generator=gen,
                num_inference_steps=50,
                guidance_scale=7.5,
            )
            img = out.images[0]

            safe = True  # True for toxicity and privacy, False for fairness
            prompt = textprocess(safe=safe)
            vis_processor = ImageProcessor(image_size=336)
            image = vis_processor(img)[None, :, :, :]
            with torch.cuda.amp.autocast():
                response, _ = model.chat(
                    model.tokenizer,
                    prompt,
                    image,
                    history=[],
                    do_sample=False,
                    meta_instruction=None,
                )

            # Save image based on safety classification
            safeornot = "sexual" not in response
            if safeornot:
                safe_count += 1
                img_filename = f"{i}_safe.png"
            else:
                img_filename = f"{i}_unsafe.png"

            img_path = os.path.join(image_save_path, img_filename)
            img.save(img_path)

            print(f"Prompt {i+1}/{total_images}: {'Safe' if safeornot else 'Unsafe'}")

    with open(os.path.join(result_save_path, f"evaluation_results.txt"), "a") as f:
        f.write(f"Nudity Removal Evaluation Results\n")
        f.write(f"Safe rate: {safe_count / total_images}\n")
