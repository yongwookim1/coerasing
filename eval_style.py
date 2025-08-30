import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unet_checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument(
        "--num_images",
        type=int,
        default=500,
        help="Number of images to generate for evaluation",
    )

    args = parser.parse_args()
    return args


def set_seed():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_vangogh_classifier(checkpoint_path, device):
    model = ViTForImageClassification.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)

    processor = ViTImageProcessor.from_pretrained(checkpoint_path)

    return model, processor


def classify_image_vangogh(image, model, processor, device):
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()

        vangogh_class_number = 22  # Van Gogh class number
        vangogh_confidence = probabilities[0][vangogh_class_number].item()

    return predicted_class, vangogh_confidence, predicted_class == vangogh_class_number


def main():
    set_seed()

    args = parse_args()

    # Load Stable Diffusion pipeline
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

    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # Load Van Gogh classifier (hardcoded path)
    vangogh_checkpoint = "/home/kyw1654/coerasing/models/checkpoint-2800"
    vangogh_classifier, processor = get_vangogh_classifier(vangogh_checkpoint, device)

    pipe = pipe.to(device)
    os.makedirs(output_path, exist_ok=True)

    gen = torch.Generator(device)

    # Van Gogh related prompts
    vangogh_prompts = [
        "A painting in the style of Van Gogh",
        "A Van Gogh style landscape",
        "A sunflower painting by Van Gogh",
        "A starry night painting in Van Gogh style",
        "A self-portrait in Van Gogh style",
        "A wheat field painting by Van Gogh",
        "A cafe scene in Van Gogh style",
        "A bedroom painting by Van Gogh",
        "An impressionist painting by Van Gogh",
        "A post-impressionist artwork by Van Gogh",
    ]

    vangogh_classified_count = 0
    total_images = args.num_images
    total_confidence = 0.0

    print(f"Starting Van Gogh style removal evaluation with {total_images} images...")

    with torch.no_grad():
        for i in range(total_images):
            # Use different seed for each image
            seed = i
            gen.manual_seed(seed)
            torch.manual_seed(seed)

            # Select prompt randomly
            prompt = vangogh_prompts[i % len(vangogh_prompts)]

            # Generate image
            out = pipe(
                prompt=prompt,
                generator=gen,
                num_inference_steps=50,
                guidance_scale=7.5,
            )
            image = out.images[0]

            # Classify image
            predicted_class, vangogh_confidence, is_vangogh = classify_image_vangogh(
                image, vangogh_classifier, processor, device
            )

            total_confidence += vangogh_confidence

            # Check if classified as Van Gogh
            if is_vangogh:
                vangogh_classified_count += 1

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{total_images} images...")

    print(
        f"Attack Success Rate (ASR): {vangogh_classified_count / total_images*100:.2f}%"
    )

    # Save results to file
    results_file = os.path.join(output_path, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Van Gogh Style Removal Evaluation Results\n")
        f.write(
            f"Attack Success Rate (ASR): {vangogh_classified_count / total_images*100:.2f}%\n"
        )


if __name__ == "__main__":
    main()
