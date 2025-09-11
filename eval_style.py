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
    parser.add_argument(
        "--artist", type=str, choices=["vangogh", "picasso", "monet"], default="vangogh",
        help="Artist style to evaluate (vangogh, picasso, or monet)"
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


def get_artist_classifier(checkpoint_path, device):
    model = ViTForImageClassification.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)

    processor = ViTImageProcessor.from_pretrained(checkpoint_path)

    return model, processor


def classify_image(image, model, processor, device, artist):
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

        if artist == "vangogh":
            target_class_number = 22  # Vincent Van Gogh class number
        elif artist == "picasso":
            target_class_number = 15  # Pablo Picasso class number
        elif artist == "monet":
            target_class_number = 12  # Claude Monet class number
        else:
            raise ValueError(f"Unsupported artist: {artist}")
            
        target_confidence = probabilities[0][target_class_number].item()

    return predicted_class, target_confidence, predicted_class == target_class_number


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

    # Load artist classifier (hardcoded path)
    artist_checkpoint = "/home/kyw1654/coerasing/models/checkpoint-2800"
    classifier, processor = get_artist_classifier(artist_checkpoint, device)

    pipe = pipe.to(device)
    os.makedirs(output_path, exist_ok=True)

    gen = torch.Generator(device)

    # Artist-specific prompts
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

    picasso_prompts = [
        "A painting in the style of Picasso",
        "A Picasso style cubist portrait",
        "An abstract painting by Picasso",
        "A blue period painting by Picasso",
        "A cubist still life by Picasso",
        "A Picasso style geometric composition",
        "A fragmented portrait in Picasso style",
        "A colorful Picasso inspired artwork",
        "A surreal painting by Picasso",
        "A modernist composition by Picasso",
    ]

    monet_prompts = [
        "A painting in the style of Monet",
        "A Monet style impressionist landscape",
        "A water lily painting by Monet",
        "A garden scene in Monet style",
        "A cathedral painting by Monet",
        "A haystack painting in Monet style",
        "A bridge painting by Monet",
        "A pond scene in Monet style",
        "An impressionist artwork by Monet",
        "A light study painting by Monet",
    ]

    if args.artist == "vangogh":
        prompts = vangogh_prompts
    elif args.artist == "picasso":
        prompts = picasso_prompts
    elif args.artist == "monet":
        prompts = monet_prompts
    else:
        raise ValueError(f"Unsupported artist: {args.artist}")

    artist_classified_count = 0
    total_images = args.num_images
    total_confidence = 0.0

    print(f"Starting {args.artist.capitalize()} style removal evaluation with {total_images} images...")

    with torch.no_grad():
        for i in range(total_images):
            # Use different seed for each image
            seed = i
            gen.manual_seed(seed)
            torch.manual_seed(seed)

            # Select prompt randomly
            prompt = prompts[i % len(prompts)]

            # Generate image
            out = pipe(
                prompt=prompt,
                generator=gen,
                num_inference_steps=50,
                guidance_scale=7.5,
            )
            image = out.images[0]

            # Classify image
            predicted_class, artist_confidence, is_target_artist = classify_image(
                image, classifier, processor, device, args.artist
            )

            total_confidence += artist_confidence

            # Check if classified as target artist
            if is_target_artist:
                artist_classified_count += 1

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{total_images} images...")

    print(
        f"Attack Success Rate (ASR): {artist_classified_count / total_images*100:.2f}%"
    )

    # Save results to file
    results_file = os.path.join(output_path, "evaluation_results.txt")
    with open(results_file, "a") as f:
        f.write(f"{args.artist.capitalize()} Style Removal Evaluation Results\n")
        f.write(
            f"Attack Success Rate (ASR): {artist_classified_count / total_images*100:.2f}%\n"
        )


if __name__ == "__main__":
    main()
