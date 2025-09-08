import os
import argparse
import time
import io
import random
import re

from tqdm.auto import tqdm
from PIL import Image

import google.generativeai as genai

from prompts.vangogh_prompts import vangogh_prompts


def extract_prompt_index_from_filename(filename):
    name_without_ext = os.path.splitext(filename)[0]
    parts = name_without_ext.split("_")
    if len(parts) >= 3 and parts[0] == "vangogh":
        # Get the second part which is the prompt index (00, 01, 02, etc.)
        prompt_index = int(parts[1])
        return prompt_index % len(vangogh_prompts)  # Ensure within bounds
    return 0  # Default to first prompt


def clean_prompt_for_content(vangogh_prompt):
    text = vangogh_prompt

    # Remove artist references (case-insensitive)
    text = re.sub(r"\bby\s+vincent\s+van\s+gogh\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvincent\s+van\s+gogh\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bvan\s+gogh\b", "", text, flags=re.IGNORECASE)

    # Remove dangling 'by' at end if any
    text = re.sub(r"\bby\b\s*$", "", text, flags=re.IGNORECASE)

    # Tidy whitespace and trailing punctuation
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"[,\-–—]+\s*$", "", text).strip()

    return text


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--output_path", type=str, default="./data/")
    parser.add_argument(
        "--input_forget_dir",
        type=str,
        default="./data/AS_forget_data2",
    )

    parser.add_argument("--num_images", type=int, default=10)

    return parser.parse_args()


def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
    return model


def create_directories(root_dir):
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    retain_save_dir = os.path.join(root_dir, f"AS_retain_data_nano")
    comparison_save_dir = os.path.join(root_dir, f"AS_comparison_data_nano")

    for directory in [retain_save_dir, comparison_save_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    return retain_save_dir, comparison_save_dir


def generate_retain_image_gemini(model, forget_image, prompt_index=0):
    # Get specific Van Gogh artwork prompt based on index
    artwork_prompt = vangogh_prompts[prompt_index]
    content_prompt = clean_prompt_for_content(artwork_prompt)
 
    instruction = (
        f"Remove Vicent Van Gogh style from this painting of '{content_prompt}'. "
        "Convert it to a photorealistic image while maintaining the same composition and scene."
    )

    # Load and prepare image
    image = Image.open(forget_image).convert("RGB")

    # Generate content with image and text
    response = model.generate_content([image, instruction])

    # Check if response has image data
    if hasattr(response, "parts") and response.parts:
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data:
                # Handle image data
                img_data = part.inline_data.data
                img = Image.open(io.BytesIO(img_data))
                img = img.convert("RGB")
                return img

    # Fallback: if no image in response, try to get from response._result
    if hasattr(response, "_result") and hasattr(response._result, "candidates"):
        for candidate in response._result.candidates:
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        img_data = part.inline_data.data
                        img = Image.open(io.BytesIO(img_data))
                        img = img.convert("RGB")
                        return img

    raise RuntimeError("No image generated from API response")


def create_comparison_image(forget_image, retain_image):
    forget_width, forget_height = forget_image.size
    retain_width, retain_height = retain_image.size
    target_height = min(forget_height, retain_height)
    forget_resized = forget_image.resize(
        (int(forget_width * target_height / forget_height), target_height)
    )
    retain_resized = retain_image.resize(
        (int(retain_width * target_height / retain_height), target_height)
    )
    total_width = forget_resized.width + retain_resized.width
    comparison_image = Image.new("RGB", (total_width, target_height))
    comparison_image.paste(forget_resized, (0, 0))
    comparison_image.paste(retain_resized, (forget_resized.width, 0))
    return comparison_image


def generate_paired_data(
    input_forget_dir,
    model,
    retain_save_dir,
    comparison_save_dir,
    num_images,
):
    # Set random seed for reproducible results
    random.seed(42)

    files = [
        f
        for f in os.listdir(input_forget_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Randomly sample files instead of taking first num_images
    if len(files) > num_images:
        files = random.sample(files, num_images)
    else:
        files = files[:num_images]

    success_count = 0
    for i, fname in enumerate(tqdm(files)):
        fpath = os.path.join(input_forget_dir, fname)

        # Extract prompt index from filename
        prompt_index = extract_prompt_index_from_filename(fname)

        # Gemini style removal with specific prompt
        retain_image = generate_retain_image_gemini(model, fpath, prompt_index)
        realistic_name = f"realistic_{i}.png"
        retain_image.save(os.path.join(retain_save_dir, realistic_name))

        # Comparison
        forget_image = Image.open(fpath).convert("RGB")
        comparison_image = create_comparison_image(forget_image, retain_image)
        comparison_name = f"comparison_{i}.png"
        comparison_image.save(os.path.join(comparison_save_dir, comparison_name))
        success_count += 1

    print(f"Processed {success_count} images.")


def main():
    args = parse_args()
    # Nano-only: Gemini for style removal
    model = setup_gemini()

    retain_save_dir, comparison_save_dir = create_directories(args.output_path)

    # Resolve input forget dir: prefer AS_forget_data2, then AS_forget_data, else provided path
    input_dir = args.input_forget_dir

    generate_paired_data(
        input_forget_dir=input_dir,
        model=model,
        retain_save_dir=retain_save_dir,
        comparison_save_dir=comparison_save_dir,
        num_images=args.num_images,
    )

    print(f"Style-removed data saved to: {retain_save_dir}")
    print(f"Comparison data saved to: {comparison_save_dir}")


if __name__ == "__main__":
    main()
