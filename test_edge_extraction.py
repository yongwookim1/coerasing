import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob


def extract_edges(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    return edges


def compare_seed_vs_van_gogh(dataset_root="/home/dataset/unlearncanvas", class_name="Cats", image_num="1"):
    # Path setup
    seed_path = os.path.join(dataset_root, "Seed_Images", class_name, f"{image_num}.jpg")
    van_gogh_path = os.path.join(dataset_root, "Van_Gogh", class_name, f"{image_num}.jpg")

    if not os.path.exists(seed_path) or not os.path.exists(van_gogh_path):
        print(f"Images not found: {class_name}/{image_num}")
        return

    # Load images and resize to same dimensions
    seed_image = Image.open(seed_path).convert("RGB")
    van_gogh_image = Image.open(van_gogh_path).convert("RGB")
    
    # Resize van_gogh_image to match seed_image dimensions
    seed_size = seed_image.size
    van_gogh_image = van_gogh_image.resize(seed_size, Image.Resampling.LANCZOS)
    
    # Extract edges
    seed_edges = extract_edges(seed_image)
    van_gogh_edges = extract_edges(van_gogh_image)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].imshow(seed_image)
    axes[0, 0].set_title(f"Seed Image ({class_name})", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(van_gogh_image)
    axes[0, 1].set_title(f"Van Gogh Style ({class_name})", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(seed_edges, cmap='gray')
    axes[1, 0].set_title("Seed Edges", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(van_gogh_edges, cmap='gray')
    axes[1, 1].set_title("Van Gogh Edges", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    # Save results
    output_dir = "edge_comparison_results"
    os.makedirs(output_dir, exist_ok=True)

    comparison_output_path = os.path.join(output_dir, f"{class_name}_{image_num}_seed_vs_van_gogh.jpg")
    seed_edge_path = os.path.join(output_dir, f"{class_name}_{image_num}_seed_edges.jpg")
    van_gogh_edge_path = os.path.join(output_dir, f"{class_name}_{image_num}_van_gogh_edges.jpg")

    Image.fromarray(seed_edges).convert("L").save(seed_edge_path)
    Image.fromarray(van_gogh_edges).convert("L").save(van_gogh_edge_path)
    plt.savefig(comparison_output_path, bbox_inches='tight', dpi=150)
    plt.show()

    # Edge difference analysis
    edge_diff = cv2.absdiff(seed_edges, van_gogh_edges)
    diff_pixels = np.sum(edge_diff > 0)
    total_pixels = edge_diff.shape[0] * edge_diff.shape[1]
    diff_percentage = (diff_pixels / total_pixels) * 100

    print(f"Edge difference: {diff_percentage:.2f}% ({diff_pixels:,}/{total_pixels:,} pixels)")

    return seed_edges, van_gogh_edges, edge_diff


if __name__ == "__main__":
    dataset_root = "/home/dataset/unlearncanvas"

    compare_seed_vs_van_gogh(dataset_root, "Cats", "1")
