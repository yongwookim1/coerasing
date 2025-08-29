import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
import random
import argparse
from collections import defaultdict

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Pix2Pix Training/Inference for UnlearnCanvas dataset"
    )

    parser.add_argument(
        "--mode", type=str, choices=["train", "inference"], required=True
    )
    parser.add_argument(
        "--dataset_root", type=str, default="/home/dataset/unlearncanvas"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/unlearncanvas"
    )
    parser.add_argument("--device", type=int, default=0)

    # Training arguments - Modified to use all styles and data by default
    parser.add_argument("--styles", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--max_samples_per_class", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=10)

    # Inference arguments
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--input", type=str, default="/home/dataset/unlearncanvas/Van_Gogh/Cats/1.jpg"
    )
    parser.add_argument("--output_dir", type=str, default="results")

    return parser.parse_args()


def extract_edges(image):
    if isinstance(image, torch.Tensor):
        image = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 100, 150)
    return edges


class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()

        if down:
            self.conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, 4, 2, 1, bias=False
            )

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) if use_dropout else None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Pix2PixGenerator, self).__init__()

        # Encoder
        self.down1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True)
        self.down5 = UNetBlock(512, 512, down=True)
        self.down6 = UNetBlock(512, 512, down=True)
        self.down7 = UNetBlock(512, 512, down=True)
        self.down8 = UNetBlock(512, 512, down=True)

        # Decoder
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up3 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.up4 = UNetBlock(1024, 512, down=False)
        self.up5 = UNetBlock(1024, 256, down=False)
        self.up6 = UNetBlock(512, 128, down=False)
        self.up7 = UNetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))

        output = self.final(torch.cat([u7, d1], 1))
        return output


class Pix2PixDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super(Pix2PixDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, stylized_edges, realistic_edges):
        x = torch.cat([stylized_edges, realistic_edges], 1)
        return self.model(x)


class UnlearnCanvasDataset(Dataset):
    def __init__(
        self,
        dataset_root="/home/dataset/unlearncanvas",
        styles=None,
        max_samples_per_class=None,
        cache_edges=True,
        disk_cache=True,
    ):
        self.dataset_root = dataset_root
        self.seed_root = os.path.join(dataset_root, "Seed_Images")
        self.cache_edges = cache_edges
        self.disk_cache = disk_cache
        self.edge_cache_dir = os.path.join(dataset_root, ".edge_cache")

        all_styles = [
            d
            for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d)) and d != "Seed_Images"
        ]
        all_styles.sort()

        self.styles = styles if styles else all_styles

        all_classes = [
            d
            for d in os.listdir(self.seed_root)
            if os.path.isdir(os.path.join(self.seed_root, d))
        ]
        all_classes.sort()
        self.image_classes = all_classes

        self.max_samples_per_class = max_samples_per_class

        self.data_pairs = self._build_data_pairs()
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]  # 0~1 to -1~1
        )

        # Edge caching
        self.edge_cache = {}

        print(f"Dataset loaded: {len(self.data_pairs)} image pairs")

        if cache_edges:
            if disk_cache:
                os.makedirs(self.edge_cache_dir, exist_ok=True)
                if self._load_all_edges_from_disk():
                    print("Loaded edges from cache")
                else:
                    self._precompute_and_save_edges()
            else:
                self._precompute_edges()

    def _build_data_pairs(self):
        pairs = []

        for image_class in self.image_classes:
            seed_class_dir = os.path.join(self.seed_root, image_class)
            if not os.path.exists(seed_class_dir):
                continue

            seed_images = glob.glob(os.path.join(seed_class_dir, "*.jpg"))
            seed_images = sorted(
                seed_images, key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )

            if self.max_samples_per_class:
                seed_images = seed_images[: self.max_samples_per_class]

            for seed_path in seed_images:
                image_name = os.path.basename(seed_path)
                for style in self.styles:
                    styled_path = os.path.join(
                        self.dataset_root, style, image_class, image_name
                    )
                    if os.path.exists(styled_path):
                        pairs.append((seed_path, styled_path))

        random.shuffle(pairs)
        return pairs

    def _get_edge_cache_path(self, image_path, prefix=""):
        """Generate cache file path for edge data"""
        rel_path = os.path.relpath(image_path, self.dataset_root)
        cache_subdir = os.path.dirname(rel_path)
        filename = os.path.splitext(os.path.basename(rel_path))[0]

        cache_dir = os.path.join(self.edge_cache_dir, cache_subdir)
        os.makedirs(cache_dir, exist_ok=True)

        cache_filename = f"{prefix}{filename}_edges.npy"
        return os.path.join(cache_dir, cache_filename)

    def _load_or_compute_edge(self, image_path, prefix=""):
        """Load edge from cache or compute and save it"""
        cache_path = self._get_edge_cache_path(image_path, prefix)

        if os.path.exists(cache_path):
            edges = np.load(cache_path)
        else:
            image = Image.open(image_path).convert("RGB")
            edges = extract_edges(image)
            np.save(cache_path, edges)

        # Convert to PIL and apply transform
        edges_pil = Image.fromarray(edges).convert("L")
        edges_tensor = self.transform(edges_pil)
        return edges_tensor

    def _load_all_edges_from_disk(self):
        """Load all edges from disk cache if available"""
        missing_cache = False
        for i, (seed_path, styled_path) in enumerate(self.data_pairs):
            seed_cache_path = self._get_edge_cache_path(seed_path, "seed_")
            styled_cache_path = self._get_edge_cache_path(styled_path, "styled_")

            if not (
                os.path.exists(seed_cache_path) and os.path.exists(styled_cache_path)
            ):
                missing_cache = True
                break

        if missing_cache:
            return False

        # Load all cached edges
        for i, (seed_path, styled_path) in enumerate(
            tqdm(self.data_pairs, desc="Loading edges")
        ):
            seed_edges_tensor = self._load_or_compute_edge(seed_path, "seed_")
            styled_edges_tensor = self._load_or_compute_edge(styled_path, "styled_")
            self.edge_cache[i] = (styled_edges_tensor, seed_edges_tensor)

        return True

    def _precompute_and_save_edges(self):
        """Precompute all edges and save to disk cache"""
        for i, (seed_path, styled_path) in enumerate(
            tqdm(self.data_pairs, desc="Computing edges")
        ):
            seed_edges_tensor = self._load_or_compute_edge(seed_path, "seed_")
            styled_edges_tensor = self._load_or_compute_edge(styled_path, "styled_")
            self.edge_cache[i] = (styled_edges_tensor, seed_edges_tensor)

        print(f"Cached {len(self.edge_cache)} edge pairs")

    def _precompute_edges(self):
        for i, (seed_path, styled_path) in enumerate(
            tqdm(self.data_pairs, desc="Computing edges")
        ):
            seed_image = Image.open(seed_path).convert("RGB")
            styled_image = Image.open(styled_path).convert("RGB")

            seed_edges = extract_edges(seed_image)
            styled_edges = extract_edges(styled_image)

            # Convert to PIL and apply transform
            seed_edges_pil = Image.fromarray(seed_edges).convert("L")
            styled_edges_pil = Image.fromarray(styled_edges).convert("L")

            seed_edges_tensor = self.transform(seed_edges_pil)
            styled_edges_tensor = self.transform(styled_edges_pil)

            self.edge_cache[i] = (styled_edges_tensor, seed_edges_tensor)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        if self.cache_edges and idx in self.edge_cache:
            return self.edge_cache[idx]

        # Fallback to original method if caching disabled or cache miss
        seed_path, styled_path = self.data_pairs[idx]

        seed_image = Image.open(seed_path).convert("RGB")
        styled_image = Image.open(styled_path).convert("RGB")

        seed_edges = extract_edges(seed_image)
        styled_edges = extract_edges(styled_image)

        seed_edges_pil = Image.fromarray(seed_edges).convert("L")
        styled_edges_pil = Image.fromarray(styled_edges).convert("L")

        seed_edges_tensor = self.transform(seed_edges_pil)
        styled_edges_tensor = self.transform(styled_edges_pil)

        return styled_edges_tensor, seed_edges_tensor


class Trainer:
    def __init__(
        self,
        dataset_root="/home/dataset/unlearncanvas",
        styles=None,
        lr=0.0002,
        batch_size=16,
        max_samples_per_class=None,
        device=0,
        cache_edges=True,
    ):
        # Set device based on input
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

        self.generator = Pix2PixGenerator().to(self.device)
        self.discriminator = Pix2PixDiscriminator().to(self.device)

        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
        )

        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        dataset = UnlearnCanvasDataset(
            dataset_root=dataset_root,
            styles=styles,
            max_samples_per_class=max_samples_per_class,
            cache_edges=cache_edges,
            disk_cache=True,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train_step(self, stylized_edges, realistic_edges):
        # Train Discriminator
        self.opt_d.zero_grad()

        d_real = self.discriminator(stylized_edges, realistic_edges)
        real_labels = torch.ones_like(d_real).to(self.device)
        loss_d_real = self.criterion(d_real, real_labels)

        with torch.no_grad():
            generated_edges = self.generator(stylized_edges)
        d_fake = self.discriminator(stylized_edges, generated_edges)
        fake_labels = torch.zeros_like(d_fake).to(self.device)
        loss_d_fake = self.criterion(d_fake, fake_labels)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.opt_d.step()

        # Train Generator
        self.opt_g.zero_grad()

        generated_edges = self.generator(stylized_edges)
        d_gen = self.discriminator(stylized_edges, generated_edges)

        loss_g_adv = self.criterion(d_gen, real_labels)
        loss_g_l1 = self.l1_loss(generated_edges, realistic_edges) * 100

        loss_g = loss_g_adv + loss_g_l1
        loss_g.backward()
        self.opt_g.step()

        return {
            "loss_d": loss_d.item(),
            "loss_g": loss_g.item(),
            "loss_g_l1": loss_g_l1.item(),
        }

    def train(self, num_epochs=100, save_interval=10, checkpoint_dir="checkpoints"):
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_losses = defaultdict(float)

            self.generator.train()
            self.discriminator.train()

            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

            for stylized_edges, realistic_edges in progress_bar:
                stylized_edges = stylized_edges.to(self.device)
                realistic_edges = realistic_edges.to(self.device)

                losses = self.train_step(stylized_edges, realistic_edges)

                for key, value in losses.items():
                    epoch_losses[key] += value

                progress_bar.set_postfix(
                    {
                        "D_loss": f"{losses['loss_d']:.4f}",
                        "G_loss": f"{losses['loss_g']:.4f}",
                    }
                )

            num_batches = len(self.dataloader)
            avg_losses = {
                key: value / num_batches for key, value in epoch_losses.items()
            }

            print(
                f"Epoch {epoch+1}/{num_epochs} - D_loss: {avg_losses['loss_d']:.4f}, G_loss: {avg_losses['loss_g']:.4f}"
            )

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, checkpoint_dir)

    def save_checkpoint(self, epoch, checkpoint_dir):
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }

        checkpoint_path = os.path.join(checkpoint_dir, f"pix2pix_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)


class Pix2PixInference:
    def __init__(self, checkpoint_path, device=0):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device}")
        else:
            self.device = torch.device("cpu")

        # Load Pix2Pix generator
        self.generator = Pix2PixGenerator().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.generator.eval()

        # Load ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_canny",  # Updated to ControlNet v1.1
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )

        self.sd_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        )

        if self.device.type == "cuda":
            self.sd_pipe = self.sd_pipe.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    def process_image(self, input_path, output_dir="inference_results"):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        # Load stylized image
        stylized_image = Image.open(input_path).convert("RGB")

        # Extract stylized edges
        stylized_edges = extract_edges(stylized_image)
        stylized_edges_pil = Image.fromarray(stylized_edges).convert("L")

        # Convert to realistic edges using Pix2Pix
        stylized_tensor = (
            self.transform(stylized_edges_pil).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            realistic_edges_tensor = self.generator(stylized_tensor)

        realistic_edges_np = realistic_edges_tensor.squeeze().cpu().numpy()
        realistic_edges_np = ((realistic_edges_np + 1) / 2 * 255).astype(np.uint8)
        realistic_edges_pil = Image.fromarray(realistic_edges_np).convert("RGB")

        # Generate realistic image using ControlNet
        generated_image = self.sd_pipe(
            prompt=" ",
            image=realistic_edges_pil,
            num_inference_steps=50,
            guidance_scale=7.5,
            controlnet_conditioning_scale=10.0,
            width=512,
            height=512,
        ).images[0]

        # Save all results
        stylized_edges_path = os.path.join(
            output_dir, f"{base_name}_stylized_edges.jpg"
        )
        realistic_edges_path = os.path.join(
            output_dir, f"{base_name}_realistic_edges.jpg"
        )
        generated_path = os.path.join(output_dir, f"{base_name}_generated.jpg")
        comparison_path = os.path.join(output_dir, f"{base_name}_comparison.jpg")

        stylized_edges_pil.save(stylized_edges_path)
        realistic_edges_pil.save(realistic_edges_path)
        generated_image.save(generated_path)

        # Create comparison image
        self.create_comparison(
            stylized_image, realistic_edges_pil, generated_image, comparison_path
        )

        print(f"Stylized edges: {stylized_edges_path}")
        print(f"Realistic edges: {realistic_edges_path}")
        print(f"Generated image: {generated_path}")

        return {
            "stylized_edges": stylized_edges_path,
            "realistic_edges": realistic_edges_path,
            "generated": generated_path,
            "comparison": comparison_path,
        }

    def create_comparison(self, original, edges, generated, output_path):
        target_size = (512, 512)
        original_resized = original.resize(target_size)
        edges_resized = edges.resize(target_size)
        generated_resized = generated.resize(target_size)

        comparison = Image.new("RGB", (1536, 512))
        comparison.paste(original_resized, (0, 0))
        comparison.paste(edges_resized, (512, 0))
        comparison.paste(generated_resized, (1024, 0))
        comparison.save(output_path)

    def process_directory(self, input_dir, output_dir="inference_results"):
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_files = []

        for ext in image_extensions:
            image_files.extend(
                [f for f in os.listdir(input_dir) if f.lower().endswith(ext)]
            )

        if not image_files:
            return

        results = []
        for img_file in image_files:
            img_path = os.path.join(input_dir, img_file)
            result = self.process_image(img_path, output_dir)
            results.append(result)

        return results


def run_training(args):
    if not os.path.exists(args.dataset_root):
        print(f"Dataset not found: {args.dataset_root}")
        return

    trainer = Trainer(
        dataset_root=args.dataset_root,
        styles=args.styles,
        lr=args.lr,
        batch_size=args.batch_size,
        max_samples_per_class=args.max_samples_per_class,
        device=args.device,
    )

    trainer.train(
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )


def run_inference(args):
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        if os.path.exists(args.checkpoint_dir):
            checkpoints = glob.glob(
                os.path.join(args.checkpoint_dir, "pix2pix_epoch_*.pth")
            )
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                checkpoint_path = checkpoints[-1]
            else:
                print(f"No checkpoints found in {args.checkpoint_dir}")
                return
        else:
            print(f"Checkpoint directory not found: {args.checkpoint_dir}")
            return

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    inference = Pix2PixInference(checkpoint_path, device=args.device)

    if os.path.isfile(args.input):
        results = inference.process_image(args.input, args.output_dir)
    else:
        results = inference.process_directory(args.input, args.output_dir)


def main():
    args = parse_arguments()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "inference":
        run_inference(args)


if __name__ == "__main__":
    main()


# Usage:

# Training mode
# python train_pix2pix_unlearncanvas.py --mode train --epochs 200 --batch_size 256

# Inference mode (generates edges + realistic images)
# python train_pix2pix_unlearncanvas.py --mode inference \
# --input /home/dataset/unlearncanvas/Van_Gogh/Cats/1.jpg \
# --output_dir results

# Dataset structure:
# /home/dataset/unlearncanvas/
# ├── Seed_Images/
# │   ├── Cats/
# │   │   ├── 1.jpg
# │   │   ├── 2.jpg
# │   │   └── ...
# │   ├── Dogs/
# │   └── ...
# ├── Van_Gogh/
# │   ├── Cats/
# │   │   ├── 1.jpg
# │   │   ├── 2.jpg
# │   │   └── ...
# │   ├── Dogs/
# │   └── ...
# └── Monet/
#     ├── Cats/
#     └── ...
