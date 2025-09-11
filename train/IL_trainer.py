import os
import random
import yaml
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from utils.training_utils import get_training_params
from utils.model_utils import (
    load_unet, load_others,
    ImageProjModel, IPAdapter, get_attn_processor,
    add_lora_to_unet,
    merge_lora_to_unet
)
from utils.diffusion_utils import (
    set_scheduler_device, denoise_to_text_timestep,
    predict_image_t_noise, predict_text_t_noise
)
from utils.helpers import to_same_device, save_model


transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def train_IL_mode(args):
    device_list = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    device_1 = torch.device(device_list[0])
    device_2 = torch.device(device_list[1])

    origin_unet = load_unet(args.ckpt_path, requires_grad=False).to(device_1)
    unet = load_unet(args.ckpt_path, requires_grad=True).to(device_2)

    if args.unet_ckpt_path:
        checkpoint = torch.load(args.unet_ckpt_path, map_location='cpu')
        unet.load_state_dict(checkpoint, strict=False)
        print(f"[Loading] Loaded UNet checkpoint from {args.unet_ckpt_path}")

    unet.train()
    origin_unet.eval()

    vae, tokenizer, text_encoder, noise_scheduler, _ = load_others(args.ckpt_path, requires_grad=False)
    text_encoder = text_encoder.to(origin_unet.device)

    if args.lora_init_method is not None:
        lora_modules = add_lora_to_unet(
            unet,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=noise_scheduler,
            device=unet.device,
            train_method=args.train_method,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_init_method=args.lora_init_method,
            lora_init_prompt=args.lora_init_prompt,
            forget_image_path=args.forget_image_path,
            retain_image_path=args.retain_image_path,
            fisher_loss=args.fisher_loss,
            )
        if args.lora_ckpt_path:
            lora_modules.load_state_dict(torch.load(args.lora_ckpt_path))
            unet.load_state_dict(torch.load(args.remained_unet_ckpt_path))
            print(f"[Loading] Loaded LoRA checkpoint from {args.lora_ckpt_path}")
            print(f"[Loading] Loaded remained UNet checkpoint from {args.remained_unet_ckpt_path}")
        parameters = lora_modules.parameters()
    else:
        parameters = get_training_params(unet, args.train_method)
    
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    criterion = torch.nn.MSELoss()
    num_inference_steps = args.num_inference_steps

    noise_scheduler.set_timesteps(num_inference_steps)

    save_path = args.save_path or os.path.join("checkpoints", args.modality, args.prompt, args.train_method, str(args.lr))
    if args.lora_init_method == None:
        unet_save_path = os.path.join(save_path, "unet")
        os.makedirs(unet_save_path, exist_ok=True)
    elif args.lora_init_method == 'default':
        lora_save_path = os.path.join(save_path, str(args.lora_rank), "default")
        os.makedirs(lora_save_path, exist_ok=True)
    elif args.lora_init_method == 'fisher':
        lora_save_path = os.path.join(save_path, str(args.lora_rank), "fisher", args.forget_image_path.split('/')[-1])
        os.makedirs(lora_save_path, exist_ok=True)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path).to(origin_unet.device)
    image_encoder.requires_grad_(False)

    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    origin_image_proj_model = ImageProjModel(
        cross_attention_dim=origin_unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    # Attention processors
    attn_procs = get_attn_processor(unet)
    unet.set_attn_processor(attn_procs)

    origin_attn_procs = get_attn_processor(origin_unet)
    origin_unet.set_attn_processor(origin_attn_procs)

    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    origin_adapter_modules = torch.nn.ModuleList(origin_unet.attn_processors.values())
    ip_adapter_path = args.ip_adapter

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, ip_adapter_path).to(unet.device)
    origin_ip_adapter = IPAdapter(origin_unet, origin_image_proj_model, origin_adapter_modules, ip_adapter_path).to(origin_unet.device)

    # Prepare text embeddings
    forget_text_input_ids = tokenizer(args.prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
    forget_text_embeddings = text_encoder(forget_text_input_ids.to(origin_unet.device))[0].to(unet.device)
    
    retain_prompt = ""
    retain_text_input_ids = tokenizer(retain_prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
    retain_text_embeddings = text_encoder(retain_text_input_ids.to(origin_unet.device))[0].to(unet.device)

    uncond_input_ids = tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids
    uncond_text_embeddings = text_encoder(uncond_input_ids.to(origin_unet.device))[0].to(unet.device)

    forget_image_list = _load_image_list(args.forget_image_path, args.image_number)
    retain_image_list = _load_image_list(args.retain_image_path, args.image_number)
    print(f"[INFO] Found {len(forget_image_list)} images to erase from: {args.forget_image_path}")
    print(f"[INFO] Found {len(retain_image_list)} images to erase from: {args.retain_image_path}")

    # Set to True for text-based training, False for image-based training
    text_condition = False  # Change this to True if you want text-based training
    # Set to True to use attention mechanism, False to use direct embeddings
    use_attention = False  # Change this to False if you want to skip attention
    # Training loop
    for idx in tqdm(range(args.iterations)):
        optimizer.zero_grad()
        
        # Use text condition
        if text_condition:
            prompt = args.prompt
            input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
            text_embed = text_encoder(input_ids.to(origin_unet.device))[0].to(unet.device)
        
        else:
            # Sample paired forget and retain image embeddings
            forget_image_embeds, retain_image_embeds = _sample_paired_image_embeddings(
                forget_image_list, retain_image_list, image_encoder)

            if use_attention:
                forget_key = forget_text_embeddings
                forget_query = origin_ip_adapter.image_proj_model(forget_image_embeds)
                forget_value = forget_key
                
                attention_scores = torch.matmul(forget_query.to(origin_unet.device), forget_key.to(origin_unet.device).transpose(1, 2))
                # Scale the attention scores
                d_k = forget_key.size(-1)  # embedding_dim
                scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
                
                # Apply softmax to get the attention weights
                attention_weights = F.softmax(scaled_attention_scores, dim=-1)  # Shape: [batch_size, 1, num_patches]
                
                # Compute the weighted sum of the image embeddings (weighted by attention)
                attended_image_embedding = torch.matmul(attention_weights, forget_value.to(origin_unet.device))  # Shape: [batch_size, 1, embedding_dim]
                forget_image_embeds = attended_image_embedding
                forget_image_embeds = forget_image_embeds.to(unet.device)
            else:
                # Use direct image embeddings without attention
                forget_image_embeds = origin_ip_adapter.image_proj_model(forget_image_embeds).to(unet.device)

            # Add noise to image embedding if specified
            if args.noise_factor > 0:
                noise = torch.rand_like(forget_image_embeds)
                forget_image_embeds = forget_image_embeds + args.noise_factor * noise

        if use_attention:
            retain_key = retain_text_embeddings
            retain_query = origin_ip_adapter.image_proj_model(retain_image_embeds)
            retain_value = retain_key
            
            attention_scores = torch.matmul(retain_query.to(origin_unet.device), retain_key.to(origin_unet.device).transpose(1, 2))
            # Scale the attention scores
            d_k = retain_key.size(-1)  # embedding_dim
            scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
            
            # Apply softmax to get the attention weights
            attention_weights = F.softmax(scaled_attention_scores, dim=-1)  # Shape: [batch_size, 1, num_patches]
            
            # Compute the weighted sum of the image embeddings (weighted by attention)
            attended_image_embedding = torch.matmul(attention_weights, retain_value.to(origin_unet.device))  # Shape: [batch_size, 1, embedding_dim]
            retain_image_embeds = attended_image_embedding
            retain_image_embeds = retain_image_embeds.to(unet.device)
        else:
            # Use direct image embeddings without attention
            retain_image_embeds = origin_ip_adapter.image_proj_model(retain_image_embeds).to(unet.device)

        # Add noise to image embedding if specified
        if args.noise_factor > 0:
            noise = torch.rand_like(retain_image_embeds)
            retain_image_embeds = retain_image_embeds + args.noise_factor * noise

        # Sample timestep
        t = torch.randint(num_inference_steps, (1,)).to(unet.device)
        t_ddpm = torch.randint(int(t * 1000 / num_inference_steps), int((t + 1) * 1000 / num_inference_steps), (1,))

        # Start from noise
        start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

        with torch.no_grad():
            set_scheduler_device(noise_scheduler, unet.device)
            if text_condition:
                z = denoise_to_text_timestep(unet, text_embed, t, start_code, noise_scheduler)
            else:
                z = denoise_to_text_timestep(unet, forget_image_embeds, t, start_code, noise_scheduler)

            if args.blur_factor > 0:
                z = torchvision.transforms.functional.gaussian_blur(z, kernel_size=args.blur_factor)
            
            if text_condition:
                cond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, text_embed)
            else:
                cond_origin_noise = predict_image_t_noise(z, t_ddpm, origin_unet, forget_image_embeds, origin_ip_adapter, forget_image_embeds)
            uncond_origin_noise = predict_image_t_noise(z, t_ddpm, origin_unet, retain_image_embeds, origin_ip_adapter, retain_image_embeds)
            null_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, uncond_text_embeddings)

        if text_condition:
            cond_noise = predict_text_t_noise(z, t_ddpm, unet, text_embed)
        else:
            cond_noise = predict_image_t_noise(z, t_ddpm, unet, forget_text_embeddings, ip_adapter, forget_image_embeds)

        # Compute loss
        cond_noise, uncond_origin_noise, cond_origin_noise, null_origin_noise = to_same_device(
            [cond_noise, uncond_origin_noise, cond_origin_noise, null_origin_noise], unet.device)

        target_noise = uncond_origin_noise - args.negative_guidance * (cond_origin_noise - uncond_origin_noise) - args.negative_guidance * (null_origin_noise - uncond_origin_noise)
        loss = criterion(cond_noise, target_noise)

        loss.backward()
        optimizer.step()

        if (idx + 1) % args.save_iter == 0:
            if args.lora_init_method is not None:
                # save_model(lora_modules, lora_save_path, idx+1, model_name='lora')
                # save_model(unet, lora_save_path, idx+1, model_name='remained_unet')
                merged_unet = merge_lora_to_unet(unet, lora_modules)
                save_model(merged_unet, lora_save_path, idx+1, model_name='merged_unet')
                print(f"[Checkpoint] Saved model at iteration {idx + 1}")
            else:
                save_model(unet, unet_save_path, idx+1, model_name='unet')
                print(f"[Checkpoint] Saved model at iteration {idx + 1}")


# ========================
#  Support functions
# ========================

def _load_image_list(image_path, image_number):
    if os.path.isdir(image_path):
        # Sort files to ensure consistent ordering across forget and retain lists
        all_images = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith(('png', 'jpg'))]
        all_images.sort()  # Ensure consistent alphabetical ordering
        return all_images[:image_number]  # Take first N images instead of random sampling
    else:
        return [image_path]


def _sample_image_embedding(image_list, image_encoder):
    image_path = random.choice(image_list)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(image_encoder.device)
    return image_encoder(image_tensor).image_embeds


def _sample_paired_image_embeddings(forget_image_list, retain_image_list, image_encoder):
    # Use the minimum length to ensure valid pairing
    max_pairs = min(len(forget_image_list), len(retain_image_list))
    if max_pairs == 0:
        raise ValueError("Either forget_image_list or retain_image_list is empty")
    
    # Sample the same index for both lists to maintain pairing
    pair_idx = random.randint(0, max_pairs - 1)
    
    # Load forget image
    forget_image_path = forget_image_list[pair_idx]
    forget_image = Image.open(forget_image_path).convert("RGB")
    forget_tensor = transform(forget_image).unsqueeze(0).to(image_encoder.device)
    forget_embeds = image_encoder(forget_tensor).image_embeds
    
    # Load retain image (paired)
    retain_image_path = retain_image_list[pair_idx]
    retain_image = Image.open(retain_image_path).convert("RGB")
    retain_tensor = transform(retain_image).unsqueeze(0).to(image_encoder.device)
    retain_embeds = image_encoder(retain_tensor).image_embeds
    
    return forget_embeds, retain_embeds


def _collect_contrastive_embeddings(args, image_encoder):
    concept_embeds = []
    pair_dirs = [os.path.join(args.contrastive_image, d) for d in os.listdir(args.contrastive_image)
                 if os.path.isdir(os.path.join(args.contrastive_image, d))]

    for pair_dir in tqdm(pair_dirs, desc="[Contrastive]"):
        pos = transform(Image.open(os.path.join(pair_dir, "positive.png")).convert("RGB")).unsqueeze(0).to(image_encoder.device)
        neg = transform(Image.open(os.path.join(pair_dir, "negative.png")).convert("RGB")).unsqueeze(0).to(image_encoder.device)

        pos_embed = image_encoder(pos).image_embeds
        neg_embed = image_encoder(neg).image_embeds

        if args.diff_method == "diff":
            embed = pos_embed - neg_embed
        elif args.diff_method == "l1":
            embed = torch.abs(pos_embed - neg_embed)
        elif args.diff_method == "l2":
            embed = (pos_embed - neg_embed) ** 2
        else:
            raise ValueError(f"Invalid diff_method: {args.diff_method}")

        concept_embeds.append(embed)

    return concept_embeds


def _sample_contrastive_embedding(args, concept_embeds, avg_embed):
    if args.use_average:
        return avg_embed
    else:
        return random.choice(concept_embeds)
