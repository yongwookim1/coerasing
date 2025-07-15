import os
import random
from tqdm import tqdm
import torch
import yaml
import torch.nn.functional as F

from utils.model_utils import load_unet, load_others, add_lora_to_unet, merge_lora_to_unet
from utils.training_utils import get_training_params
from utils.diffusion_utils import denoise_to_text_timestep, predict_text_t_noise, set_scheduler_device
from utils.helpers import save_model, to_same_device


def train_text_mode(args):
    device_list = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    device_1 = torch.device(device_list[0])
    device_2 = torch.device(device_list[1])

    origin_unet = load_unet(args.ckpt_path, requires_grad=False).to(device_1)
    unet = load_unet(args.ckpt_path, requires_grad=True).to(device_2)
    
    origin_unet.eval()
    unet.train()

    if args.unet_ckpt_path:
        origin_unet.load_state_dict(torch.load(args.unet_ckpt_path))
        unet.load_state_dict(torch.load(args.unet_ckpt_path))
        print(f"[Loading] Loaded UNet checkpoint from {args.unet_ckpt_path}")

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
            image_path=args.image,
            retain_image_path=args.retain_image_path,
            )
        if args.lora_ckpt_path:
            lora_modules.load_state_dict(torch.load(args.lora_ckpt_path))
            unet.load_state_dict(torch.load(args.remained_unet_ckpt_path))
            print(f"[Loading] Loaded LoRA checkpoint from {args.lora_ckpt_path}")
            print(f"[Loading] Loaded remained UNet checkpoint from {args.remained_unet_ckpt_path}")
        parameters = lora_modules.parameters()
        unet.eval()
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
        lora_save_path = os.path.join(save_path, str(args.lora_rank), "fisher")
        os.makedirs(lora_save_path, exist_ok=True)

    prompt_list = [p.strip() for p in args.prompt.split(',')]

    for idx in tqdm(range(args.iterations)):
        optimizer.zero_grad()

        prompt = random.choice(prompt_list)
        input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True).input_ids
        text_embed = text_encoder(input_ids.to(origin_unet.device))[0].to(unet.device)

        uncond_ids = tokenizer("", return_tensors="pt", padding="max_length", truncation=True).input_ids
        uncond_embed = text_encoder(uncond_ids.to(origin_unet.device))[0].to(unet.device)

        t = torch.randint(num_inference_steps, (1,)).to(unet.device)
        t_ddpm = torch.randint(int(t * 1000 / num_inference_steps),
                               int((t + 1) * 1000 / num_inference_steps), (1,))

        start_code = torch.randn((1, 4, 64, 64)).to(unet.device)

        with torch.no_grad():
            set_scheduler_device(noise_scheduler, unet.device)
            z = denoise_to_text_timestep(unet, text_embed, t, start_code, noise_scheduler)
            uncond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, uncond_embed)
            cond_origin_noise = predict_text_t_noise(z, t_ddpm, origin_unet, text_embed)

        cond_noise = predict_text_t_noise(z, t_ddpm, unet, text_embed)

        [cond_origin_noise, uncond_origin_noise, cond_noise] = to_same_device(
            [cond_origin_noise, uncond_origin_noise, cond_noise], unet.device)

        loss = criterion(cond_noise, uncond_origin_noise - args.negative_guidance * (cond_origin_noise - uncond_origin_noise))
        loss.backward()
        optimizer.step()

        if (idx + 1) % args.save_iter == 0:
            if args.lora_init_method is not None:
                save_model(lora_modules, lora_save_path, idx+1, model_name='lora')
                save_model(unet, lora_save_path, idx+1, model_name='remained_unet')
                merged_unet = merge_lora_to_unet(unet, lora_modules)
                save_model(merged_unet, lora_save_path, idx+1, model_name='merged_unet')
                print(f"[Checkpoint] Saved model at iteration {idx + 1}")
            else:
                save_model(unet, unet_save_path, idx+1, model_name='unet')
                print(f"[Checkpoint] Saved model at iteration {idx + 1}")
