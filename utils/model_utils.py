import torch
import math
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.unets import UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from diffusers.schedulers import DDPMScheduler
from ip_adapter.utils import is_torch2_available
import numpy as np
from tqdm.auto import tqdm
import copy

if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        tokens = self.proj(image_embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        return self.norm(tokens)


class IPAdapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds, schedule=None):
        if image_embeds.shape[-1] != 768:
            ip_tokens = self.image_proj_model(image_embeds)
        else:
            ip_tokens = image_embeds

        if schedule is not None:
            schedule = schedule.to(self.unet.device)
            encoder_hidden_states *= schedule
            ip_tokens *= (1 - schedule)

        hidden = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        return self.unet(noisy_latents, timesteps, hidden).sample

    def load_from_checkpoint(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"])
        print(f"[IPAdapter] Loaded checkpoint: {ckpt_path}")


def get_attn_processor(unet):
# init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    return attn_procs


def load_unet(ckpt_path, requires_grad=False):
    unet = UNet2DConditionModel.from_pretrained(ckpt_path, subfolder="unet")
    unet.requires_grad_(requires_grad)
    return unet


def load_others(ckpt_path, requires_grad=False, image_encoder_path=None):
    vae = AutoencoderKL.from_pretrained(ckpt_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(ckpt_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(ckpt_path, subfolder="text_encoder")
    scheduler = DDPMScheduler.from_pretrained(ckpt_path, subfolder="scheduler")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, subfolder="image_encoder") if image_encoder_path else None

    # Freeze all
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if image_encoder:
        image_encoder.requires_grad_(False)

    return vae, tokenizer, text_encoder, scheduler, image_encoder


class LoRAModule(torch.nn.Module):
    def __init__(self, lora_down, lora_up, alpha, rank):
        super().__init__()
        self.lora_down = lora_down
        self.lora_up = lora_up
        self.scale = alpha / rank
        self.rank = rank
        
    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scale


def _get_hook_factory(lora_module):
    def hook(module, input_tensor, output_tensor):
        return output_tensor + lora_module(input_tensor[0])
    return hook


def compute_fisher_information(unet, prompt, module_names, tokenizer, text_encoder, device, iterations=10, nsteps=50):
    origin_text_encoder_device = text_encoder.device
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    fisher_info = {name: torch.zeros_like(dict(unet.named_modules())[name].weight) for name in module_names}
    unet.eval()
    criteria = torch.nn.MSELoss()
    prompt = [p.strip() for p in prompt.split(',')]
    for p in prompt:
        for _ in tqdm(range(iterations), desc='[Fisher]'):
            text_tokens = tokenizer([p], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_tokens.input_ids.to(device))[0]
            unconditional_tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            unconditional_embeddings = text_encoder(unconditional_tokens.input_ids.to(device))[0]
            text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])
            latents = torch.randn(2, unet.in_channels, 64, 64, device=device)
            t = torch.randint(0, 1000, (1,), device=device).long()
            latents = latents.to(device)
            t = t.to(device)
            latents.requires_grad = True
            noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings).sample
            loss = criteria(noise_pred, torch.zeros_like(noise_pred))
            loss.backward()
            for name in module_names:
                module = dict(unet.named_modules())[name]
                if hasattr(module, 'weight') and module.weight.grad is not None:
                    fisher_info[name] += module.weight.grad.data.pow(2)
    for name in fisher_info:
        fisher_info[name] /= (iterations * len(prompt))
    text_encoder.to(origin_text_encoder_device)
    return fisher_info


def add_lora_to_unet(unet, lora_rank=16, lora_alpha=1.0, train_method='xattn', lora_init_prompt=None, lora_init_method=None, tokenizer=None, text_encoder=None, device=None):
    lora_modules = torch.nn.ModuleDict()
    module_names = []
    for name, module in unet.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        if train_method == 'xattn':
            if 'attn2' not in name:
                continue
        elif train_method == 'xattn-strict':
            if 'attn2' not in name or ('to_q' not in name and 'to_k' not in name):
                continue
        elif train_method == 'noxattn':
            if 'attn2' in name:
                continue 
        elif train_method == 'selfattn':
            if 'attn1' not in name:
                continue
        elif train_method != 'full':
            raise NotImplementedError(f"train_method: {train_method} is not implemented for LoRA.")
        module_names.append(name)
    fisher_info_dict = None
    if lora_init_method == 'fisher' and lora_init_prompt is not None:
        fisher_info_dict = compute_fisher_information(unet, lora_init_prompt, module_names, tokenizer, text_encoder, device)
    
    lora_scale = lora_alpha / lora_rank
    
    for name, module in unet.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            continue
        if train_method == 'xattn':
            if 'attn2' not in name:
                continue
        elif train_method == 'xattn-strict':
            if 'attn2' not in name or ('to_q' not in name and 'to_k' not in name):
                continue
        elif train_method == 'noxattn':
            if 'attn2' in name:
                continue 
        elif train_method == 'selfattn':
            if 'attn1' not in name:
                continue
        elif train_method != 'full':
            raise NotImplementedError(f"train_method: {train_method} is not implemented for LoRA.")
        device = module.weight.device
        dtype = module.weight.dtype
        fisher_info = None
        if fisher_info_dict is not None:
            fisher_info = fisher_info_dict.get(name, None)
        if isinstance(module, torch.nn.Linear):
            lora_down = torch.nn.Linear(module.in_features, lora_rank, bias=False).to(device=device, dtype=dtype)
            lora_up = torch.nn.Linear(lora_rank, module.out_features, bias=False).to(device=device, dtype=dtype)
            if lora_init_method == 'fisher' and fisher_info is not None:
                W = module.weight.data
                F = fisher_info.detach()
                row_importance = F.sum(dim=1).sqrt().to(device=device)
                U, S, V = torch.svd_lowrank(row_importance[:,None] * W, q=lora_rank)
                
                lora_A = (V * torch.sqrt(S)).t()
                lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S))
                W_star = W - (lora_B @ lora_A) * lora_scale
                
                lora_down.weight.data.copy_(lora_A)
                lora_up.weight.data.copy_(lora_B)
                module.weight.data.copy_(W_star)
            else:
                torch.nn.init.normal_(lora_down.weight, std=1.0/lora_rank)
                torch.nn.init.zeros_(lora_up.weight)
        elif isinstance(module, torch.nn.Conv2d):
            if lora_init_method == 'fisher' and fisher_info is not None:
                W = module.weight.data
                F = fisher_info.detach()
                out_c, in_c, kh, kw = W.shape
                W2d = W.reshape(out_c, -1)
                F2d = F.reshape(out_c, -1)
                row_importance = F2d.sum(dim=1).sqrt().to(device=device)
                
                lora_rank = min(lora_rank, in_c * kh * kw, out_c)
                U, S, V = torch.svd_lowrank(row_importance[:,None] * W2d, q=lora_rank)
                
                lora_A = (V * torch.sqrt(S)).t()
                lora_B = (1/(row_importance+1e-5))[:,None] * (U * torch.sqrt(S))
                W_star = W - (lora_B @ lora_A).reshape(out_c, in_c, kh, kw) * lora_scale
                
                lora_down = torch.nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=lora_rank,
                    kernel_size=module.kernel_size,
                    padding=module.padding,
                    stride=module.stride,
                    bias=False,
                    ).to(device=device, dtype=dtype)
                lora_up = torch.nn.Conv2d(
                    in_channels=lora_rank,
                    out_channels=module.out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                    ).to(device=device, dtype=dtype)
                lora_down.weight.data.copy_(lora_A.reshape(lora_rank, in_c, kh, kw))
                lora_up.weight.data.copy_(lora_B.reshape(out_c, lora_rank, 1, 1))
                module.weight.data.copy_(W_star)
            else:
                lora_down = torch.nn.Conv2d(
                    in_channels=module.in_channels,
                    out_channels=lora_rank,
                    kernel_size=module.kernel_size,
                    padding=module.padding,
                    stride=module.stride,
                    bias=False,
                    ).to(device=device, dtype=dtype)
                lora_up = torch.nn.Conv2d(
                    in_channels=lora_rank,
                    out_channels=module.out_channels,
                    kernel_size=1,
                    padding=0,
                    bias=False,
                    ).to(device=device, dtype=dtype)
                torch.nn.init.normal_(lora_down.weight, std=1.0/lora_rank)
                torch.nn.init.zeros_(lora_up.weight)
        else:
            continue
        lora_module = LoRAModule(lora_down, lora_up, lora_alpha, lora_rank)
        lora_modules[name.replace('.', '_')] = lora_module
        module.register_forward_hook(_get_hook_factory(lora_module))
    
    return lora_modules


def merge_lora_to_unet(unet, lora_modules):
    merged_unet = copy.deepcopy(unet)
    module_dict = {name.replace('.', '_'): module for name, module in merged_unet.named_modules()}

    merged_count = 0
    for lora_name, lora_module in lora_modules.items():
        if lora_name not in module_dict:
            continue
        module = module_dict[lora_name]
        if not (hasattr(module, 'weight') and module.weight is not None):
            continue

        # Linear
        if isinstance(lora_module.lora_down, torch.nn.Linear):
            lora_weight = lora_module.lora_up.weight @ lora_module.lora_down.weight
            lora_weight = lora_weight * lora_module.scale
            module.weight.data += lora_weight
            merged_count += 1
        # Conv2d
        elif isinstance(lora_module.lora_down, torch.nn.Conv2d):
            down_weight = lora_module.lora_down.weight
            up_weight = lora_module.lora_up.weight
            rank = down_weight.shape[0]
            out_c = up_weight.shape[0]
            in_c = down_weight.shape[1]
            kh = down_weight.shape[2]
            kw = down_weight.shape[3]
            down_2d = down_weight.view(rank, -1)
            up_2d = up_weight.view(out_c, rank)
            lora_weight_2d = up_2d @ down_2d
            lora_weight = lora_weight_2d.view(out_c, in_c, kh, kw) * lora_module.scale
            module.weight.data += lora_weight
            merged_count += 1

    hook_count = 0
    for name, module in merged_unet.named_modules():
        if hasattr(module, '_forward_hooks') and module._forward_hooks:
            hook_count += len(module._forward_hooks)
            module._forward_hooks.clear()

    return merged_unet
