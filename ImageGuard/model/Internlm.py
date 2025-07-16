import torch
from model.internlm_xcomposer.configuration_internlm_xcomposer2 import InternLMXcomposer2Config
from model.internlm_xcomposer.modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM
from model.internlm_xcomposer.tokenization_internlm_xcomposer2 import InternLMXComposer2Tokenizer
from peft import LoraConfig, get_peft_model

from .base import BaseModel

class InternLM(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_model_tokenizer(self):
        config = InternLMXcomposer2Config.from_pretrained(self.model_path)
        config.use_cache = False
        config.max_length = self.training_args.max_length
        
        model = InternLMXComposer2ForCausalLM.from_pretrained(
            self.model_path,
            config=config,
            device_map=None,
            use_caption=self.use_caption
        )

        if self.data_args.img_size != 336:
            model.vit.resize_pos()

        tokenizer_path = self.model_path if self.lora_args.lora_weight_path == '' \
            else self.lora_args.lora_weight_path
        tokenizer = InternLMXComposer2Tokenizer.from_pretrained(
            tokenizer_path,
            padding_side='right',
            use_fast=False,
        )

        self.model = model
        self.tokenizer = tokenizer

    def configure_training_args(self):
        training_args = self.training_args
        if training_args.fix_vit:
            self.model.vit.requires_grad_(False)
        else:
            self.model.vit.requires_grad_(True)
            self.model.vit.vision_tower.vision_model.post_layernorm = torch.nn.Identity()

        if training_args.fix_sampler or self.use_caption:
            self.model.vision_proj.requires_grad_(False)
        else:
            self.model.vision_proj.requires_grad_(True)

    def configure_peft(self):
        if not self.training_args.use_lora:
            for name, param in self.model.model.named_parameters():
                if 'vision_cross' not in name:
                    param.requires_grad = False
            return
        
        lora_args = self.lora_args
        if lora_args.lora_type == 'lora':
            for name, param in self.model.model.named_parameters():
                if 'vision_cross' in name:
                    continue
                param.requires_grad = False
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type='CAUSAL_LM',
            )

            self.model = get_peft_model(self.model, lora_config)
        elif lora_args.lora_type == 'plora':
            for name, param in self.model.model.named_parameters():
                if 'Plora' not in name:
                    param.requires_grad = False

        if self.use_caption:
            if lora_args.lora_type == 'lora':
                self.model.model.vision_proj.requires_grad_(True)
                self.model.model.model.tok_embeddings.requires_grad_(True)
                self.model.model.logit_scale.requires_grad_(True)
            else:
                self.model.vision_proj.requires_grad_(True)
                self.model.model.tok_embeddings.requires_grad_(True)
                self.model.logit_scale.requires_grad_(True)

        #### 
        for name, param in self.model.model.named_parameters():
            if 'vision_cross' in name:
                param.requires_grad = True

        if self.training_args.gradient_checkpointing:
            self.model.enable_input_require_grads()
            # self.model.gradient_checkpointing_enable()
            self.model.model.vit.vision_tower.gradient_checkpointing_enable({"use_reentrant": True})


        

