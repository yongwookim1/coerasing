---
license: apache-2.0
tag:
- vision
- image-classification
- image-to-text
- image-captioning
base_model:
- internlm/internlm-xcomposer2-vl-7b
pipeline_tag: image-to-text
---


<p align="center">
    <img src="logo_en.png" width="400"/>
<p>

<p align="center">
    <b><font size="6">ImageGuard</font></b> 
<p>

<div align="center">

[💻Github Repo](https://github.com/adwardlee/t2i_safety)

[Paper](https://arxiv.org/abs/)

</div>

**ImageGuard** is a vision-language model (VLM) based on [InternLM-XComposer2](https://github.com/InternLM/InternLM-XComposer) for advanced image safety evaluation. 

### Import from Transformers
ImageGuard works with transformers>=4.42.

## Quickstart
We provide a simple example to show how to use InternLM-XComposer with 🤗 Transformers.
```python
import os
import json
import torch
import time
import numpy as np
import argparse
import yaml

from PIL import Image
from utils.img_utils import ImageProcessor
from utils.arguments import ModelArguments, DataArguments, EvalArguments, LoraArguments
from utils.model_utils import init_model
from utils.conv_utils import fair_query, safe_query

def load_yaml(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result

def textprocess(safe=True):
    if safe:
        conversation = safe_query('Internlm')
    else:
        conversation = fair_query('Internlm')
    return conversation

def model_init(
    model_args: ModelArguments, 
    data_args: DataArguments, 
    training_args: EvalArguments,
    lora_args: LoraArguments,
    model_cfg):
    model, tokenizer = init_model(model_args.model_name_or_path, training_args, data_args, lora_args, model_cfg)
    model.eval()
    model.cuda().eval().half()
    model.tokenizer = tokenizer
    return model
    
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_dir', required=False, type=str, default='lora/')
    parser.add_argument('--base_model', type=str, default='internlm/internlm-xcomposer2-vl-7b') 
    args = parser.parse_args()
    load_dir = args.load_dir
    config = load_yaml(os.path.join(load_dir, 'config.yaml'))
    model_cfg = config['model_cfg']
    data_cfg = config['data_cfg']['data_cfg']
    model_cfg['model_name'] = 'Internlm'
    data_cfg['train']['model_name'] = 'Internlm'
    lora_cfg = config['lora_cfg']
    training_cfg = config['training_cfg']
    
    model_args = ModelArguments()
    model_args.model_name_or_path = args.base_model
    Lora_args = LoraArguments()
    Lora_args.lora_alpha = lora_cfg['lora_alpha']
    Lora_args.lora_bias = lora_cfg['lora_bias']
    Lora_args.lora_dropout = lora_cfg['lora_dropout']
    Lora_args.lora_r = lora_cfg['lora_r']
    Lora_args.lora_target_modules = lora_cfg['lora_target_modules']
    Lora_args.lora_weight_path = load_dir  ### comment for base model testing ### llj ## change ##
    train_args = EvalArguments()
    train_args.max_length = training_cfg['max_length']
    train_args.fix_vit = training_cfg['fix_vit']
    train_args.fix_sampler = training_cfg['fix_sampler']
    train_args.use_lora = training_cfg['use_lora']
    train_args.gradient_checkpointing = training_cfg['gradient_checkpointing']
    data_args = DataArguments()

    model = model_init(model_args, data_args, train_args, Lora_args, model_cfg)
    print(' model device: ', model.device, flush=True)

    img = Image.open('punch.png')
    safe = True ## True for toxicity and privacy, False for fairness
    prompt = textprocess(safe=safe)
    vis_processor = ImageProcessor(image_size=490)
    image = vis_processor(img)[None, :, :, :]
    with torch.cuda.amp.autocast():
        response, _ = model.chat(model.tokenizer, prompt, image, history=[], do_sample=False, meta_instruction=None)
    print(response)
    # unsafe\n violence
```

### Open Source License
The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow free commercial usage.