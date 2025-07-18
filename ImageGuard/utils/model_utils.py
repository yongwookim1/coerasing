import os
import torch
import transformers
import yaml
# from transformers import deepspeed
from transformers.integrations import deepspeed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

from .arguments import ModelArguments, DataArguments, EvalArguments, LoraArguments, TrainingArguments
from .conv_utils import fair_query, safe_query
from transformers.modeling_utils import _load_state_dict_into_model
from model import get_model

def maybe_zero_3(param):
    if hasattr(param, 'ds_id'):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == 'none':
        to_return = {k: t for k, t in named_params if 'lora_' in k}
    elif bias == 'all':
        to_return = {
            k: t
            for k, t in named_params if 'lora_' in k or 'bias' in k
        }
    elif bias == 'lora_only':
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if 'lora_' in k:
                to_return[k] = t
                bias_name = k.split('lora_')[0] + 'bias'
                lora_bias_names.add(bias_name)
            elif 'bias' in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str,
                                   bias='none'):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias)
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)
    
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(trainer.model.named_parameters())
    torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))

def init_model(model_path, training_args: TrainingArguments, data_args: DataArguments, lora_args: LoraArguments, model_cfg: dict):

    model = get_model(
        model_name = model_cfg['model_name'],
        model_path = model_path,
        training_args = training_args,
        data_args = data_args,
        lora_args = lora_args,
        use_caption = model_cfg.get('use_caption', None),
    )
    if model_cfg['model_name'] == 'Idefics2':
        model, tokenizer = model.get_model_processor()
    else:
        model, tokenizer = model.get_model_tokenizer()
    
    if training_args.use_lora and lora_args.lora_weight_path != '':
        if lora_args.lora_type == 'lora':
            try:
                delta_path = os.path.join(lora_args.lora_weight_path, 'adapter_model.bin')
                delta_ckpt = torch.load(delta_path, 'cpu')
            except:
                from safetensors.torch import load_file
                delta_path = os.path.join(lora_args.lora_weight_path, 'adapter_model.safetensors')
                delta_ckpt = load_file(delta_path, 'cpu')
            new_dict = {}
            for key, value in delta_ckpt.items():
                new_dict[f'{key[:-7]}.default.weight'] = value 
            _load_state_dict_into_model(model, new_dict, start_prefix='')
            print(f'load delta ckpt from {os.path.abspath(delta_path)}')

            non_lora_ckpt_path = os.path.join(lora_args.lora_weight_path, 'non_lora_trainables.bin')
            if os.path.exists(non_lora_ckpt_path):
                non_lora_trainables = torch.load(non_lora_ckpt_path, map_location='cpu')
                _load_state_dict_into_model(model, non_lora_trainables, start_prefix='')
                print(f'load non lora ckpt from {os.path.abspath(non_lora_ckpt_path)}')
        else:
            raise NotImplementedError
   
    return model, tokenizer


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
    model_cfg,
    device):
    model, tokenizer = init_model(model_args.model_name_or_path, training_args, data_args, lora_args, model_cfg)
    model.eval()
    model.to(device).eval().half()
    model.tokenizer = tokenizer
    return model