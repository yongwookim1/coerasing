import transformers
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='')

@dataclass
class DataArguments:
    given_num: bool = False
    img_size: int = 490
    hd_num: int = -1
    data_cfg: str = ''
    data_version: int = 3


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default='adamw_torch')
    max_length: int = field(
        default=4096,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = False
    # eval_flag: int = 0
    label_names: List[str] = field(default_factory=lambda: ['samples'])
    seed: int = 3407
    gradient_checkpointing: bool = True

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    ### for internlm ###
    lora_target_modules: List[str] = field(default_factory=lambda: [
        'attention.wqkv',
        'attention.wo',
        'feed_forward.w1',
        'feed_forward.w2',
        'feed_forward.w3',
    ])
    #### for idefics2 ###
    # lora_target_modules: List[str] = field(default_factory=lambda: [
    #     'self_attn.q_proj',
    #     'self_attn.k_proj',
    #     'self_attn.v_proj',
    #     'self_attn.o_proj',
    #     'mlp.gate_proj',
    #     'mlp.up_proj',
    #     'mlp.down_proj',
    # ])
    lora_weight_path: str = ''
    lora_bias: str = 'none'
    lora_type: str = 'lora'


@dataclass
class EvalArguments:
    max_length: int = field(
        default=4096,
        metadata={
            'help':
            'Maximum sequence length. Sequences will be right padded (and possibly truncated).'
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    fix_sampler: bool = True
    # eval_flag: int = 0
    label_names: List[str] = field(default_factory=lambda: ['samples'])
    gradient_checkpointing: bool = False