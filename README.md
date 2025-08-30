# Paper Name

This is the official code for the paper.

**Authors:** Yongwoo Kim, Sungmin Cha, Hyunsoo Kim, Donghyun Kim

## Installation

- Some codes and checkpoints will update as soon as possible.

Clone this repository:

```bash
git clone git@github.com:yongwookim1/coerasing.git
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

# Training

Before training, there are a few steps.

- Step1: Generate reference images (for use image modality during training)

  ```bash
  python generate_data.py --label nudity
  ```

- Step2: Train the model with LoRA weight initialization

  ```bash
  python main.py \
  --modality text \
  --train_method full \
  --prompt tench \
  --devices 0,1 \
  --ckpt_path CompVis/stable-diffusion-v1-4 \
  --forget_image_path FORGET_IMAGE_PATH \
  --retain_image_path RETAIN_IMAGE_PATH \
  --iterations 1000 \
  --save_iter 100 \
  --lora_init_method fisher \
  --lora_init_prompt tench \
  --lora_rank LORA_RANK \
  --lr LR
  ```

# Evaluation

The evaluation follows [UnlearnDiff](https://github.com/OPTML-Group/Diffusion-MU-Attack).

## Citation

If you find this work or repository useful, please cite the following:

```bib
-
```

## Contact us

If you have any detailed questions or suggestions, feel free to email us: [lifeiran@iie.ac.cn](mailto:lifeiran@iie.ac.cn)! Thanks for your interest in our work!

## Acknowledgement

This repository is built upon the great works of [ESD](https://github.com/rohitgandikota/erasing) and [Co‑Erasing](https://github.com/Ferry-Li/Co-Erasing).
