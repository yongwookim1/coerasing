from datasets import load_dataset

dataset = load_dataset('YijunYang280/MMA-Diffusion-NSFW-adv-prompts-benchmark', split='train')

mma_nudity_prompts = [item['target_prompt'] for item in dataset]