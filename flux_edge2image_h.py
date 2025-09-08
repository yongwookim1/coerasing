import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
import os
exp_id = "0902_stylized"
pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16).to("cuda:4")
prompt = "A photo of cat"
# prompt = " "
# prompt = "photorealistic, natural colors, high quality, detailed, realistic lighting"
if "edge" in exp_id:
    control_image = load_image("/home/hyunsoo/coerasing/example/fix2fix_edge.jpg")
elif "original" in exp_id:
    control_image = load_image("/home/hyunsoo/coerasing/example/img_original.png")
else:
    # control_image = load_image("/home/hyunsoo/coerasing/example/img_stylized.png")
    control_image = load_image("/home/kyw1654/coerasing/edges/cat_diffedge.png")
# processor = CannyDetector()
# low_lst = [10,20] #,30,40,50,60,70,80,90,100]
# high_lst = [100,150,200]
# for low in low_lst:
#     for high in high_lst:
#         control_image = processor(control_image, low_threshold=low, high_threshold=high, detect_resolution=1024, image_resolution=1024)
#         # control_image = processor(control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024)
#         control_image.save(f"results/edge/stylized_{low}_{high}.png")

for i in range(50, 55):
    save_path = f"{exp_id}"
    image = pipe(
        prompt=prompt,
        control_image=control_image,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]
    if not os.path.exists(f"results/{save_path}"):
        os.makedirs(f"results/{save_path}")
    image.save(f"results/{save_path}/output_gs_{i}.png")