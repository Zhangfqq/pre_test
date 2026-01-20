import torch
from diffusers.utils import load_image
from diffusers import FluxControlNetPipeline, FluxControlNetModel

base_model = "/home/gongke/model/FLUX.1-Kontext-dev"
controlnet_model = "/home/gongke/model/FLUX.1-dev-ControlNet-Depth"

controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")

control_image = load_image("/home/gongke/workspace/flux_depth/Gemini_Generated_Image_918zwi918zwi918z.png")
prompt = "modern style, a fully furnished bedroom, double bed in center, side tables, large wardrobe, cozy atmosphere, 4k, photorealistic, cinematic lighting."

image = pipe(prompt,
             control_image=control_image,
             controlnet_conditioning_scale=0.7,
             width=control_image.size[0],
             height=control_image.size[1],
             num_inference_steps=20,
             guidance_scale=3.5,
).images[0]
image.save("image.png")