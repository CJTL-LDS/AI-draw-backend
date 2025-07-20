import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载图生图pipeline
model_path = "./assets/ink.safetensors"
pipe = StableDiffusionImg2ImgPipeline.from_single_file(model_path, torch_dtype=torch.float32).to(device)

# 读取初始图片
init_image = Image.open("1.png").convert("RGB")

# 推理
prompt = "Identical with the origin picture"
generator = torch.Generator(device=device).manual_seed(2023)

image = pipe(
    prompt=prompt,                 # 描述你希望生成图像的文本提示
    negative_prompt=" ",              # 指定你不希望图像中出现的内容 用于优化生成结果
    strength=0.1,                  # 控制初始图像保留的程度 数值越小 输出图像越接近原始图像
    guidance_scale=7.5,            # 控制生成图像与 prompt 匹配程度的强度。数值越高，生成结果越贴近提示词，但可能导致图像失真或过度饱和
    image=init_image,              # 输入的初始图像，用于作为生成图像的基础
    num_inference_steps=7.5,          # 控制扩散模型推理步数，步数越多，图像质量越高，但耗时也更长
    num_images_per_prompt=2,       # 生成图片的数量
    height=512,                    # 生成图片的高度
    width=512,                     # 生成图片的宽度
    generator=generator            # 生成图片的generator
).images[0]


