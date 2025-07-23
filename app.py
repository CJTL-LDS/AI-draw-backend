import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

import io
import base64
from diffusers import StableDiffusionImg2ImgPipeline
import torch


# 注册flask应用
app = Flask(__name__)
CORS(app)

# 读取项目配置文件
with open("./configuration.json", 'r') as f:
    CONFIGS = json.load(f)

def init_pipe(config: dict) -> StableDiffusionImg2ImgPipeline:
    """
    用于初始化项目的pipe
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(config["base"]["base_model_path"], torch_dtype=torch.float32).to(CONFIGS["base"]["device"])
    pipe.load_lora_weights(config["lora"]["lora_model_path"])
    return pipe

@app.route('/generate', methods=['POST'])
def generate_image():
    """
    接收前端发来的json请求信息，并按照配置生成图片，返还给前端
    """
    try:
        # 获取请求中的图像数据
        data = request.json
        sketch_base64 = data.get('sketch')
        sketch_data = base64.b64decode(sketch_base64.split(',')[1])  # 将base64转换为PIL图像
        image = Image.open(io.BytesIO(sketch_data))

        # 生成图像
        pipe = init_pipe(CONFIGS)
        generator = torch.Generator(device = CONFIGS["base"]["device"]).manual_seed(523)
        result = pipe(
                prompt = CONFIGS["train"]["prompt"],
                negative_prompt = CONFIGS["train"]["negative_prompt"],
                strength = CONFIGS["train"]["strength"],
                guidance_scale = CONFIGS["train"]["guidance_scale"],
                image = image,
                num_inference_steps = CONFIGS["train"]["num_inference_steps"],
                num_images_per_prompt = CONFIGS["train"]["num_images_per_prompt"],
                height = CONFIGS["train"]["height"],
                width = CONFIGS["train"]["width"],
                generator=generator
        ).images[0]

        # 将生成的图像转换为base64
        buffer = io.BytesIO()
        result.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        # 返回前端
        return jsonify({
                'success': True,
                'image'  : f'data:image/png;base64,{img_str}'
        })

    except Exception as e:
        return jsonify({
                'success': False,
                'error'  : str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)