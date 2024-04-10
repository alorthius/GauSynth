import base64
import requests
import json

import cv2

from utils import *


host = "http://127.0.0.1:8888"


# def text2img(params: dict) -> dict:
#     """
#     text to image
#     """
#     result = requests.post(url=f"{host}/v1/generation/text-to-image",
#                            data=json.dumps(params),
#                            headers={"Content-Type": "application/json"})
#     return result.json()


def image_prompt(params: dict) -> dict:
    """
    image prompt
    """
    # response = requests.post(url=f"{host}/v2/generation/image-prompt",
    response = requests.post(url=f"{host}/v2/generation/image-upscale-vary",
                             data=json.dumps(params),
                             headers={"Content-Type": "application/json"})
    return response.json()


# result = text2img({
#     "prompt": "1girl sitting on the ground",
#     "async_process": False})
#
# print(result)


image = open("test_input/silk2k.jpg", "rb").read()
liquid = open("test_input/liquid.png", "rb").read()

advance_params = {
    "overwrite_vary_strength": 0.7,
    "mixing_image_prompt_and_vary_upscale": False,
    # "canny_low_threshold": 250,
    # "canny_high_threshold": 255,
}

params = {
    "performance_selection": "Quality",
    "aspect_ratios_selection": "1024*1024",
    "image_seed": 666,
    "sharpness": 2.0,  # 0-30
    "guidance_scale": 4.0,  # 1-30
    "base_model_name": "juggernautXL_v9Rundiffusion.safetensors",
    "style_selections": ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],

    "prompt": "Blue silk texture",

    "require_base64": True,
    "image_prompts": [
        {
            "cn_img": base64.b64encode(liquid).decode('utf-8'),
            "cn_stop": 0.8,
            "cn_weight": 1,
            "cn_type": "ImagePrompt"
        }, {
            "cn_img": base64.b64encode(liquid).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 1,
            "cn_type": "PyraCanny"
        }],
    "async_process": False,


    "advanced_params": advance_params,
    "uov_method": "Vary (Subtle)",
    "input_image": base64.b64encode(image).decode('utf-8'),
    }

###### important!!!
# if image_prompts are set in vary, mixing = True always!

res = image_prompt(params)[0]
print(res["finish_reason"])

image = base64_to_opencv(res["base64"])
cv2.imwrite("test_output/out.png", image)


# print(json.dumps(result, indent=4, ensure_ascii=False))
