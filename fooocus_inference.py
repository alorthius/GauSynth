import base64
import requests
import json

import cv2

from utils import *


host = "http://127.0.0.1:8888"


def text2img(params: dict) -> dict:
    """
    text to image
    """
    result = requests.post(url=f"{host}/v1/generation/text-to-image",
                           data=json.dumps(params),
                           headers={"Content-Type": "application/json"})
    return result.json()


def image_prompt(params: dict) -> dict:
    """
    image prompt
    """
    response = requests.post(url=f"{host}/v2/generation/image-prompt",
                             data=json.dumps(params),
                             headers={"Content-Type": "application/json"})
    return response.json()


# result = text2img({
#     "prompt": "1girl sitting on the ground",
#     "async_process": False})
#
# print(result)


image = open("silk.jpg", "rb").read()

params = {
    "prompt": "Blue silk texture",
    "aspect_ratios_selection": "1024*1024",
    "require_base64": True,
    "image_prompts": [
        {
            "cn_img": base64.b64encode(image).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        },{
            "cn_img": base64.b64encode(image).decode('utf-8'),
            "cn_stop": 0.6,
            "cn_weight": 0.6,
            "cn_type": "ImagePrompt"
        }],
    "async_process": False,
    }

image = image_prompt(params)[0]["base64"]
image = base64_to_opencv(image)
cv2.imwrite("out.png", image)


# print(json.dumps(result, indent=4, ensure_ascii=False))
