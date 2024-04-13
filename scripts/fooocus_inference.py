import requests
import json

from scripts.utils import *


host = "http://127.0.0.1:8888"

# image = open("../test_input/silk2k.jpg", "rb").read()
# liquid = open("../test_input/liquid.png", "rb").read()

###### important!!!
# if image_prompts are set in vary, mixing = True always!

advanced_params = {
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

    "prompt": "",

    "require_base64": True,
    "image_prompts": [
        {
            "cn_img": None,
            "cn_stop": 0.8,
            "cn_weight": 1,
            "cn_type": "ImagePrompt"
        }, {
            "cn_img": None,
            "cn_stop": 0.6,
            "cn_weight": 1,
            "cn_type": "PyraCanny"
        }],
    "async_process": False,

    "advanced_params": advanced_params,
    "uov_method": "Vary (Subtle)",
    "input_image": None,
    }


def ip_request(params: dict) -> dict:
    response = requests.post(url=f"{host}/v2/generation/image-upscale-vary",
                             data=json.dumps(params),
                             headers={"Content-Type": "application/json"})
    return response.json()


def image_prompt(image_sheet, prompt):
    image = opencv_to_base64(image_sheet)

    params["input_image"] = image
    for ip in range(0, 2):
        params["image_prompts"][ip]["cn_img"] = image

    params["prompt"] = prompt


    res = ip_request(params)[0]
    print(res["finish_reason"])

    image = base64_to_opencv(res["base64"])
    return image

    # print(json.dumps(result, indent=4, ensure_ascii=False))
