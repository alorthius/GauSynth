import requests
import json
import copy

from Fooocus.extras.preprocessors import canny_pyramid

from scripts.utils import *

# important!!!
# if image_prompts are set in vary, mixing = True always!


host = "http://127.0.0.1:8888"

advanced_params = {
    "overwrite_vary_strength": 0.7,
    # "mixing_image_prompt_and_vary_upscale": True,
    # "canny_low_threshold": 250,
    # "canny_high_threshold": 255,
    "disable_preview": True,
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
    "async_process": False,

    "advanced_params": advanced_params,
    "uov_method": "Vary (Subtle)",
    "input_image": None,
}


image_prompts = [
    {
        "cn_img": None,
        "cn_stop": 0.8,
        "cn_weight": 0,
        "cn_type": "ImagePrompt"
    }, {
        "cn_img": None,
        "cn_stop": 0.6,
        "cn_weight": 0,
        "cn_type": "PyraCanny"
    }
]


def resize_image(im, width, height, resize_mode=1):
    im = Image.fromarray(im)

    def resize(im, w, h):
        return im.resize((w, h), resample=Image.LANCZOS)

    if resize_mode == 0:
        res = resize(im, width, height)

    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
                res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                          box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
                res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                          box=(fill_width + src_w, 0))

    return np.array(res)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def canny_preview(image, low, high):
    image = pillow_to_numpy(image)
    cn_img = resize_image(HWC3(image), width=1024, height=1024)
    cn_img = canny_pyramid(cn_img, int(low), int(high))
    return cn_img


def ip_request(params: dict) -> dict:
    response = requests.post(
        url=f"{host}/v2/generation/image-upscale-vary",
        data=json.dumps(params),
        headers={"Content-Type": "application/json"}
    )
    return response.json()


def image_prompt(
        image_sheet, prompt, strength, seed, sd_checkpoint, controlnet_check,
        ip_weight, ip_stop_at,
        canny_weight, canny_stop_at, canny_low, canny_high,
):
    image = pillow_to_base64(numpy_to_pillow(image_sheet))

    p = copy.deepcopy(params)

    p["input_image"] = image

    p["prompt"] = prompt
    p["advanced_params"]["overwrite_vary_strength"] = strength
    p["image_seed"] = int(seed)
    p["base_model_name"] = "juggernautXL_v9Rundiffusion.safetensors" if sd_checkpoint == "Juggernaut" else "realisticStockPhoto_v20.safetensors"

    if not controlnet_check:
        p["advanced_params"]["mixing_image_prompt_and_vary_upscale"] = False

    else:
        p["advanced_params"]["mixing_image_prompt_and_vary_upscale"] = True
        p["image_prompts"] = copy.deepcopy(image_prompts)

        for ip in range(0, 2):
            p["image_prompts"][ip]["cn_img"] = image

        p["image_prompts"][0]["cn_weight"] = ip_weight
        p["image_prompts"][0]["cn_stop"] = ip_stop_at

        p["image_prompts"][1]["cn_weight"] = canny_weight
        p["image_prompts"][1]["cn_stop"] = canny_stop_at
        p["advanced_params"]["canny_low_threshold"] = canny_low
        p["advanced_params"]["canny_high_threshold"] = canny_high

    res = ip_request(p)[0]
    print(res["finish_reason"])
    image = pillow_to_numpy(base64_to_pillow(res["base64"]))
    return image
