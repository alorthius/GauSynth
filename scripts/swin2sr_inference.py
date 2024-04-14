import os
from itertools import islice

import numpy as np
from PIL import Image
import torch

from transformers import Swin2SRForImageSuperResolution
from transformers import Swin2SRImageProcessor

from scripts.utils import numpy_to_pillow


gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")

model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr").to(cpu_device)
processor = Swin2SRImageProcessor()


def list_to_batches(input_list, batch):
    it = iter(input_list)
    return iter(lambda: list(islice(it, batch)), [])


def sr_inference(image, m):
    image = processor(image, return_tensors="pt").pixel_values.to(gpu_device)

    with torch.no_grad():
        output = m(image)

        output = output.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = [numpy_to_pillow((np.moveaxis(o, source=0, destination=-1) * 255.0).round().astype(np.uint8)) for o in output]
        return output


def sr_inference_dir(input_dir, output_dir, batch=2):
    m = model.to(gpu_device)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    batches = list_to_batches(files, batch)

    for b in batches:
        print(b)
        img_orig = [Image.open(os.path.join(input_dir, i)).convert("RGB") for i in b]
        img_sr = sr_inference(img_orig, m)
        for i, img in enumerate(img_sr):
            img.save(os.path.join(output_dir, b[i]))

    m = m.to(cpu_device)
    torch.cuda.empty_cache()
