import base64
import io

from PIL import Image
import numpy as np
import cv2


def base64_to_opencv(image):
    image = base64.b64decode(image)
    image = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def opencv_to_base64(image):
    _, image = cv2.imencode(".png", image)
    image = base64.b64encode(image).decode("utf-8")
    return image


def base64_to_pillow(image):
    image = base64.b64decode(image)
    image = io.BytesIO(image)
    image = Image.open(image)
    return image


def pillow_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image = buffer.getvalue()
    image = base64.b64encode(image).decode('utf-8')
    return image


def opencv_to_pillow(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def pillow_to_opencv(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
