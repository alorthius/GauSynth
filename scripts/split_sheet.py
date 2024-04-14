import os
import argparse
from PIL import Image


def split_image(input_image, output_dir, rows_num, cols_num):
    names = os.path.splitext(os.path.basename(input_image))[0]
    names = names.split("_")[1:]
    image = Image.open(input_image)

    total_width, total_height = image.size
    width = total_width // cols_num
    height = total_height // rows_num

    os.makedirs(output_dir, exist_ok=True)

    i = 0
    for row in range(rows_num):
        for col in range(cols_num):
            left = col * width
            upper = row * height
            right = (col + 1) * width
            lower = (row + 1) * height

            split_image = image.crop((left, upper, right, lower))
            split_image = split_image.resize((512, 512), Image.BICUBIC)

            output_path = os.path.join(output_dir, f"{names[i]}.png")
            split_image.save(output_path)

            i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-r", "--rows_num", type=int, default=3)
    parser.add_argument("-c", "--cols_num", type=int, default=3)

    args = parser.parse_args()

    split_image(**vars(args))
