import os
import argparse
from PIL import Image


def merge_images(input_dir, output_dir, rows_num, cols_num):
    files = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"))])

    keep_num = rows_num * cols_num
    step = len(files) // keep_num
    selected_files = files[::step][:keep_num]

    images = [Image.open(os.path.join(input_dir, f)) for f in selected_files]

    width, height = images[0].size
    merged_width, merged_height = width * cols_num, height * rows_num
    merged_image = Image.new("RGB", (merged_width, merged_height))

    for i, image in enumerate(images):
        row = i // cols_num
        col = i % cols_num
        merged_image.paste(image, (col * width, row * height))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"sheet_{keep_num}.png")
    merged_image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--rows_num", type=int, default=3)
    parser.add_argument("--cols_num", type=int, default=3)

    args = parser.parse_args()

    merge_images(**vars(args))
