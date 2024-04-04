import os
import argparse
from PIL import Image

def split_image(input_image, output_dir, n_rows, n_cols):
    image = Image.open(input_image)

    total_width, total_height = image.size
    width = total_width // n_cols
    height = total_height // n_rows

    os.makedirs(output_dir, exist_ok=True)
    for row in range(n_rows):
        for col in range(n_cols):
            left = col * width
            upper = row * height
            right = (col + 1) * width
            lower = (row + 1) * height

            split_image = image.crop((left, upper, right, lower))

            output_path = os.path.join(output_dir, f"{row}_{col}.png")
            split_image.save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_image", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("--n_rows", type=int, default=3)
    parser.add_argument("--n_cols", type=int, default=3)

    args = parser.parse_args()

    split_image(**vars(args))
