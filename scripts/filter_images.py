import os
import shutil
import argparse


def filter_images(input_dir, output_dir, keep_num):
    files = sorted([f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"))])
    step = len(files) // keep_num
    files_to_keep = files[::step]

    diff = len(files_to_keep) - keep_num
    files_to_keep = files_to_keep[diff : diff + keep_num]

    os.makedirs(output_dir, exist_ok=True)
    for i, f in enumerate(files_to_keep):
        shutil.copy(os.path.join(input_dir, f), os.path.join(output_dir, f"{str(i).zfill(2)}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-n", "--keep_num", default=100)
    args = parser.parse_args()

    filter_images(**vars(args))
