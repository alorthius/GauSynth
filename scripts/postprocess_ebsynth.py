import os
import shutil

from PIL import Image
from rembg import remove


def merge_directories(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for subdir, _, _ in os.walk(input_dir):
        try:
            offset = int(subdir.split("/")[-1])
        except ValueError:
            continue

        for f in os.listdir(subdir):
            new_f = str(offset + int(os.path.splitext(os.path.basename(f))[0])).zfill(2)
            shutil.copy(f"{subdir}/{f}", f"{output_dir}/{new_f}.png")


def remove_background(orig_dir, ebsynth_dir, output_dir_orig, output_dir_ebsynth, output_dir_blend):
    for d in [output_dir_orig, output_dir_ebsynth, output_dir_blend]:
        os.makedirs(d, exist_ok=True)
    files = sorted([f for f in os.listdir(ebsynth_dir) if f.endswith(".png")])

    for f in files:
        orig = Image.open(f"{orig_dir}/{f}")
        orig_alpha = remove(orig)

        ebsynth = Image.open(f"{ebsynth_dir}/{f}")
        ebsynth_alpha = remove(ebsynth)

        white_bg = Image.new("RGBA", orig.size, "WHITE")
        white_bg.paste(orig_alpha, mask=orig_alpha)
        white_bg.resize((2048, 2048)).save(f"{output_dir_orig}/{f}")

        white_bg.paste(ebsynth_alpha, mask=orig_alpha)
        white_bg.save(f"{output_dir_blend}/{f}")

        white_bg = Image.new("RGBA", orig.size, "WHITE")
        white_bg.paste(ebsynth_alpha, mask=orig_alpha)
        white_bg.save(f"{output_dir_ebsynth}/{f}")
