import os
import imageio
import datetime

from scripts.process_video import split_video, form_video
from scripts.filter_images import filter_images
from scripts.merge_sheet import merge_images
from scripts.fooocus_inference import image_prompt
from scripts.split_sheet import split_image
from scripts.preprocess_ebsynth import split_directory, split_keyframes
from scripts.ebsynth_interp import interpolate


def create_dir(dir_name):
    os.makedirs(f"demo_outputs_dir/{dir_name}", exist_ok=True)


def process_video(fps, vid_path, dir_name):
    frames_sd_path = f"demo_outputs_dir/{dir_name}/orig_frames_sd"
    filtered_sd_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    
    frames_colmap_path = f"demo_outputs_dir/{dir_name}/orig_frames_colmap"
    filtered_colmap_path = f"demo_outputs_dir/{dir_name}/filtered_frames_colmap"
    
    output_vid_path = f"demo_outputs_dir/{dir_name}/filtered_vid"
    output_vid = f"demo_outputs_dir/{dir_name}/filtered_vid/vid.mp4"

    for d in [frames_sd_path, filtered_sd_path, frames_colmap_path, filtered_colmap_path, output_vid_path]:
        os.makedirs(d, exist_ok=True)

    # frames for sd 512x512
    split_video(fps, vid_path, frames_sd_path, 512)
    filter_images(frames_sd_path, filtered_sd_path, 100)
    form_video(fps, filtered_sd_path, output_vid)

    # frames for colmap 2048x2048
    split_video(fps, vid_path, frames_colmap_path, 2048)
    filter_images(frames_colmap_path, filtered_colmap_path, 100)
    
    return output_vid


def create_sheet(n, dir_name):
    images_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    sheet_path = f"demo_outputs_dir/{dir_name}/orig_sheets"
    sheet_file, name = merge_images(images_path, sheet_path, n, n)
    return sheet_file, name


def reimagine(image_sheet, dir_name, image_file, prompt):
    reimagine_dir = f"demo_outputs_dir/{dir_name}/reimagine_sheets"
    time_stamp = datetime.datetime.now().strftime("%d-%B-%I:%M:%S-%p")
    reimagine_file = f"{reimagine_dir}/{time_stamp}_{image_file}"
    os.makedirs(reimagine_dir, exist_ok=True)

    image = image_prompt(image_sheet, prompt)
    imageio.imwrite(reimagine_file, image)

    return image, reimagine_file


def interpolate_frames(reimagine_file, dir_name, n):
    reimagine_path = f"demo_outputs_dir/{dir_name}/reimagine_sheets/{reimagine_file}"
    keyframes_path = f"demo_outputs_dir/{dir_name}/keyframes/"
    keyframes_ebsynth_path = f"demo_outputs_dir/{dir_name}/keyframes_ebsynth/"
    os.makedirs(keyframes_path, exist_ok=True)

    split_image(reimagine_path, keyframes_path, n, n)
    split_keyframes(keyframes_path, keyframes_ebsynth_path)

    orig_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    orig_ebsynth_path = f"demo_outputs_dir/{dir_name}/filtered_ebsynth"
    os.makedirs(orig_ebsynth_path, exist_ok=True)

    split_directory(orig_path, orig_ebsynth_path, n * n)

    ebsynth_splitted_path = f"demo_outputs_dir/{dir_name}/ebsynth_splitted"
    step = 100 // (n * n)
    for i in range(0, 100, step):
        print(i)
        key_path = f"{keyframes_ebsynth_path}/{str(i).zfill(2)}/00.png"
        or_path = f"{orig_ebsynth_path}/{str(i).zfill(2)}"
        out_path = f"{ebsynth_splitted_path}/{str(i).zfill(2)}"

        interpolate(key_path, or_path, out_path)

    ebsynth_all_path = f"demo_outputs_dir/{dir_name}/ebsynth_all"
