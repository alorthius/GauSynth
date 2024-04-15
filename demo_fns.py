import os
import shutil
import imageio
import datetime

from scripts.process_video import split_video, form_video, form_colmap_video
from scripts.filter_images import filter_images
from scripts.run_colmap import run_colmap, visualize_colmap
from scripts.merge_sheet import merge_images
from scripts.fooocus_inference import image_prompt
from scripts.split_sheet import split_image
from scripts.preprocess_ebsynth import split_directory, split_keyframes
from scripts.ebsynth_interp import interpolate
from scripts.postprocess_ebsynth import merge_directories, remove_background
from scripts.swin2sr_inference import sr_inference_dir


def create_dir(dir_name):
    os.makedirs(f"demo_outputs_dir/{dir_name}", exist_ok=True)


def process_video(fps, vid_path, dir_name):
    frames_sd_path = f"demo_outputs_dir/{dir_name}/orig_frames_sd"
    filtered_sd_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    
    frames_colmap_path = f"demo_outputs_dir/{dir_name}/orig_frames_colmap"
    filtered_colmap_path = f"demo_outputs_dir/{dir_name}/filtered_frames_colmap"
    
    output_vid_path = f"demo_outputs_dir/{dir_name}/new_videos"
    output_vid = f"demo_outputs_dir/{dir_name}/new_videos/orig.mp4"

    for d in [frames_sd_path, filtered_sd_path, frames_colmap_path, filtered_colmap_path, output_vid_path]:
        os.makedirs(d, exist_ok=True)

    # frames for sd 512x512
    split_video(fps, vid_path, frames_sd_path, 512)
    filter_images(frames_sd_path, filtered_sd_path, 100)
    form_video(fps, filtered_sd_path, output_vid)

    # frames for colmap 2048x2048
    split_video(fps, vid_path, frames_colmap_path, 2048)
    filter_images(frames_colmap_path, filtered_colmap_path, 100)

    for d in [frames_sd_path, frames_colmap_path]:
        shutil.rmtree(d)

    return output_vid


def run_sfm(dir_name):
    images_path = f"demo_outputs_dir/{dir_name}/filtered_frames_colmap"
    workdir_path = f"demo_outputs_dir/{dir_name}/colmap/distorted"
    os.makedirs(workdir_path, exist_ok=True)

    run_colmap(workdir_path, images_path)

    # show point cloud and cameras visualization
    screenshots_path = f"demo_outputs_dir/{dir_name}/screenshots_colmap"
    os.makedirs(screenshots_path, exist_ok=True)
    cameras, images, points = visualize_colmap(f"{workdir_path}/sparse/0/", screenshots_path)

    output_vid = f"demo_outputs_dir/{dir_name}/new_videos/colmap.mp4"
    form_colmap_video(30, screenshots_path, output_vid)

    shutil.rmtree(screenshots_path)

    return output_vid, [[cameras, images, points]]


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
    keyframes_path = f"demo_outputs_dir/{dir_name}/keyframes/"
    keyframes_ebsynth_path = f"demo_outputs_dir/{dir_name}/keyframes_ebsynth/"
    os.makedirs(keyframes_path, exist_ok=True)

    split_image(reimagine_file, keyframes_path, n, n)
    split_keyframes(keyframes_path, keyframes_ebsynth_path)

    orig_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    orig_ebsynth_path = f"demo_outputs_dir/{dir_name}/filtered_ebsynth"
    os.makedirs(orig_ebsynth_path, exist_ok=True)

    split_directory(orig_path, orig_ebsynth_path, n * n)

    ebsynth_splitted_path = f"demo_outputs_dir/{dir_name}/ebsynth_splitted"
    step = 100 // (n * n)
    for i in range(0, 99, step):
        print(i)
        key_path = f"{keyframes_ebsynth_path}/{str(i).zfill(2)}/00.png"
        or_path = f"{orig_ebsynth_path}/{str(i).zfill(2)}"
        out_path = f"{ebsynth_splitted_path}/{str(i).zfill(2)}"

        interpolate(key_path, or_path, out_path)

    ebsynth_all_path = f"demo_outputs_dir/{dir_name}/ebsynth_all"
    merge_directories(ebsynth_splitted_path, ebsynth_all_path)

    ebsynth_vid = f"demo_outputs_dir/{dir_name}/new_videos/ebsynth.mp4"
    form_video(30, ebsynth_all_path, ebsynth_vid)

    for d in [ebsynth_splitted_path, orig_ebsynth_path, keyframes_path, keyframes_ebsynth_path]:
        shutil.rmtree(d)

    return ebsynth_vid


def ebsynth_post_process(dir_name):
    ebsynth_transparent = f"demo_outputs_dir/{dir_name}/ebsynth_transparent"
    ebsynth_all = f"demo_outputs_dir/{dir_name}/ebsynth_all"
    orig_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    remove_background(orig_path, ebsynth_all, ebsynth_transparent)

    vid = f"demo_outputs_dir/{dir_name}/new_videos/ebsynth_transparent.mp4"
    form_video(30, ebsynth_transparent, vid)
    return vid


def run_sr(dir_name):
    lr_dir = f"demo_outputs_dir/{dir_name}/ebsynth_transparent"
    sr_dir = f"demo_outputs_dir/{dir_name}/sr_frames"
    os.makedirs(sr_dir, exist_ok=True)

    sr_inference_dir(lr_dir, sr_dir)

    vid = f"demo_outputs_dir/{dir_name}/new_videos/sr.mp4"
    form_video(30, sr_dir, vid)
    return vid
