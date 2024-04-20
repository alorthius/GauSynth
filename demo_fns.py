import os
import shutil
import imageio
import datetime
from distutils.dir_util import copy_tree

from scripts.process_video import split_video, form_video, form_colmap_video, calc_new_fps
from scripts.filter_images import filter_images
from scripts.run_colmap import run_colmap, visualize_colmap
from scripts.merge_sheet import merge_images
from scripts.fooocus_inference import image_prompt, canny_preview
from scripts.split_sheet import split_image
from scripts.preprocess_ebsynth import split_directory, split_keyframes
from scripts.ebsynth_interp import interpolate
from scripts.postprocess_ebsynth import merge_directories, remove_background
from scripts.swin2sr_inference import sr_inference_dir
from scripts.metrics_on_dirs import ssim_psnr_lpips_on_dirs
from scripts.run_gs import gs_pipeline


DEL_UNUSED_DIRS = False


def create_dir(dir_name):
    os.makedirs(f"demo_outputs_dir/{dir_name}", exist_ok=True)


def process_video(fps, vid_path, dir_name, frames):
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
    filter_images(frames_sd_path, filtered_sd_path, frames)

    # frames for colmap 2048x2048
    split_video(fps, vid_path, frames_colmap_path, 2048)
    filter_images(frames_colmap_path, filtered_colmap_path, frames)

    old_frames = len([f for f in os.listdir(frames_sd_path) if f.endswith((".png", ".jpg", ".jpeg", ".PNG"))])
    new_fps = calc_new_fps(int(fps), old_frames, frames)

    form_video(new_fps, filtered_sd_path, output_vid)

    if DEL_UNUSED_DIRS:
        for d in [frames_sd_path, frames_colmap_path]:
            shutil.rmtree(d)

    return output_vid, new_fps


def create_sheet(n, dir_name):
    images_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"
    sheet_path = f"demo_outputs_dir/{dir_name}/orig_sheets"
    sheet_file, name = merge_images(images_path, sheet_path, n, n)
    return sheet_file, name


def preview_canny(image_sheet, canny_low, canny_high):
    return canny_preview(image_sheet, canny_low, canny_high)


def reimagine(
        image_sheet, dir_name, image_file,
        prompt, strength, seed, sd_checkpoint, controlnet_check,
        ip_weight, ip_stop_at,
        canny_weight, canny_stop_at, canny_low, canny_high
):
    reimagine_dir = f"demo_outputs_dir/{dir_name}/reimagine_sheets"
    time_stamp = datetime.datetime.now().strftime("%d-%B-%I:%M:%S-%p")
    reimagine_file = f"{reimagine_dir}/{time_stamp}_{image_file}"
    os.makedirs(reimagine_dir, exist_ok=True)

    image = image_prompt(
        image_sheet, prompt, strength, seed, sd_checkpoint, controlnet_check,
        ip_weight, ip_stop_at,
        canny_weight, canny_stop_at, canny_low, canny_high,
    )
    imageio.imwrite(reimagine_file, image)

    return image, reimagine_file


def interpolate_frames(reimagine_file, dir_name, n, frames, new_fps):
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
    step = frames // (n * n)
    for i in range(0, frames - 1, step):
        print(i)
        key_path = f"{keyframes_ebsynth_path}/{str(i).zfill(2)}/00.png"
        or_path = f"{orig_ebsynth_path}/{str(i).zfill(2)}"
        out_path = f"{ebsynth_splitted_path}/{str(i).zfill(2)}"

        interpolate(key_path, or_path, out_path)

    ebsynth_all_path = f"demo_outputs_dir/{dir_name}/ebsynth_all"
    merge_directories(ebsynth_splitted_path, ebsynth_all_path)

    ebsynth_vid = f"demo_outputs_dir/{dir_name}/new_videos/ebsynth.mp4"
    form_video(new_fps, ebsynth_all_path, ebsynth_vid)

    if DEL_UNUSED_DIRS:
        for d in [ebsynth_splitted_path, orig_ebsynth_path, keyframes_path, keyframes_ebsynth_path]:
            shutil.rmtree(d)

    return ebsynth_vid


def ebsynth_post_process(dir_name, new_fps):
    ebsynth_all = f"demo_outputs_dir/{dir_name}/ebsynth_all"
    orig_path = f"demo_outputs_dir/{dir_name}/filtered_frames_sd"

    orig_alpha = f"demo_outputs_dir/{dir_name}/orig_transparent"
    ebsynth_alpha = f"demo_outputs_dir/{dir_name}/ebsynth_transparent"
    blend_alpha = f"demo_outputs_dir/{dir_name}/blend_transparent"

    remove_background(orig_path, ebsynth_all, orig_alpha, ebsynth_alpha, blend_alpha)

    vid = f"demo_outputs_dir/{dir_name}/new_videos/blend_transparent.mp4"
    form_video(new_fps, blend_alpha, vid)
    return vid


def run_sr(dir_name, new_fps):
    lr_dir = f"demo_outputs_dir/{dir_name}/blend_transparent"
    sr_dir = f"demo_outputs_dir/{dir_name}/sr_frames"
    os.makedirs(sr_dir, exist_ok=True)

    sr_inference_dir(lr_dir, sr_dir)

    vid = f"demo_outputs_dir/{dir_name}/new_videos/sr.mp4"
    form_video(new_fps, sr_dir, vid)
    return vid


def calc_metrics(dir_name):
    gt = f"demo_outputs_dir/{dir_name}/filtered_frames_colmap"
    gt_alpha = f"demo_outputs_dir/{dir_name}/orig_transparent"

    interpolation = f"demo_outputs_dir/{dir_name}/ebsynth_all"
    interpolation_alpha = f"demo_outputs_dir/{dir_name}/ebsynth_transparent"
    post_processing = f"demo_outputs_dir/{dir_name}/blend_transparent"
    super_res = f"demo_outputs_dir/{dir_name}/sr_frames"

    interpolation_metrics = ssim_psnr_lpips_on_dirs(gt, interpolation)
    interpolation_alpha_metrics = ssim_psnr_lpips_on_dirs(gt_alpha, interpolation_alpha)
    post_processing_metrics = ssim_psnr_lpips_on_dirs(gt_alpha, post_processing)
    super_res_metrics = ssim_psnr_lpips_on_dirs(gt_alpha, super_res)

    table = [
        ["Interp", *interpolation_metrics],
        ["Interp Alpha", *interpolation_alpha_metrics],
        ["Post-proc", *post_processing_metrics],
        ["SR", *super_res_metrics],
    ]
    return table


def run_sfm(dir_name, new_fps):
    images_path = f"demo_outputs_dir/{dir_name}/filtered_frames_colmap"
    workdir_path = f"demo_outputs_dir/{dir_name}/colmap/distorted"
    os.makedirs(workdir_path, exist_ok=True)

    run_colmap(workdir_path, images_path)

    # show point cloud and cameras visualization
    screenshots_path = f"demo_outputs_dir/{dir_name}/screenshots_colmap"
    os.makedirs(screenshots_path, exist_ok=True)
    cameras, images, points = visualize_colmap(f"{workdir_path}/sparse/0/", screenshots_path)

    output_vid = f"demo_outputs_dir/{dir_name}/new_videos/colmap.mp4"
    form_colmap_video(new_fps, screenshots_path, output_vid)

    if DEL_UNUSED_DIRS:
        shutil.rmtree(screenshots_path)

    return output_vid, [[cameras, images, points]]


def gs_reconstruct(dir_name, iters, new_fps):
    c_dir = f"demo_outputs_dir/{dir_name}/colmap"
    dirs_prev = ["images", "input", "sparse", "stereo"]
    for d in dirs_prev:
        try:
            shutil.rmtree(f"{c_dir}/{d}")
        except FileNotFoundError:
            pass

    new_imgs_dir = f"demo_outputs_dir/{dir_name}/colmap/input/"
    os.makedirs(new_imgs_dir)
    copy_tree(f"demo_outputs_dir/{dir_name}/sr_frames/", new_imgs_dir)

    gs_dir = f"demo_outputs_dir/{dir_name}/gs"
    os.makedirs(gs_dir, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%d-%B-%I:%M:%S-%p")
    output_dir = f"{gs_dir}/reim_{iters}_{time_stamp}"

    metrics = gs_pipeline(dir_name, output_dir, iters)
    metrics = [[str(iter.split("_")[-1]), *list(map(lambda x: round(x, 3), mdict.values()))] for iter, mdict in metrics.items()]

    renders_dir = f"{output_dir}/train/ours_{iters}/renders"
    vid = f"demo_outputs_dir/{dir_name}/new_videos/gs_reim_{iters}_{time_stamp}.mp4"
    form_video(new_fps, renders_dir, vid)
    return vid, metrics


def gs_reconstruct_orig(dir_name, iters, new_fps):
    c_dir = f"demo_outputs_dir/{dir_name}/colmap"
    dirs_prev = ["images", "input", "sparse", "stereo"]
    for d in dirs_prev:
        try:
            shutil.rmtree(f"{c_dir}/{d}")
        except FileNotFoundError:
            pass

    new_imgs_dir = f"demo_outputs_dir/{dir_name}/colmap/input/"
    os.makedirs(new_imgs_dir)
    copy_tree(f"demo_outputs_dir/{dir_name}/orig_transparent/", new_imgs_dir)

    gs_dir = f"demo_outputs_dir/{dir_name}/gs"
    os.makedirs(gs_dir, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%d-%B-%I:%M:%S-%p")
    output_dir = f"{gs_dir}/orig_{iters}_{time_stamp}"

    metrics = gs_pipeline(dir_name, output_dir, iters)
    metrics = [[str(iter.split("_")[-1]), *list(map(lambda x: round(x, 3), mdict.values()))] for iter, mdict in metrics.items()]

    renders_dir = f"{output_dir}/train/ours_{iters}/renders"
    vid = f"demo_outputs_dir/{dir_name}/new_videos/gs_orig_{iters}_{time_stamp}.mp4"
    form_video(new_fps, renders_dir, vid)
    return vid, metrics
