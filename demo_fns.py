import os

from scripts.process_video import split_video, form_video
from scripts.filter_images import filter_images


def create_dir(dir_name):
    os.makedirs(f"demo_outputs_dir/{dir_name}")


def process_video(fps, vid_path, dir_name):
    frames_path = f"demo_outputs_dir/{dir_name}/orig_frames"
    filtered_path = f"demo_outputs_dir/{dir_name}/filtered_frames"
    output_vid_path = f"demo_outputs_dir/{dir_name}/filtered_vid"
    output_vid = f"demo_outputs_dir/{dir_name}/filtered_vid/vid.mp4"

    for d in [frames_path, filtered_path, output_vid_path]:
        os.makedirs(d)

    split_video(fps, vid_path, frames_path)
    filter_images(frames_path, filtered_path, 100)
    form_video(fps, filtered_path, output_vid)

    return output_vid
