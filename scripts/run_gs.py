import os
import bisect
import shutil
from distutils.dir_util import copy_tree
import datetime

import json

test_iterations = [7000, 10000, 12000, 15000]


def filter_test_iters(it):
    pos = bisect.bisect_right(test_iterations, it)
    cropped_iters = test_iterations[:pos]
    if pos == 0 or cropped_iters[-1] != it:
        cropped_iters.append(it)
    return cropped_iters


def convert_colmap(colmap_dir):
    os.system(f"python convert.py -s {colmap_dir} --skip_matching")


def train_gs(colmap_dir, images_dir, output_dir, test_iters):
    iters_str = " ".join(map(str, test_iters))
    iters_last = max(test_iters)
    os.system(f"python train.py -s {colmap_dir} -i {images_dir} --iterations {iters_last} --test_iterations {iters_str} --save_iterations {iters_str} -m {output_dir} -w --eval")


def render_gs(output_dir, test_iters):
    for iter in test_iters:
        os.system(f"python render.py -m {output_dir} -w --iteration {iter}")


def metric_test(output_dir):
    os.system(f"python metrics.py -m {output_dir}")


def gs_pipeline(dir_name, output_dir, iters):
    colmap_dir = f"../demo_outputs_dir/{dir_name}/colmap/"
    images_dir = f"images"
    output_dir = f"../{output_dir}"

    os.chdir("gaussian-splatting")

    test_iters = filter_test_iters(iters)

    convert_colmap(colmap_dir)
    train_gs(colmap_dir, images_dir, output_dir, test_iters)
    render_gs(output_dir, test_iters)
    metric_test(output_dir)

    metrics_file = f"{output_dir}/results.json"
    with open(metrics_file) as f:
        metrics = json.load(f)
        print(metrics)

    os.chdir("..")
    return metrics


def reconstruction(dir_name, iters, mode):
    c_dir = f"demo_outputs_dir/{dir_name}/colmap"
    dirs_prev = ["images", "input", "sparse", "stereo"]
    for d in dirs_prev:
        try:
            shutil.rmtree(f"{c_dir}/{d}")
        except FileNotFoundError:
            pass

    new_imgs_dir = f"demo_outputs_dir/{dir_name}/colmap/input/"
    os.makedirs(new_imgs_dir)

    input_folder = "sr_frames" if mode == "reimagine" else "orig_transparent"
    input_dir = f"demo_outputs_dir/{dir_name}/{input_folder}/"
    copy_tree(input_dir, new_imgs_dir)

    gs_dir = f"demo_outputs_dir/{dir_name}/gs"
    os.makedirs(gs_dir, exist_ok=True)
    time_stamp = datetime.datetime.now().strftime("%d-%B-%I:%M:%S-%p")
    prefix = mode[:4]
    output_dir = f"{gs_dir}/{prefix}_{iters}_{time_stamp}"

    metrics = gs_pipeline(dir_name, output_dir, iters)
    metrics = [[int(iter.split("_")[-1]), *list(map(lambda x: round(x, 3), mdict.values()))] for iter, mdict in metrics.items()]
    metrics.sort(key=lambda x: x[0])

    vid = f"demo_outputs_dir/{dir_name}/new_videos/gs_{prefix}_{iters}_{time_stamp}.mp4"
    return output_dir, metrics, vid


def merge_train_test_renderings(train_dir, test_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img_train = sorted(os.listdir(train_dir))
    img_test = sorted(os.listdir(test_dir))

    train_i = 0
    test_i = 0
    output_i = 0

    while output_i < len(img_train) + len(img_test) - 1:
        # add one image from test
        if test_i < len(img_test):
            src_path = os.path.join(test_dir, img_test[test_i])
            dst_path = os.path.join(output_dir, f"{str(output_i).zfill(2)}.png")
            shutil.copy(src_path, dst_path)
            test_i += 1
            output_i += 1

        # add 7 images from train
        for _ in range(7):
            if train_i < len(img_train):
                src_path = os.path.join(train_dir, img_train[train_i])
                dst_path = os.path.join(output_dir, f"{str(output_i).zfill(2)}.png")
                shutil.copy(src_path, dst_path)
                train_i += 1
                output_i += 1
