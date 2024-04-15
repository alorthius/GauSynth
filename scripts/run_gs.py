import bisect
import os

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
