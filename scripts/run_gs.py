import os


def convert_colmap(colmap_dir):
    os.system(f"python convert.py -s {colmap_dir} --skip_matching")


def train_gs(colmap_dir, images_dir, output_dir, iters):
    os.system(f"python train.py -s {colmap_dir} -i {images_dir} --iterations {iters} -m {output_dir} -r 1 -w")


def render_gs(output_dir):
    os.system(f"python render.py -m {output_dir} -r 1 -w")


def gs_pipeline(dir_name, output_dir, iters):
    colmap_dir = f"../demo_outputs_dir/{dir_name}/colmap/"
    images_dir = f"images"
    output_dir = f"../{output_dir}"

    os.chdir("gaussian-splatting")

    convert_colmap(colmap_dir)
    train_gs(colmap_dir, images_dir, output_dir, iters)
    render_gs(output_dir)

    os.chdir("..")
