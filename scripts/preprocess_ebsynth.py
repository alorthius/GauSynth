import os
import shutil


def split_directory(input_dir, output_dir, k):
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])

    images_per_dir = len(files) // k

    last_j = k
    for i in range(k):
        new_dir = f"{output_dir}/{str(i * images_per_dir).zfill(2)}"
        os.makedirs(new_dir, exist_ok=True)

        start_index = i * images_per_dir
        end_index = min(start_index + images_per_dir + 1, len(files))

        for j, file in enumerate(files[start_index:end_index]):
            new_filename = f"{str(j).zfill(2)}.png"
            shutil.copy(os.path.join(input_dir, file), os.path.join(new_dir, new_filename))
            last_j = j

    # copy the very last frame twice
    shutil.copy(
        os.path.join(input_dir, files[-1]),
        os.path.join(f"{output_dir}/{str((k - 1) * images_per_dir).zfill(2)}", f"{str(last_j + 1).zfill(2)}.png")
    )
    
    
def split_keyframes(input_dir, output_dir):
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    for i, kf_file in enumerate(files):
        subdir = os.path.splitext(os.path.basename(kf_file))[0]
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        shutil.copy(os.path.join(input_dir, kf_file), os.path.join(output_dir, subdir, "00.png"))
