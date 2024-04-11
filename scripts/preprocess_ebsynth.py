import os
import shutil


def split_directory(input_dir, output_dir, k):
    files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])
    print(files)

    images_per_dir = len(files) // k

    for i in range(k):
        new_dir = f'{output_dir}/{i * images_per_dir}'
        os.makedirs(new_dir, exist_ok=True)

        start_index = i * images_per_dir
        end_index = min(start_index + images_per_dir + 1, len(files))

        for j, file in enumerate(files[start_index:end_index]):
            new_filename = f'{str(j).zfill(2)}.png'
            shutil.copy(os.path.join(input_dir, file), os.path.join(new_dir, new_filename))

    shutil.copy(os.path.join(input_dir, files[-1]), os.path.join(new_dir, f'{str(images_per_dir).zfill(2)}.png'))


split_directory('test_input/ebsynth/input/', 'test_input/ebsynth/pre_ebs/', 4)

