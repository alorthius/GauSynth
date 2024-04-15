import os

import torch
from torchvision.io import read_image
from torchvision.transforms import ConvertImageDtype

from scripts.metrics import structural_similarity_index_measure, peak_signal_noise_ratio, learned_perceptual_image_patch_similarity


def load_images(image_path):
    image = read_image(image_path)
    image = ConvertImageDtype(torch.float32)(image)
    return image


def calculate_ssim_for_directories(dir_a, dir_b):
    ssim_scores, lpips_scores, psnr_scores = [], [], []
    filenames = sorted(os.listdir(dir_a))

    for filename in filenames:
        path_a = os.path.join(dir_a, filename)
        path_b = os.path.join(dir_b, filename)

        if os.path.isfile(path_a) and os.path.isfile(path_b):
            image_gt = load_images(path_a)
            image_new = load_images(path_b)

            ssim_score = structural_similarity_index_measure(
                image_gt.unsqueeze(0),
                image_new.unsqueeze(0)
            )
            ssim_scores.append(ssim_score.item())

            lpips_score = learned_perceptual_image_patch_similarity(
                image_gt.unsqueeze(0) * 2 - 1,
                image_new.unsqueeze(0) * 2 - 1
            )
            lpips_scores.append(lpips_score.item())

            psnr_score = peak_signal_noise_ratio(
                image_gt.unsqueeze(0),
                image_new.unsqueeze(0)
            )
            psnr_scores.append(psnr_score.item())
        else:
            print(f"Missing file for {filename}")

    average_ssim = sum(ssim_scores) / len(ssim_scores)
    average_lpips = sum(lpips_scores) / len(lpips_scores)
    average_psnr = sum(psnr_scores) / len(psnr_scores)

    return average_ssim, average_lpips, average_psnr


if __name__ == "__main__":
    # dir_a = 'demo_outputs_dir/6/filtered_frames_colmap'
    # dir_b = 'demo_outputs_dir/6/filtered_frames_colmap'
    # dir_b = 'demo_outputs_dir/6/sr_frames'
    dir_a = 'demo_outputs_dir/6/gs/reim_10000_15-April-06:15:53-PM/test/ours_10000/gt'
    dir_b = 'demo_outputs_dir/6/gs/reim_10000_15-April-06:15:53-PM/test/ours_10000/renders'
    average_ssim, average_lpips, average_psnr = calculate_ssim_for_directories(dir_a, dir_b)
    print(average_ssim, average_lpips, average_psnr)
