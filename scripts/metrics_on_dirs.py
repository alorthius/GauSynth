import os

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import ConvertImageDtype, Resize

from scripts.metrics import \
    structural_similarity_index_measure, \
    peak_signal_noise_ratio, \
    learned_perceptual_image_patch_similarity, \
    clip_score


def load_and_resize_image(image_path, target_size=None):
    image = read_image(image_path, mode=ImageReadMode.RGB)
    image = ConvertImageDtype(torch.float32)(image)
    if target_size is not None:
        resize = Resize(target_size)
        image = resize(image)
    return image


def ssim_psnr_lpips_on_dirs(dir_gt, dir_gen):
    ssim_scores, lpips_scores, psnr_scores = [], [], []
    filenames = sorted(os.listdir(dir_gt))

    for filename in filenames:
        path_gt = os.path.join(dir_gt, filename)
        path_gen = os.path.join(dir_gen, filename)

        if os.path.isfile(path_gt) and os.path.isfile(path_gen):
            image_gen = load_and_resize_image(path_gen)
            target_size = image_gen.shape[-2:]
            image_gt = load_and_resize_image(path_gt, target_size)

            ssim_score = structural_similarity_index_measure(
                image_gt.unsqueeze(0),
                image_gen.unsqueeze(0)
            )
            ssim_scores.append(ssim_score.item())

            lpips_score = learned_perceptual_image_patch_similarity(
                image_gt.unsqueeze(0) * 2 - 1,
                image_gen.unsqueeze(0) * 2 - 1
            )
            lpips_scores.append(lpips_score.item())

            psnr_score = peak_signal_noise_ratio(
                image_gt.unsqueeze(0),
                image_gen.unsqueeze(0)
            )
            psnr_scores.append(psnr_score.item())
        else:
            print(f"Missing file for {filename}")

    average_ssim = sum(ssim_scores) / len(ssim_scores)
    average_psnr = sum(psnr_scores) / len(psnr_scores)
    average_lpips = sum(lpips_scores) / len(lpips_scores)

    return (round(m, 3) for m in (average_ssim, average_psnr, average_lpips))


def ssim_psnr_lpips_clip_on_dirs(dir_gt, dir_gen, prompt):
    ssim_scores, lpips_scores, psnr_scores, clip_gt_scores, clip_gen_scores = [], [], [], [], []
    filenames = sorted(os.listdir(dir_gt))

    for filename in filenames:
        path_gt = os.path.join(dir_gt, filename)
        path_gen = os.path.join(dir_gen, filename)

        if os.path.isfile(path_gt) and os.path.isfile(path_gen):
            image_gen = load_and_resize_image(path_gen)
            target_size = image_gen.shape[-2:]
            image_gt = load_and_resize_image(path_gt, target_size)

            ssim_score = structural_similarity_index_measure(
                image_gt.unsqueeze(0),
                image_gen.unsqueeze(0)
            )
            ssim_scores.append(ssim_score.item())

            lpips_score = learned_perceptual_image_patch_similarity(
                image_gt.unsqueeze(0) * 2 - 1,
                image_gen.unsqueeze(0) * 2 - 1
            )
            lpips_scores.append(lpips_score.item())

            psnr_score = peak_signal_noise_ratio(
                image_gt.unsqueeze(0),
                image_gen.unsqueeze(0)
            )
            psnr_scores.append(psnr_score.item())

            clip_gt = clip_score(
                image_gt.unsqueeze(0),
                [prompt],
            )
            clip_gt_scores.append(clip_gt)

            clip_gen = clip_score(
                image_gen.unsqueeze(0),
                [prompt],
            )
            clip_gen_scores.append(clip_gen)
        else:
            print(f"Missing file for {filename}")

    scores = ssim_scores, lpips_scores, psnr_scores, clip_gt_scores, clip_gen_scores
    return (round(sum(s) / len(s), 3) for s in scores)


if __name__ == "__main__":
    # dir_gt = 'demo_outputs_dir/6/filtered_frames_colmap'
    # dir_gen = 'demo_outputs_dir/6/filtered_frames_colmap'
    # dir_gen = 'demo_outputs_dir/6/sr_frames'
    dir_gt = 'demo_outputs_dir/6/gs/reim_10000_15-April-06:15:53-PM/test/ours_10000/gt'
    dir_gen = 'demo_outputs_dir/6/gs/reim_10000_15-April-06:15:53-PM/test/ours_10000/renders'
    average_ssim, average_lpips, average_psnr = ssim_psnr_lpips_on_dirs(dir_gt, dir_gen)
    print(average_ssim, average_lpips, average_psnr)
