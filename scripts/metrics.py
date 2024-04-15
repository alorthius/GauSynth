import cv2

import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore


def frechet_inception_distance(images_real, images_fake):
    fid = FrechetInceptionDistance()
    fid.update(images_real, real=True)
    fid.update(images_fake, real=False)
    score = fid.compute()
    return score


def kernel_inception_distance(images_real, images_fake, subsets, subset_size):
    kid = KernelInceptionDistance(subsets=subsets, subset_size=subset_size)
    kid.update(images_real, real=True)
    kid.update(images_fake, real=False)
    score = kid.compute()
    return score


def learned_perceptual_image_patch_similarity(images_real, images_fake):
    lpips = LearnedPerceptualImagePatchSimilarity()
    score = lpips(images_real, images_fake)
    return score


def peak_signal_noise_ratio(images_real, images_fake):
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    score = psnr(images_fake, images_real)
    return score


def structural_similarity_index_measure(images_real, images_fake):
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    score = ssim(images_fake, images_real)
    return score


def inception_score(images):
    inception = InceptionScore()
    inception.update(images)
    score = inception.compute()
    return score


def clip_score(images, prompts):
    clip = CLIPScore()
    score = clip(images, prompts)
    score = score.detach()
    return score


if __name__ == "__main__":
    images_gt = torch.randint(0, 200, (10, 3, 299, 299), dtype=torch.uint8)
    images_gen = torch.randint(100, 255, (10, 3, 299, 299), dtype=torch.uint8)

    print(frechet_inception_distance(images_gt, images_gen))
    print(kernel_inception_distance(images_gt, images_gen, 3, 5))
    print(learned_perceptual_image_patch_similarity(images_gt.float() / 255.0 * 2 - 1, images_gen.float() / 255.0 * 2 - 1))
    print(peak_signal_noise_ratio(images_gt, images_gen))
    print(structural_similarity_index_measure(images_gt.float() / 255.0, images_gen.float() / 255.0))

    print(inception_score(images_gen))
    print(clip_score(images_gen, ["cat"] * 10))
