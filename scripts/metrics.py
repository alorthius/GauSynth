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
    try:
        frechet_inception_distance.loss.update(images_real, real=True)
        frechet_inception_distance.loss.update(images_fake, real=False)
        score = frechet_inception_distance.loss.compute()
    except AttributeError:
        frechet_inception_distance.loss = FrechetInceptionDistance()
        score = frechet_inception_distance(images_real, images_fake)
    return score


def kernel_inception_distance(images_real, images_fake, subsets, subset_size):
    try:
        kernel_inception_distance.loss.update(images_real, real=True)
        kernel_inception_distance.loss.update(images_fake, real=False)
        score = kernel_inception_distance.loss.compute()
    except AttributeError:
        kernel_inception_distance.loss = KernelInceptionDistance(subsets=subsets, subset_size=subset_size)
        score = kernel_inception_distance(images_real, images_fake, subsets, subset_size)
    return score


def learned_perceptual_image_patch_similarity(images_real, images_fake):
    try:
        score = learned_perceptual_image_patch_similarity.loss(images_real, images_fake)
    except AttributeError:
        learned_perceptual_image_patch_similarity.loss = LearnedPerceptualImagePatchSimilarity()
        score = learned_perceptual_image_patch_similarity(images_real, images_fake)
    return score


def peak_signal_noise_ratio(images_real, images_fake):
    try:
        score = peak_signal_noise_ratio.loss(images_fake, images_real)
    except AttributeError:
        peak_signal_noise_ratio.loss = PeakSignalNoiseRatio(data_range=1.0)
        score = peak_signal_noise_ratio(images_real, images_fake)
    return score


def structural_similarity_index_measure(images_real, images_fake):
    try:
        score = structural_similarity_index_measure.loss(images_fake, images_real)
    except AttributeError:
        structural_similarity_index_measure.loss = StructuralSimilarityIndexMeasure(data_range=1.0)
        score = structural_similarity_index_measure(images_real, images_fake)
    return score


def inception_score(images):
    try:
        inception_score.loss.update(images)
        score = inception_score.loss.compute()
    except AttributeError:
        inception_score.loss = InceptionScore()
        score = inception_score(images)
    return score


def clip_score(images, prompts):
    try:
        score = clip_score.loss(images, prompts)
        score = score.detach()
    except AttributeError:
        clip_score.loss = CLIPScore()
        score = clip_score(images, prompts)
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
