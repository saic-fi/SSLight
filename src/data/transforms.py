import random, math
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
from torchvision import transforms
import torchvision.transforms.functional as F


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class MultiCropsDataAugmentation(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage = cfg.STAGE
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=cfg.MULTI_VIEWS_TRANSFORMS.RANDOM_HORIZONTAL_FLIP_PROB[0]),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[0], 
                        cfg.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[1],
                        cfg.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[2],
                        cfg.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[3]
                    )
                ],
                p=cfg.MULTI_VIEWS_TRANSFORMS.COLOR_JITTER_PROB[0]
            ),
            transforms.RandomGrayscale(p=cfg.MULTI_VIEWS_TRANSFORMS.GREYSCALE_PROB[0]),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        ])

        self.global_crop = transforms.RandomResizedCrop(cfg.MULTI_VIEWS_TRANSFORMS.GLOBAL_CROP_SIZE, 
            scale=(cfg.MULTI_VIEWS_TRANSFORMS.GLOBAL_CROPS_SCALE[0], cfg.MULTI_VIEWS_TRANSFORMS.GLOBAL_CROPS_SCALE[1]), 
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.local_crop = transforms.RandomResizedCrop(cfg.MULTI_VIEWS_TRANSFORMS.LOCAL_CROP_SIZE, 
            scale=(cfg.MULTI_VIEWS_TRANSFORMS.LOCAL_CROPS_SCALE[0], cfg.MULTI_VIEWS_TRANSFORMS.LOCAL_CROPS_SCALE[1]), 
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        self.global_trans1 = transforms.Compose([flip_and_color_jitter, GaussianBlur(p=cfg.MULTI_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB[0]), normalize])
        self.global_trans2 = transforms.Compose([flip_and_color_jitter, GaussianBlur(p=cfg.MULTI_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB[1]), Solarization(p=cfg.MULTI_VIEWS_TRANSFORMS.SOLARIZATION_PROB[1]), normalize])
        self.local_trans = transforms.Compose([flip_and_color_jitter, GaussianBlur(p=cfg.MULTI_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB[2]), normalize])
        self.local_crops_number = cfg.MULTI_VIEWS_TRANSFORMS.LOCAL_CROPS_NUMBER 
        
        self.val_trans1 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
            ]
        )
        # this transformation would not be used for the final evaluation but just an extra datapoint in our experiment
        self.val_trans2 = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
            ]
        )

    def __call__(self, image):
        crops = []
        if self.stage.lower() in ('train', 'ft'):
            g1 = self.global_crop(image)
            g2 = self.global_crop(image)
            crops.append(self.global_trans1(g1))
            crops.append(self.global_trans2(g2))
            for i in range(self.local_crops_number):
                ll = self.local_crop(image)
                crops.append(self.local_trans(ll))
        else:
            crops.append(self.val_trans1(image))
            crops.append(self.val_trans2(image))
        return crops


class TwoCropsDataAugmentation(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.stage = cfg.STAGE

        self.global_crop = transforms.RandomResizedCrop(cfg.TWO_VIEWS_TRANSFORMS.GLOBAL_CROP_SIZE, 
            scale=(cfg.TWO_VIEWS_TRANSFORMS.GLOBAL_CROPS_SCALE[0], cfg.TWO_VIEWS_TRANSFORMS.GLOBAL_CROPS_SCALE[1]), 
            interpolation=transforms.InterpolationMode.BICUBIC
        )

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=cfg.TWO_VIEWS_TRANSFORMS.RANDOM_HORIZONTAL_FLIP_PROB[0]),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[0], 
                        cfg.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[1],
                        cfg.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[2],
                        cfg.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_INTENSITY[3]
                    )
                ],
                p=cfg.TWO_VIEWS_TRANSFORMS.COLOR_JITTER_PROB[0]
            ),
            transforms.RandomGrayscale(p=cfg.TWO_VIEWS_TRANSFORMS.GREYSCALE_PROB[0]),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
        ])

        self.val_trans1 = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
            ]
        )
        # this transformation would not be used for the final evaluation but just an extra datapoint in our experiment
        self.val_trans2 = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD)
            ]
        )

        self.global_trans1 = transforms.Compose([flip_and_color_jitter, GaussianBlur(p=cfg.TWO_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB[0]), Solarization(p=cfg.TWO_VIEWS_TRANSFORMS.SOLARIZATION_PROB[0]), normalize])
        self.global_trans2 = transforms.Compose([flip_and_color_jitter, GaussianBlur(p=cfg.TWO_VIEWS_TRANSFORMS.GAUSSIAN_BLUR_PROB[1]), Solarization(p=cfg.TWO_VIEWS_TRANSFORMS.SOLARIZATION_PROB[1]), normalize])
            
    def __call__(self, image):
        crops = []
        if self.stage.lower() in ('train', 'ft'):
            g1 = self.global_crop(image)
            g2 = self.global_crop(image)
            crops.append(self.global_trans1(g1))
            crops.append(self.global_trans2(g2))
        else:
            crops.append(self.val_trans1(image))
            crops.append(self.val_trans2(image))
        return crops


DataAugmentationSWAV = MultiCropsDataAugmentation
DataAugmentationDINO = MultiCropsDataAugmentation
DataAugmentationMOCO = MultiCropsDataAugmentation