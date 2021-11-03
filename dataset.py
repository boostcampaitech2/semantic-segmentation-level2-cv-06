import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import random

# from PIL import Image, ImageOps, ImageEnhance
# from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransformCustom
# from albumentations.augmentations import functional as F


dataset_path = '/opt/ml/segmentation/semantic-segmentation-level2-cv-06/input/data/'
category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                  'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""
    def __init__(self, data_dir, mode = 'train', transform = None, augmix = None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)
        self.augmix = augmix 
        
    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]
        
        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id + 1" 할당
            # Background = 0
            # # Unknown = 1, General trash = 2, ... , Cigarette = 11

            ##--original--
            masks = np.zeros((image_infos["height"], image_infos["width"]))

            if self.mode == 'val': 
                # anns = sorted(anns, key=lambda idx : len(idx['segmentation']), reverse=True)
                anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)

            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
            ## --original--


            # anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            # masks = np.zeros((image_infos["height"], image_infos["width"], len(anns)))
            # # General trash = 1, ... , Cigarette = 10
            
            # for i in range(len(anns)):
            #     mask = np.zeros((image_infos["height"], image_infos["width"]))
            #     className = get_classname(anns[i]['category_id'], cats)
            #     pixel_value = category_names.index(className)
            #     mask[self.coco.annToMask(anns[i]) == 1] = pixel_value
            #     masks[:, :, i] = mask
            # masks = masks.astype(np.int8)
            
            if self.augmix:
                r = np.random.rand(1) 
                if r <= 0.5:
                    images, masks = self.augmix_search(images, masks)  
                    
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            images /= 255.
            return images, masks, image_infos
        
        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            images /= 255.
            return images, image_infos
    
    
    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())
    
    def augmix_search(self, images, masks):
      # image 3, 512, 512 ,mask: 512, 512 (둘 다 numpy)
        tfms = A.Compose([
                    # A.Resize(384, 384, p=1.0)
                    A.GridDistortion(p=0.3, distort_limit=[-0.01, 0.01]),
                    A.Rotate(limit=60, p=1.0),
                    A.VerticalFlip(p=0.5),
                    A.HorizontalFlip(p=0.5)
              ])
        
        num = [3, 4, 5, 9, 10]

        label = random.choice(num)  # ex) 4
        # print('label',label)
        # print('len(self.augmix[label])', len(self.augmix[label]))
        idx = np.random.randint(len(self.augmix[label]))
        augmix_img = self.augmix[label][idx]
        # print('first augmix_img', augmix_img.shape)
        augmix_mask = np.zeros((512, 512))
        # augmix img가 있는 만큼 label로 mask를 채워줌
        augmix_mask[augmix_img[:, :, 0] != 0] = label
        ################################################## 새로 추가한 transform을 적용해보자 
        transformed=tfms(image=augmix_img, mask=augmix_mask)
        augmix_img = transformed['image']
        # print('second augmix_img', augmix_img.shape)
        augmix_mask = transformed['mask']
        # print('third augmix_mask', augmix_mask.shape)
        # print('masks', masks.shape)
        ####################################################
        images[augmix_img != 0] = augmix_img[augmix_img != 0]
        masks[augmix_mask != 0] = augmix_mask[augmix_mask != 0]

        return images, masks


# class CustomDataLoader(Dataset):
#     """
#     coco format
#     """
#     def __init__(self, data_dir, mode='train', transform=None):
#         super().__init__()
#         self.mode = mode
#         self.transform = transform
#         self.coco = COCO(data_dir)

#         # Load the categories in a variable
#         self.cat_ids = self.coco.getCatIds()
#         self.cats = self.coco.loadCats(self.cat_ids)
    
#     def __getitem__(self, index):
#         # dataset이 index되어 list처럼 동작
#         image_id = self.coco.getImgIds(imgIds=index)
#         image_infos = self.coco.loadImgs(image_id)[0]

#         # cv2를 활용하여 image 불러오기
#         images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
#         images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
#         images /= 255.0

#         if (self.mode in ('train', 'val')):
#             ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
#             anns = self.coco.loadAnns(ann_ids)

#             # masks : size가 (height x width)인 2D
#             # 각각의 pixel 값에는 "category id" 할당
#             # Background = 0
#             masks = np.zeros((image_infos["height"], image_infos["width"]))

#             # General trash = 1, ... , Clothing = 10
#             # anns = sorted(anns, key=lambda idx: len(idx['segmentation'][0]), reverse=True)
#             anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            
#             for i in range(len(anns)):
#                 # className = get_classname(anns[i]['category_id'], self.cats)
#                 # pixel_value = category_names.index(className)
#                 # masks[self.coco.annToMask(anns[i])==1] = pixel_value
#                 masks[self.coco.annToMask(anns[i])==1] = anns[i]['category_id']
#             masks = masks.astype(np.int8)

#             # [기존] transform -> albumentations
#             if self.transform is not None:
#                 transformed = self.transform(image=images, mask=masks)
#                 images = transformed["image"]
#                 masks = transformed["mask"]


#             # [마스크 돌리기] transform -> albumentations
#             # if self.tr            ansform is not None:
#             #     transformed = self.transform(imageAndMask=[images, masks])
#             #     images, masks = transformed["imageAndMask"]
#             # images = (torch.from_numpy(images)).permute([2,0,1])
#             # masks = torch.from_numpy(masks)

#             return images, masks, image_infos
#         elif self.mode == 'test':
#             # transform -> albumentations
#             if self.transform is not None:
#                 transformed = self.transform(image=images)
#                 images = transformed["image"]
#             return images, image_infos
#         else:
#             raise RuntimeError("CustomDataLoader mode error")
    
#     def __len__(self):
#         # 전체 dataset의 size를 return
#         return len(self.coco.getImgIds())


def collate_fn(batch):
    return tuple(zip(*batch))


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled to
      level/PARAMETER_MAX.
    Returns:
    A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, level, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, level, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, level, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.size[0] / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(pil_img.size,
                           Image.AFFINE, (1, 0, 0, 0, 1, level),
                           resample=Image.BILINEAR)


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast, equalize, posterize, solarize, 
    # rotate, shear_x, shear_y,
    # translate_x, translate_y
]

augmentations_all = [
    autocontrast, equalize, posterize, solarize, 
    rotate, shear_x, shear_y,
    translate_x, translate_y, 
    color, contrast, brightness, sharpness
]

def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    return image - 127

def apply_op(image, op, severity):
    #   image = np.clip(image, 0, 255)

    image = image * 255
    image = image.astype(np.uint8)
    
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)

    return np.asarray(pil_img)

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(
      np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image).astype(np.float32)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations_all)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
#         mix += ws[i] * normalize(image_aug)

    # mixed = (1 - m) * image + m * mix
    mixed = m * (image * 255) + (1 - m) * mix
    # mixed = (1 - m) * normalize(image) + m * mix

    return mixed  / 255


# class RandomAugMix(ImageOnlyTransform):

#     def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):
#         super().__init__(always_apply, p)
#         self.severity = severity
#         self.width = width
#         self.depth = depth
#         self.alpha = alpha

#     def apply(self, image, **params):
#         image = augment_and_mix(
#             image,
#             self.severity,
#             self.width,
#             self.depth,
#             self.alpha
#         )
#         return image


train_transform = A.Compose([
    # RandomAugMix(severity=3, width=3, alpha=1., p=1),
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])

test_transform = A.Compose([
    ToTensorV2()
])
