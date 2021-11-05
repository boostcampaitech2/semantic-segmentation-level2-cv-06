import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import Dataset

# copy paste
from datasets.copy_paste import CopyPaste
from datasets.transform_test import RandomAugMix

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

dataset_path = "./sample_data/"
category_names = [
    "Background",
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
]


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


class CustomDataLoader(Dataset):
    """
    coco format
    """

    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

        # Load the categories in a variable
        self.cat_ids = self.coco.getCatIds()
        self.cats = self.coco.loadCats(self.cat_ids)

    def __getitem__(self, index):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))

            # General trash = 1, ... , Clothing = 10
            anns = sorted(anns, key=lambda idx: idx["area"], reverse=True)
            for i in range(len(anns)):
                masks[self.coco.annToMask(anns[i]) == 1] = anns[i]["category_id"]
            masks = masks.astype(np.int8)

            # transform -> albumentations
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks  # image_infos
        elif self.mode == "test":
            # transform -> albumentations
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
        else:
            raise RuntimeError("CustomDataLoader mode error")

    def __len__(self):
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def make_final_mask(data):
    image = data["image"]
    bboxes = data["bboxes"]
    masks = data["masks"]

    category = np.array([b[-2] for b in bboxes])
    final_masks = np.zeros((512, 512))
    pmasks = [[m, c] for m, c in zip(masks, category)]
    pmasks = sorted(pmasks, key=lambda x: len(x[0]), reverse=True)

    for i in range(len(pmasks)):
        final_masks[pmasks[i][0] == 1] = pmasks[i][1]
    final_masks = final_masks.astype(np.int8)
    final_masks = torch.tensor(final_masks)
    return image, final_masks


def cp_collate_fn(batch):
    new_batch = [[], []]
    for i in batch:
        data = make_final_mask(i)
        new_batch[0].append(data[0])
        new_batch[1].append(data[1])
    return new_batch


def collate_fn(batch):
    return tuple(zip(*batch))


train_transform = A.Compose([ToTensorV2()])

train_augmix_transform = A.Compose(
    [RandomAugMix(severity=3, width=14, alpha=1.0, p=1), ToTensorV2()]
)

train_copypaste_transform = A.Compose(
    [
        CopyPaste(
            blend=True, sigma=1, pct_objects_paste=0.4, p=1.0
        ),  # pct_objects_paste is a guess
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="coco", min_visibility=0.05),
)

val_transform = A.Compose([ToTensorV2()])

test_transform = A.Compose([ToTensorV2()])
