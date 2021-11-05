# anno 정렬 방식에 따라 두가지 방식의 마스크의 차이점을 시각화 해주는 프로그램

import json
import os
import warnings

warnings.filterwarnings("ignore")

import sys

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm

import seaborn as sns

sns.set()


sys.path.insert(1, "/opt/ml/semantic-segmentation-level2-cv-06/")
# from utils import label_accuracy_score, add_hist

plt.rcParams["axes.grid"] = False

dataset_path = "../input/data"
anns_file_path = dataset_path + "/" + "train_all.json"

# Read annotations
with open(anns_file_path, "r") as f:
    dataset = json.loads(f.read())


categories = dataset["categories"]
category_names = []
for cat in categories:
    category_names.append(cat["name"])
category_names.insert(0, "Background")
# category_names

class_colormap = pd.read_csv("../class_dict.csv")
# class_colormap


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]["id"] == classID:
            return cats[i]["name"]
    return "None"


def create_trash_label_colormap():
    """Creates a label colormap used in Trash segmentation.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((11, 3), dtype=np.uint8)
    for inex, (_, r, g, b) in enumerate(class_colormap.values):
        colormap[inex] = [r, g, b]

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
                is the color indexed by the corresponding element in the input label
                to the trash color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
              map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_trash_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


# train.json / validation.json / test.json 디렉토리 설정
train_path = dataset_path + "/train.json"
val_path = dataset_path + "/val.json"
test_path = dataset_path + "/test.json"
batch_size = 32
# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


train_transform = A.Compose([ToTensorV2()])


class CustomDataLoader2(Dataset):
    """COCO format"""

    def __init__(self, data_dir, mode="train", transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.coco = COCO(data_dir)

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(dataset_path, image_infos["file_name"]))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if self.mode in ("train", "val"):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos["id"])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            # anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            anns = sorted(
                anns, key=lambda idx: len(idx["segmentation"][0]), reverse=True
            )
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            masks2 = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(
                anns, key=lambda idx: len(idx["segmentation"][0]), reverse=False
            )
            for i in range(len(anns)):
                className = get_classname(anns[i]["category_id"], cats)
                pixel_value = category_names.index(className)
                masks2[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks2 = masks2.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            origin_image = images
            if self.transform is not None:
                transformed = self.transform(image=origin_image, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
                transformed = self.transform(image=origin_image, mask=masks2)
                masks2 = transformed["mask"]

            return images, masks, image_infos, masks2

        if self.mode == "test":
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


train_dataset2 = CustomDataLoader2(
    data_dir=train_path, mode="train", transform=train_transform
)
train_loader2 = torch.utils.data.DataLoader(
    dataset=train_dataset2,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

figsize = 8

for idx, (imgs, masks, image_infos, masks2) in enumerate(tqdm(train_loader2)):
    if idx < 2000:
        continue
    if torch.any(torch.ne(masks[0], masks2[0])).item() == True:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(figsize * 4, figsize))

        draw_mask = torch.ne(masks[0], masks2[0]).type(torch.int8)

        ax[0].imshow(imgs[0].permute([1, 2, 0]))
        ax[0].grid(False)
        # ax[i,0].set_xlabel(image_infos[0]['file_name'])

        ax[1].imshow(label_to_color_image(draw_mask.detach().cpu().numpy()))
        ax[1].grid(False)

        ax[2].imshow(label_to_color_image(masks[0].detach().cpu().numpy()))
        ax[2].grid(False)

        ax[3].imshow(label_to_color_image(masks2[0].detach().cpu().numpy()))
        ax[3].grid(False)

        plt.savefig(f'{image_infos[0]["file_name"][:-4]}')
