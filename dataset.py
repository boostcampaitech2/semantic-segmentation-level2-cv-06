import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from train import RandomAugMix


dataset_path = './input/data'
# category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
#                   'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class CustomDataLoader(Dataset):
    """
    coco format
    """
    def __init__(self, data_dir, mode='train', transform=None):
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
        images = cv2.imread(os.path.join(dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))

            # General trash = 1, ... , Clothing = 10
            anns = sorted(anns, key=lambda idx: len(idx['segmentation'][0]), reverse=False)
            for i in range(len(anns)):
                # className = get_classname(anns[i]['category_id'], self.cats)
                # pixel_value = category_names.index(className)
                # masks[self.coco.annToMask(anns[i])==1] = pixel_value
                masks[self.coco.annToMask(anns[i])==1] = anns[i]['category_id']
            masks = masks.astype(np.int8)

            # transform -> albumentations
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        elif self.mode == 'test':
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


def collate_fn(batch):
    return tuple(zip(*batch))


train_transform = A.Compose([
    RandomAugMix(severity=3, width=3, alpha=1., p=1.),
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])

test_transform = A.Compose([
    ToTensorV2()
])
