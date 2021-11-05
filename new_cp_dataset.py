import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import random

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
                anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)

            anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
            
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

train_transform = A.Compose([
    ToTensorV2()
])

val_transform = A.Compose([
    ToTensorV2()
])

test_transform = A.Compose([
    ToTensorV2()
])
