import albumentations as A
import numpy as np
import cv2
import torch
import random
from torch.utils.data import DataLoader


from datasets.dataset import CustomDataLoader, collate_fn, train_transform


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


seed_everything(1004)

# dataset
train_dataset = CustomDataLoader(
    data_dir="/opt/ml/segmentation/input/data/train.json",
    mode="train",
    transform=train_transform,
)
use_cuda = torch.cuda.is_available()

# data_loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    pin_memory=use_cuda,
    collate_fn=collate_fn,
    drop_last=True,
)


classdict = {
    3: [],  # 3: 'Paper pack'
    4: [],  # 4: 'Metal'
    5: [],  # 5: 'Glass'
    9: [],  # 9: 'Battery'
    10: [],  # 10: 'Clothing'
}

tfms_to_small = A.Compose(
    [
        A.Resize(256, 256),
        A.PadIfNeeded(512, 512, border_mode=0),
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512),
    ]
)
tfms_to_big = A.Compose(
    [
        A.CropNonEmptyMaskIfExists(256, 256, ignore_values=[0]),
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1.0),
        A.Resize(512, 512),
    ]
)
tfms = A.Compose(
    [
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=0.5),
        A.Resize(512, 512),
    ]
)

import copy

# train_loader의 output 결과(image 및 mask) 확인
for imgs, masks, image_infos in train_loader:
    image_infos = image_infos[0]

    mask = masks[0].numpy().astype(np.uint8)
    img = imgs[0].permute([1, 2, 0]).numpy().astype(np.uint8)

    mask[mask == 1] = 0
    mask[mask == 2] = 0
    mask[mask == 6] = 0
    mask[mask == 7] = 0
    mask[mask == 8] = 0
    class_type = np.unique(mask)
    if len(class_type) == 1:
        continue

    mask3d = np.dstack([mask] * 3)
    res = np.where(mask3d, 0, img)
    res1 = cv2.bitwise_and(img, img, mask=mask)
    for j in class_type:
        if j == 0:
            continue

        temp_mask = copy.deepcopy(mask)
        temp_mask[temp_mask != j] = 0
        if (np.sum(temp_mask != 0)) < 400:
            continue
        temp = copy.deepcopy(res1)
        temp = cv2.bitwise_and(temp, temp, mask=temp_mask)
        if np.sum(temp != 0) > 20000:
            transformed = tfms_to_small(image=temp, mask=temp_mask)
            mask = transformed["mask"]
            temp = transformed["image"]

        elif np.sum(temp != 0) < 5000:
            transformed = tfms_to_big(image=temp, mask=temp_mask)
            mask = transformed["mask"]
            temp = transformed["image"]

        else:
            transformed = tfms(image=temp, mask=temp_mask)
            mask = transformed["mask"]
            temp = transformed["image"]
        #        fig, axes = plt.subplots(1, 2)
        #        axes[0].imshow(res)
        #        axes[1].imshow(temp)
        if np.sum(temp != 0) >= 400:
            classdict[j].append(temp)

# save data
import pickle

with open("classdict_cp.pickle", "wb") as fw:
    pickle.dump(classdict, fw)
print("output saved!!!")
