import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import json
from collections import Counter

from datasets.dataset import CustomDataLoader, collate_fn, train_transform
from tqdm import tqdm


def pixelcount(args):

    use_cuda = torch.cuda.is_available()

    # dataset
    train_dataset = CustomDataLoader(
        data_dir=args["train_path"], mode="train", transform=train_transform
    )

    # data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        drop_last=False,
    )

    n_classes = 11
    t_count = dict([(k, 0) for k in range(n_classes)])

    for i, (images, masks, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
        masks_1d = torch.stack(masks).flatten().type(torch.int8).tolist()
        c = Counter(masks_1d)

        for k in t_count.keys():
            t_count[k] += c[k]

    s = sum(t_count.values())
    for k in t_count.keys():
        t_count[k] = round(1 / (t_count[k] / s), 2)

    return t_count


def read_json(json_dir: str):
    with open(json_dir) as json_file:
        source_anns = json.load(json_file)

    return source_anns["annotations"]


def instancecount(args, method="ratio"):

    source = read_json(args["train_path"])
    c = Counter([annos["category_id"] for annos in source])

    if method == "ratio":
        s = sum(c.values())
        for k in c.keys():
            c[k] = round(s / c[k], 2)

        c[0] = 1  # 강제로 배경의 가중치는 1이라고 가정
    else:
        c[0] = len(source["images"])  # 강제로 이미지 갯수마다 배경은 한 개라고 가정

    return dict(sorted(c.items(), key=lambda x: x[0]))


def get_weight(mode="instance"):
    args = {}
    args["train_path"] = "../input/data/train.json"
    args["val_path"] = "../input/data/val.json"

    if mode == "instance":
        counts = instancecount(args)
    else:
        counts = pixelcount(args)
    return counts.values()
