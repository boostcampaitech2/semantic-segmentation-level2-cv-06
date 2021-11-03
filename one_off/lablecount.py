import argparse
import os
import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('/opt/ml/segmentation/semantic-segmentation-level2-cv-06')
from dataset import CustomDataLoader, collate_fn, train_transform, val_transform
from tqdm import tqdm
from collections import Counter



def train(args):

    use_cuda = torch.cuda.is_available()

    # dataset
    train_dataset = CustomDataLoader(data_dir=args.train_path, mode='train', transform=train_transform)

    # data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        drop_last=False
    )


    n_classes = 11
    t_count = dict([(k, 0) for k in range(n_classes)])

    for i, (images, masks, _) in tqdm(enumerate(train_loader), total = len(train_loader)):
        masks_1d = torch.stack(masks).flatten().type(torch.int8).tolist()
        c = Counter(masks_1d)

        for k in t_count.keys():
            t_count[k] += c[k]

    s = sum(t_count.values())
    for k in t_count.keys():
        t_count[k] = round(1 / (t_count[k] / s), 2)

    return t_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Container environment
    parser.add_argument('--train_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/train.json'))
    parser.add_argument('--val_path', type=str, default=os.environ.get('SM_CHANNEL_VAL', '/opt/ml/segmentation/input/data/val.json'))

    args = parser.parse_args()

    counts = train(args)
    print(counts)
    print(counts.values())