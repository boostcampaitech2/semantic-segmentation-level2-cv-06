import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module

import torch
from datasets.dataset import CustomDataLoader, collate_fn
from torch.utils.data import DataLoader
import albumentations as A
from datasets.transform_test import create_transforms

@torch.no_grad()
def inference(model_dir, args):
    print("Start prediction..")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.custom_trs:
        #override
        custom = create_transforms(criterion_name = 'transunet', seed = None)
        test_transform = custom.test_transform_img
    else:
        from datasets.dataset import test_transform

    test_dataset = CustomDataLoader(data_dir=args.test_path, mode='test', transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        collate_fn=collate_fn
    )

    model_module = getattr(import_module("models.model"), args.model)
    model = model_module(
        num_classes=11, pretrained=True
    )

    model_path = os.path.join(model_dir)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(device)

    torch.save(model.state_dict(), args.save_dir)
        
    
    
    print(f"End save at {args.save_dir}")

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--model', type=str, default='FCNRes50', help='model type (default: FCNRes50)')

    # Container environment
    parser.add_argument('--test_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/test.json'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # custom args
    parser.add_argument('--custom_trs', default=False, help='option for custom transform function')
    parser.add_argument('--save_dir', type = str, help='option for custom transform function')
    
    

    args = parser.parse_args()



    model_dir = args.model_dir
    os.makedirs('/opt/ml/segmentation/semantic-segmentation-level2-cv-06/save_state/', exist_ok=True)

    inference(model_dir, args)

