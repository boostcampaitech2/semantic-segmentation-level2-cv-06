import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from importlib import import_module

import torch
from dataset import CustomDataLoader, train_transform, val_transform, test_transform, train_collate_fn, test_collate_fn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


@torch.no_grad()
def inference(model_dir, args):
    print("Start prediction..")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    test_dataset = CustomDataLoader(data_dir=args.test_path, mode='test', transform=test_transform)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        # collate_fn=collate_fn
    )

    model_module = getattr(import_module("models.model"), args.model)
    model = model_module(
        num_classes=11, pretrained=True
    )

    model_path = os.path.join(model_dir, 'best.pt')
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model = model.to(device)

    size = 256
    transform = A.Compose([A.Resize(size, size)])

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.compat.long)

    with torch.no_grad():
        for step, imgs in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            if args.model in ('FCNRes50', 'FCNRes101', 'DeepLabV3_Res50', 'DeepLabV3_Res101'):
                outs = model(imgs.to(device))['out']
            else:
                outs = model(imgs.to(device))
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(imgs, oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    
    print("End prediction!")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for validing (default: 16)')
    parser.add_argument('--model', type=str, default='FCNRes50', help='model type (default: FCNRes50)')

    # Container environment
    parser.add_argument('--test_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/test.json'))
    parser.add_argument('--model_di', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # load sample_submission.csv
    submission = pd.read_csv('../baseline_code/submission/sample_submission.csv', index_col=None)

    # prediction using test set
    file_names, preds = inference(model_dir, args)

    # write PredictionString
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                                       ignore_index=True)
    
    # save submission.csv
    submission.to_csv(output_dir+'/submission.csv', index=False)
