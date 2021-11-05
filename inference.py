import argparse
import os
from importlib import import_module

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import CustomDataLoader, collate_fn
from datasets.transform_test import create_transforms


@torch.no_grad()
def inference(model_dir, args):
    print("Start prediction..")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.custom_trs:
        #override
        custom = create_transforms(criterion_name = args.model, seed = None)
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
        num_classes=11, pretrained=False
    )

    model_path = os.path.join(model_dir)
    checkpoint = torch.load(model_path, map_location=device) #every model has to be state_dict
    model.load_state_dict(checkpoint)
    model = model.to(device)
    
    size = 256
    transform = A.Compose([A.Resize(size, size)])

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.compat.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader, leave=False)):

            # inference
            if args.model in ('FCNRes50', 'FCNRes101', 'DeepLabV3_Res50', 'DeepLabV3_Res101'):
                outs = model(torch.stack(imgs).to(device))['out']
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            elif args.model in ('MscaleOCRNet'):
                outs = model(torch.stack(imgs).to(device))
                oms = torch.argmax(outs['pred'].squeeze(), dim=1).detach().cpu().numpy()
            else:
                outs = model(torch.stack(imgs).to(device))
                oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
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
    parser.add_argument('--test_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/semantic-segmentation-level2-cv-06/sample_data/train.json'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    # custom args
    parser.add_argument('--custom_trs', default=False, help='option for custom transform function')

    args = parser.parse_args()

    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    # load sample_submission.csv
    submission = pd.read_csv('/opt/ml/segmentation/baseline_code/submission/sample_submission.csv', index_col=None)

    # prediction using test set
    file_names, preds = inference(model_dir, args)

    # write PredictionString / revised for efficiency
    id_list, mask_list = [], []
    for file_name, string in tqdm(zip(file_names, preds), leave=False, total = preds.shape[0]):
        # submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
        #                                ignore_index=True)
        id_list.append(file_name)
        mask_list.append(' '.join(str(e) for e in string.tolist()))

    submission['image_id'] = id_list
    submission['PredictionString'] = mask_list
    
    # save submission.csv
    submission.to_csv(output_dir+'/submission.csv', index=False)
