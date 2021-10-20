import argparse
import glob
import json
import os
import re
import random
from pathlib import Path
from importlib import import_module

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from dataset import CustomDataLoader, collate_fn, train_transform, val_transform
from loss import create_criterion
from utils import add_hist, grid_image, label_accuracy_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # settings
    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.device_count())

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset
    train_dataset = CustomDataLoader(data_dir=args.train_path, mode='train', transform=train_transform)
    val_dataset = CustomDataLoader(data_dir=args.val_path, mode='val', transform=val_transform)
    
    # data_loader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        drop_last=True
    )

    # model
    n_classes = 11
    
    model_module = getattr(import_module("model"), args.model)
    model = model_module(
        num_classes=n_classes, pretrained=True
    )
    wandb.watch(model)

    # loss & optimizer
    criterion = create_criterion(
        args.criterion,
        # if weighted cross-entropy
        # weight=torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]).to(device)
        )

    # 여러 옵티마이저 가능하게 수정 필요
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-6)

    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    # start train
    category_names = ['Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                      'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']
    best_val_mIoU = 0
    step = 0
    for epoch in range(args.epochs):
        print(f'Start training..')

        # train loop
        model.train()
        
        hist = np.zeros((n_classes, n_classes))
        for images, masks, _ in train_loader:
            images = torch.stack(images)
            masks = torch.stack(masks).long()
            
            # gpu device 할당
            images, masks = images.to(device), masks.to(device)
            model = model.to(device)

            # inference
            if args.model in ('FCNRes50', 'FCNRes101', 'DeepLabV3_Res50', 'DeepLabV3_Res101'):
                outputs = model(images)['out']
            else:
                outputs = model(images)

            # calculate loss
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 데이터 검증
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            hist = add_hist(hist, masks, outputs, n_class=n_classes)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # step 주기에 따른 loss 출력
            if (step + 1) % args.log_interval == 0:
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}] Step [{step+1}/{len(train_loader)}] || "
                    f"training loss {round(loss.item(),4)} || mIoU {round(mIoU,4)} || lr {current_lr}"
                )

                # wandb log
                if args.wandb == True:
                    wandb.log({
                        "Train/Train loss": round(loss.item(), 4),
                        "Train/Train mIoU": round(mIoU.item(), 4),
                        "Train/Train acc": round(acc.item(), 4),
                        "learning_rate": current_lr
                        },
                        step=step)

            step += 1

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            
            total_loss = 0
            cnt = 0
            figure = None

            hist = np.zeros((n_classes, n_classes))
            for images, masks, _ in val_loader:
                images = torch.stack(images)
                masks = torch.stack(masks).long()

                # gpu device 할당
                images, masks = images.to(device), masks.to(device)
                model = model.to(device)
                
                # inference
                if args.model in ('FCNRes50', 'FCNRes101', 'DeepLabV3_Res50', 'DeepLabV3_Res101'):
                    outputs = model(images)['out']
                else:
                    outputs = model(images)

                # calculate loss
                loss = criterion(outputs, masks)
                total_loss += loss
                cnt += 1

                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                masks = masks.detach().cpu().numpy()
                
                hist = add_hist(hist, masks, outputs, n_class=n_classes)

                if figure is None:
                    figure = grid_image(images.detach().cpu().permute(0, 2, 3, 1).numpy(), masks, outputs)

            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            IoU_by_class = [{classes : round(IoU, 4)} for IoU, classes in zip(IoU, category_names)]

            avg_loss = total_loss / cnt
            print(f"[Val] Average Loss : {round(avg_loss.item(), 4)}, Accuracy : {round(acc, 4)} || "
                  f"mIoU : {round(mIoU, 4)}, IoU by class : {IoU_by_class}")
            

            # save best model
            if mIoU > best_val_mIoU:
                best_val_mIoU = mIoU
                print(f"Best performance {best_val_mIoU} at Epoch {epoch+1}")

                torch.save(model, f"{save_dir}/best.pt")
                print(f"Save best model in {save_dir}")
            
            torch.save(model, f"{save_dir}/last.pt")

            # wandb log
            if args.wandb == True:
                wandb.log({
                    "Media/predict images": figure,
                    "Valid/Valid loss": round(avg_loss.item(), 4),
                    "Valid/Valid mIoU": round(mIoU, 4),
                    "Valid/Valid acc": round(acc, 4),
                    "Metric/Background_IoU": IoU_by_class[0]['Background'], "Metric/General_trash_IoU": IoU_by_class[1]['General trash'], "Metric/Paper_IoU": IoU_by_class[2]['Paper'],
                    "Metric/Paper_pack_IoU": IoU_by_class[3]['Paper pack'], "Metric/Metal_IoU": IoU_by_class[4]['Metal'], "Metric/Glass_IoU": IoU_by_class[5]['Glass'],
                    "Metric/Plastic_IoU": IoU_by_class[6]['Plastic'], "Metric/Styrofoam_IoU": IoU_by_class[7]['Styrofoam'], "Metric/Plastic_bag_IoU": IoU_by_class[8]['Plastic bag'],
                    "Metric/Battery_IoU": IoU_by_class[9]['Battery'], "Metric/Clothing_IoU": IoU_by_class[10]['Clothing']
                    },
                    step=step)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=1004, help='random seed (default: 1004)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training (default: 16)')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for training (default: 4)')
    parser.add_argument('--model', type=str, default='FCNRes50', help='model type (default: FCNRes50)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    
    # Container environment
    parser.add_argument('--train_path', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/segmentation/input/data/train.json'))
    parser.add_argument('--val_path', type=str, default=os.environ.get('SM_CHANNEL_VAL', '/opt/ml/segmentation/input/data/val.json'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # wandb
    parser.add_argument('--wandb', type=bool, default=False, help='wandb implement or not (default: False)')
    parser.add_argument('--entity', type=str, default='cider6', help='wandb entity name (default: cider6)')
    parser.add_argument('--project', type=str, default='test', help='wandb project name (default: test)')

    args = parser.parse_args()
    print(args)

    # wandb init
    if args.wandb == True:
        wandb.init(entity=args.entity, project=args.project)
        wandb.run.name = args.name
        wandb.config.update(args)
        
    model_dir = args.model_dir

    train(model_dir, args)