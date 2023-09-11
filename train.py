import models
import datasets
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

import yaml
from easydict import EasyDict
import wandb


def mIoU(input, target):
    # input: (N, C, H, W) | target: (N, H, W)
    N, C, H, W = input.shape
    input = torch.sigmoid(input)
    input = input.argmax(dim=1) # (N, H, W)
    iou = 0
    for cls in range(C):
        pred = (input == cls)
        true = (target == cls)
        inter = (pred & true).sum()
        union = (pred | true).sum()
        iou += inter / (union + 1e-8)
    
    return iou / C 

def main(args):
    curdate = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    if os.listdir("./experiments") == []:
        expr_num = "001"
    else:
        expr_num = int(sorted(os.listdir("./experiments"), key=lambda x: int(x[:3]))[-1][:3]) + 1
    expr_save_path = os.path.join("./experiments", f"{expr_num:03d}-{curdate}")
    os.makedirs(expr_save_path, exist_ok=True)
    
    if args.wandb.use:
        wandb.init(entity="lijm1358", project="vmt", name=args.wandb.run_name, config=args)
    
    device = args.device
    
    train_transform = getattr(datasets.augmentations, args.train_dataset.transform)()
    val_transform = getattr(datasets.augmentations, args.train_dataset.transform)()
    # train validation dataset, dataloader
    train_ds = getattr(datasets, args.train_dataset.type)(csv_file='./open/train_source.csv', transform=train_transform)
    val_ds = getattr(datasets, args.val_dataset.type)(csv_file='./open/val_source.csv', transform=val_transform)
    train_dataloader = DataLoader(train_ds, **args.train_dataset.args)
    val_dataloader = DataLoader(val_ds, **args.val_dataset.args)

    print("train dataset length: ", len(train_ds))
    print("val dataset length: ", len(val_ds))
    
    model = getattr(models, args.model.type)().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, args.optimizer.type)(model.parameters(), **args.optimizer.args)
    
    cur_epoch = 0
    # model checkpoint load
    if args.model.load_from is not None:
        checkpoint = torch.load(args.model.load_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        print("model loaded from: ", args.model.load_from)

    # training loop
    for epoch in range(cur_epoch, args.epochs):  # 20 에폭 동안 학습합니다.
        print("------------------------")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("------------------------")
        print('training')
        model.train()
        epoch_loss = 0
        epoch_metric = 0
        for i, (images, masks) in enumerate(tqdm(train_dataloader)):
            images = images.float().to(device)
            masks = masks.long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_metric += mIoU(outputs, masks)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_dataloader)}, mIoU: {epoch_metric/len(train_dataloader)}')
        
        print("validation")
        epoch_loss_val = 0
        epoch_metric_val = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(val_dataloader)):
                images = images.float().to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))

                epoch_loss_val += loss.item()
                epoch_metric_val += mIoU(outputs, masks)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss_val/len(val_dataloader)}, mIoU: {epoch_metric_val/len(val_dataloader)}')
        
        wandb.log({
            "train_loss": epoch_loss/len(train_dataloader),
            "train_mIoU": epoch_metric/len(train_dataloader),
            "val_loss": epoch_loss_val/len(val_dataloader),
            "val_mIoU": epoch_metric_val/len(val_dataloader)
        })
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(expr_save_path, f"{epoch+1:02d}.pt"))
    

if __name__ == '__main__':
    with open("./config.yaml") as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)
    main(args)