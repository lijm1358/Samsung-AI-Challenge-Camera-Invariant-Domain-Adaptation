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

import segmentation_models_pytorch as smp


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
        expr_num = 1
    else:
        expr_num = int(sorted(os.listdir("./experiments"), key=lambda x: int(x[:3]))[-1][:3]) + 1
    expr_save_path = os.path.join("./experiments", f"{expr_num:03d}-{curdate}-{args.wandb.run_name}")
    os.makedirs(expr_save_path, exist_ok=True)
    
    if args.wandb.use:
        wandb.init(entity="lijm1358", project="fisheye_segmentation", name=args.wandb.run_name, config=args)
    
    device = args.device
    
    train_transform = getattr(datasets.augmentations, args.train_dataset.transform.type)(**args.train_dataset.transform.args)
    # train validation dataset, dataloader
    train_ds = getattr(datasets, args.train_dataset.type)(csv_file='./data/train_source.csv', transform=train_transform)
    train_dataloader = DataLoader(train_ds, **args.train_dataset.args)
    
    val_dataloaders = []
    for val_ds in args.val_dataset:
        val_transform = getattr(datasets.augmentations, val_ds.transform.type)(**val_ds.transform.args)
        val_dataset = getattr(datasets, val_ds.type)(csv_file='./data/val_source.csv', transform=val_transform)
        val_dataloaders.append(DataLoader(val_dataset, **val_ds.args))

    print("train dataset length: ", len(train_ds))
    print("val dataset length: ", len(val_ds))
    
    if args.model.lib == "smp":
        print(getattr(smp, args.model.type))
        model = getattr(smp, args.model.type)(**args.model.args).to(device)
    else:
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
        
    best_val_metric = 0
    patience = args.earlystop_patience
    earlystop_counter = 0

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
        model.eval()
        val_loss_list = []
        val_metric_list = []
        for val_dataloader in val_dataloaders:
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
            val_loss_list.append(epoch_loss_val/len(val_dataloader))
            val_metric_list.append(epoch_metric_val/len(val_dataloader))
        
        wandb.log({
            "train_loss": epoch_loss/len(train_dataloader),
            "train_mIoU": epoch_metric/len(train_dataloader),
            "val_loss_1": val_loss_list[0],
            "val_mIoU_1": val_metric_list[0],
            "val_loss_2": val_loss_list[1],
            "val_mIoU_2": val_metric_list[1],
        })
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(expr_save_path, f"{epoch+1:02d}.pt"))
        
        if best_val_metric < epoch_metric_val/len(val_dataloader):
            print(f"validation metric improved from {best_val_metric:.4f} to {epoch_metric_val/len(val_dataloader):.4f}")
            best_val_metric = epoch_metric_val/len(val_dataloader)
            earlystop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(expr_save_path, "best.pt"))
        else:
            earlystop_counter += 1
            if earlystop_counter >= patience:
                print("early stopping")
                break
    

if __name__ == '__main__':
    with open("./config.yaml") as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)
    main(args)