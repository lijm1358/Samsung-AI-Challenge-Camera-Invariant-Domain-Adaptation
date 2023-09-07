import models
import datasets

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

import yaml
from easydict import EasyDict


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
        iou += inter / union
        
    if iou == np.nan:
        return 0
    return iou / C 

def main(args):
    device = args.device
    # model 초기화
    
    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    train_ds = getattr(datasets, args.train_dataset.type)(csv_file='.\\open\\train_source.csv', transform=transform)
    val_ds = getattr(datasets, args.val_dataset.type)(csv_file='.\\open\\val_source.csv', transform=transform)
    train_dataloader = DataLoader(train_ds, **args.train_dataset.args)
    val_dataloader = DataLoader(val_ds, **args.val_dataset.args)

    print("train dataset length: ", len(train_ds))
    print("val dataset length: ", len(val_ds))
    
    model = getattr(models, args.model.type)().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(args.epochs):  # 20 에폭 동안 학습합니다.
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
        epoch_loss = 0
        epoch_metric = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(val_dataloader)):
                images = images.float().to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))

                epoch_loss += loss.item()
                epoch_metric += mIoU(outputs, masks)

        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(val_dataloader)}, mIoU: {epoch_metric/len(val_dataloader)}')
    

if __name__ == '__main__':
    with open("./config.yaml") as f:
        args = yaml.safe_load(f)
    args = EasyDict(args)
    main(args)