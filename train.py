import models
import datasets
import os
from datetime import datetime
import math

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

from utils import make_expr_directory
from runner import base_trainer, validator


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

def main(cfg):
    expr_save_path = make_expr_directory(cfg.expr_save_path, cfg.wandb.run_name)
    
    if cfg.wandb.use:
        wandb.init(entity="lijm1358", project="fisheye_segmentation", name=cfg.wandb.run_name, config=cfg)
    
    device = cfg.device
    
    # train transform, dataset, dataloader
    train_transform = getattr(datasets.augmentations, cfg.train_dataset.transform.type)(**cfg.train_dataset.transform.args)
    train_ds = getattr(datasets, cfg.train_dataset.type)(csv_file=cfg.train_dataset.path, transform=train_transform, **cfg.train_dataset.args)
    train_dataloader = DataLoader(train_ds, **cfg.train_dataset.loader_args)
    
    # validation transform, dataset, dataloader(여러 개 지원)
    val_dataloaders = []
    for val_ds in cfg.val_dataset:
        val_transform = getattr(datasets.augmentations, val_ds.transform.type)(**val_ds.transform.args)
        val_dataset = getattr(datasets, val_ds.type)(csv_file=val_ds.path, transform=val_transform, **val_ds.args)
        val_dataloaders.append(DataLoader(val_dataset, **val_ds.loader_args))

    print("train dataset length: ", len(train_ds))
    print("val dataset length: ", {f"ds{i}": len(val_ds) for i, val_ds in enumerate(val_dataloaders)})
    
    # model 정의
    if cfg.model.lib == "smp":
        model = getattr(smp, cfg.model.type)(**cfg.model.args).to(device)
    else:
        model = getattr(models, cfg.model.type)().to(device)

    # loss function과 optimizer 정의
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, cfg.optimizer.type)(model.parameters(), **cfg.optimizer.args)
    
    cur_epoch = 0
    # model checkpoint load
    if cfg.model.load_from is not None:
        checkpoint = torch.load(cfg.model.load_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cur_epoch = checkpoint['epoch']
        
    # early stopping
    best_criterion_value = math.inf if cfg.earlystop.monitor == "val_loss" else 0
    patience = cfg.earlystop.patience
    earlystop_counter = 0
    
    # 실험 정보 출력
    print()
    print("model: ", cfg.model.type, f"({cfg.model.lib})")
    print("optimizer: ", cfg.optimizer.type)
    print("train dataset: ", cfg.train_dataset.type, f"({cfg.train_dataset.transform.type})")
    for i, val_ds in enumerate(cfg.val_dataset):
        print(f"val dataset {i}: ", val_ds.type, f"({val_ds.transform.type})")
    print("experiment name: ", cfg.wandb.run_name)
    print("experiment save path: ", expr_save_path)
    print()

    # training
    for epoch in range(cur_epoch, cfg.epochs):
        print("------------------------")
        print(f"Epoch {epoch+1}/{cfg.epochs}")
        print("------------------------")
        print('training')
        train_results = base_trainer(model, train_dataloader, optimizer, criterion, mIoU, device)
        
        print("\nvalidation")
        valid_loss_results, valid_metric_results = validator(model, val_dataloaders, criterion, mIoU, device)
        
        # wandb logging
        wandb.log({**train_results, **valid_loss_results, **valid_metric_results})
        
        # save model
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(expr_save_path, f"{epoch+1:02d}.pt"))
        
        # early stopping
        valid_comparison_value = valid_loss_results.values()[0] if cfg.earlystop.monitor == "val_loss" else valid_metric_results.values()[0]
        
        if cfg.earlystop.monitor == "val_loss":
            _best_criterion_value = -best_criterion_value
            _valid_comparison_value = -valid_comparison_value
        else:
            _best_criterion_value = best_criterion_value
            _valid_comparison_value = valid_comparison_value
        
        if _best_criterion_value < _valid_comparison_value:
            print(f"validation metric improved from {best_criterion_value:.4f} to {valid_comparison_value:.4f}")
            valid_metric_results = valid_comparison_value
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
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    main(cfg)