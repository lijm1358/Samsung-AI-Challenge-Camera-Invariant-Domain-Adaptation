import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from time import time


def base_trainer(model, train_dataloader, optimizer, criterion, metric, device, *args, **kwargs):
    model.train()
    train_epoch_loss = 0
    train_epoch_metric = 0
    for i, (images, masks) in enumerate(tqdm(train_dataloader)):
        images = images.float().to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.squeeze(1))
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        train_epoch_metric += metric(outputs, masks).item()

    return (
        {"train_loss": train_epoch_loss / len(train_dataloader)},
        {"train_metric": train_epoch_metric / len(train_dataloader)},
    )
    
def adaptseg_trainer(
        model_seg, model_D1, model_D2, train_dataloader, target_dataloader, optimizer_seg, optimizer_d1, optimizer_d2, criterion, metric, device
    ):
    bce_loss = nn.BCEWithLogitsLoss()
    
    interp = nn.Upsample(size=(448, 448), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(448, 448), mode='bilinear', align_corners=True)

    loss_seg_value1 = 0
    loss_adv_target_value1 = 0
    loss_D_value1 = 0

    loss_seg_value2 = 0
    loss_adv_target_value2 = 0
    loss_D_value2 = 0
    
    miou_value = 0
    
    model_seg.train()
    model_D1.train()
    model_D2.train()
    
    # 2195
    
    for i, ((images, masks), (images_t)) in enumerate(zip(tqdm(train_dataloader), target_dataloader)):
        optimizer_seg.zero_grad()
        optimizer_d1.zero_grad()
        optimizer_d2.zero_grad()
        
        for param in model_D1.parameters():
            param.requires_grad = False
        for param in model_D2.parameters():
            param.requires_grad = False
            
        images = images.float().to(device)
        masks = masks.long().to(device)
        
        # source segmentation training
        pred1, pred2 = model_seg(images)
        pred1 = interp(pred1)
        pred2 = interp(pred2)
        
        loss_seg1 = criterion(pred1, masks)
        loss_seg2 = criterion(pred2, masks)
        loss = loss_seg2 + 0.1 * loss_seg1
        
        loss.backward()
        loss_seg_value1 += loss_seg1.item()
        loss_seg_value2 += loss_seg2.item()
        
        miou_value += metric(pred2, masks).item()
        
        # target segmentation training
        images_t = images_t.float().to(device)
        
        pred_target1, pred_target2 = model_seg(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)
        
        D_out1 = model_D1(F.softmax(pred_target1, dim=1))
        D_out2 = model_D2(F.softmax(pred_target2, dim=1))
        
        loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(0).to(device))
        loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(0).to(device))
        
        loss = 0.0002 * loss_adv_target1 + 0.001 * loss_adv_target2
        loss.backward()
        loss_adv_target_value1 += loss_adv_target1.item()
        loss_adv_target_value2 += loss_adv_target2.item()
        
        # discriminator training
        for params in model_D1.parameters():
            params.requires_grad = True
        for params in model_D2.parameters():
            params.requires_grad = True
            
        pred1 = pred1.detach()
        pred2 = pred2.detach()

        D_out1 = model_D1(F.softmax(pred1, dim=1))
        D_out2 = model_D2(F.softmax(pred2, dim=1))
        
        loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(0).to(device))
        loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(0).to(device))
        
        loss_D1 /= 2
        loss_D2 /= 2
        
        loss_D1.backward()
        loss_D2.backward()
        
        loss_D_value1 += loss_D1.item()
        loss_D_value2 += loss_D2.item()
        
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        
        D_out1 = model_D1(F.softmax(pred_target1, dim=1))
        D_out2 = model_D2(F.softmax(pred_target2, dim=1))
        
        loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(1).to(device))
        loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(1).to(device))
        
        loss_D1 = loss_D1 / 2
        loss_D2 = loss_D2 / 2
        
        loss_D1.backward()
        loss_D2.backward()

        loss_D_value1 += loss_D1.item()
        loss_D_value2 += loss_D2.item()
        
        optimizer_seg.step()
        optimizer_d1.step()
        optimizer_d2.step()

    return {"train_loss_seg_1": loss_seg_value1 / len(train_dataloader),
        "train_loss_adv_1": loss_adv_target_value1 / len(train_dataloader),
        "train_loss_D_1": loss_D_value1 / len(train_dataloader),
        "train_loss_seg_2": loss_seg_value2 / len(train_dataloader),
        "train_loss_adv_2": loss_adv_target_value2 / len(train_dataloader),
        "train_loss_D_2": loss_D_value2 / len(train_dataloader),
        "train_metric": miou_value / len(train_dataloader)}

def validator(model, val_dataloaders, criterion, metric, device, *args, **kwargs):
    interp = nn.Upsample(size=(448, 448), mode='bilinear', align_corners=True)
    
    model.eval()
    val_loss_list = {}
    val_metric_list = {}
    for i, val_dataloader in enumerate(val_dataloaders):
        epoch_loss_val = 0
        epoch_metric_val = 0
        with torch.no_grad():
            for j, (images, masks) in enumerate(tqdm(val_dataloader)):
                images = images.float().to(device)
                masks = masks.long().to(device)

                _, outputs = model(images)
                outputs = interp(outputs)
                loss = criterion(outputs, masks.squeeze(1))

                epoch_loss_val += loss.item()
                epoch_metric_val += metric(outputs, masks).item()

        val_loss_list[f"val_loss_{i}"] = epoch_loss_val / len(val_dataloader)
        val_metric_list[f"val_metric_{i}"] = epoch_metric_val / len(val_dataloader)

    return val_loss_list, val_metric_list
