import torch
from torch import nn
from tqdm import tqdm



def base_trainer(model, train_dataloader, optimizer, criterion, metric, device, *args, **kwargs):
    upsample = nn.Upsample((448, 448), mode="bilinear")
    model.train()
    train_epoch_loss = 0
    train_epoch_metric = 0
    for i, (images, masks) in enumerate(tqdm(train_dataloader)):
        images = images.float().to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        if model.__class__.__name__ in ["ResNetMulti", "ResNetMultiPretrained"]:
            _, outputs = model(images)
            outputs = upsample(outputs)
        else:
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
    
    
def segformer_trainer(model, train_dataloader, optimizer, criterion, metric, device, *args, **kwargs):
    model.train()
    train_epoch_loss = 0
    train_epoch_metric = 0
    for i, (images, masks) in enumerate(tqdm(train_dataloader)):
        images = images.float().to(device)
        masks = masks.long().to(device)

        optimizer.zero_grad()
        loss, logits = model(images, masks)
        loss.backward()
        optimizer.step()
        
        outputs = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        # outputs = upsampled_logits.argmax(dim=1)
    
        train_epoch_loss += loss.item()
        train_epoch_metric += metric(outputs, masks).item()
        
        if i==5:
            break

    return (
        {"train_loss": train_epoch_loss / len(train_dataloader)},
        {"train_metric": train_epoch_metric / len(train_dataloader)},
    )


def validator(model, val_dataloaders, criterion, metric, device, *args, **kwargs):
    upsample = nn.Upsample((448, 448), mode="bilinear")
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

                if model.__class__.__name__ in ["ResNetMulti", "ResNetMultiPretrained"]:
                    _, outputs = model(images)
                    outputs = upsample(outputs)
                else:
                    outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))

                epoch_loss_val += loss.item()
                epoch_metric_val += metric(outputs, masks).item()

        val_loss_list[f"val_loss_{i}"] = epoch_loss_val / len(val_dataloader)
        val_metric_list[f"val_metric_{i}"] = epoch_metric_val / len(val_dataloader)

        return val_loss_list, val_metric_list
        
def segformer_validator(model, val_dataloaders, criterion, metric, device, *args, **kwargs):
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

                loss, logits = model(images, masks)
                outputs = nn.functional.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

                epoch_loss_val += loss.item()
                epoch_metric_val += metric(outputs, masks).item()

        val_loss_list[f"val_loss_{i}"] = epoch_loss_val / len(val_dataloader)
        val_metric_list[f"val_metric_{i}"] = epoch_metric_val / len(val_dataloader)

        return val_loss_list, val_metric_list
