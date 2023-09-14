from tqdm import tqdm
import torch

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
        train_epoch_metric += metric(outputs, masks)
        
    return {"train_loss": train_epoch_loss/len(train_dataloader), "train_metric": train_epoch_metric/len(train_dataloader)}


def validator(model, val_dataloaders, criterion, metric, device, *args, **kwargs):
    model.eval()
    val_loss_list = {}
    val_metric_list = {}
    for i, val_dataloader in enumerate(val_dataloaders):
        epoch_loss_val = 0
        epoch_metric_val = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(tqdm(val_dataloader)):
                images = images.float().to(device)
                masks = masks.long().to(device)

                outputs = model(images)
                loss = criterion(outputs, masks.squeeze(1))

                epoch_loss_val += loss.item()
                epoch_metric_val += metric(outputs, masks)

        val_loss_list[f"ds{i}"] = epoch_loss_val/len(val_dataloader)
        val_metric_list[f"ds{i}"] = epoch_metric_val/len(val_dataloader)
        
        return val_loss_list, val_metric_list