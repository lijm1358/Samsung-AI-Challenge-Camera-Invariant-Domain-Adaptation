import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FisheyeDataset
from datasets.augmentations import BaseAugmentation
from models import UNet
import models
import segmentation_models_pytorch as smp
from easydict import EasyDict
import yaml


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def main():
    with open("./config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    
    transform = BaseAugmentation(resize=cfg.train_dataset.transform.args.resize)
    test_dataset = FisheyeDataset(csv_file="./data/test.csv", transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.model.lib == "smp":
        model = getattr(smp, cfg.model.type)(**cfg.model.args).to(device)
    else:
        model = getattr(models, cfg.model.type)(**cfg.model.args).to(device)
    ckpt = torch.load("./experiments/042_20230925_115651_segformer_finetune/best.pt")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    upsample = nn.Upsample((540, 960), mode="bilinear")
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            outputs = model(images)
            outputs = upsample(outputs)
            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()
            # batch에 존재하는 각 이미지에 대해서 반복
            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred)  # 이미지로 변환
                pred = pred.resize((960, 540), Image.NEAREST)  # 960 x 540 사이즈로 변환
                pred = np.array(pred)  # 다시 수치로 변환
                # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0:  # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else:  # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)

    submit = pd.read_csv("./data/sample_submission.csv")
    submit["mask_rle"] = result
    submit.to_csv("./segformer_pretrained.csv", index=False)


if __name__ == "__main__":
    main()
