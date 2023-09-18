import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FisheyeDataset
from datasets.augmentations import BaseAugmentation
from models import UNet, DeeplabMulti


def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def main():
    transform = BaseAugmentation(resize=(448, 448))
    test_dataset = FisheyeDataset(csv_file="./data/test.csv", transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = UNet()
    model = DeeplabMulti(num_classes=13, multi=True)
    # model = smp.DeepLabV3(
    #     encoder_name="resnet101",
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=13
    # )
    ckpt = torch.load("./experiments/020_20230917_012553_adaptseg/07.pt")
    model.load_state_dict(ckpt["model_state_dict_seg"])
    model.to(device)
    interp = nn.Upsample(size=(448, 448), mode='bilinear', align_corners=True)

    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            _, outputs = model(images)
            outputs = interp(outputs)
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
    submit.to_csv("./adaptseg.csv", index=False)


if __name__ == "__main__":
    main()
