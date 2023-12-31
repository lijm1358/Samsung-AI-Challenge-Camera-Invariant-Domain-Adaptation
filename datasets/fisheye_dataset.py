import os

import cv2
import pandas as pd
from torch.utils.data import Dataset


class FisheyeDataset(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):
        self.csv_path = "./data"
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            os.path.normpath(self.csv_path), os.path.normpath(self.data.iloc[idx, 1])
        )
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)["image"]
            return image

        mask_path = os.path.join(self.csv_path, self.data.iloc[idx, 2])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12  # 배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask
