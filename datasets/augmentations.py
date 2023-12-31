import random

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class BaseAugmentation:
    def __init__(self, resize=(224, 224), p=0.5):
        super(BaseAugmentation, self).__init__()
        
        if isinstance(resize, str):
            resize = eval(resize)
        elif isinstance(resize, int):
            resize = (resize, resize)

        self.transform = A.Compose([A.Resize(*resize), A.Normalize(), ToTensorV2()])

    def __call__(self, image, mask=None):
        if mask is None:
            return self.transform(image=image)
        return self.transform(image=image, mask=mask)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"


class FisheyeAugmentation:
    def __init__(self, resize=(224, 224), focal_ratio=0.25, center_x_ratio=0.5, center_y_ratio=0.5, p=0.5):
        self.focal = focal_ratio
        self.center_x = center_x_ratio
        self.center_y = center_y_ratio
        self.p = p
        
        if isinstance(resize, str):
            resize = eval(resize)
        elif isinstance(resize, int):
            resize = (resize, resize)

        self.transform = A.Compose([A.Resize(*resize), A.Normalize(), ToTensorV2()])

    def apply_fisheye(self, image, is_mask=False):
        # 이미지 크기 가져오기
        height, width = image.shape[:2]

        if is_mask:
            image += 1

        # 카메라 매트릭스 생성
        focal_length = width * self.focal
        center_x = width * self.center_x
        center_y = height * self.center_y
        camera_matrix = np.array(
            [[focal_length, 0, center_x], [0, focal_length, center_y], [0, 0, 1]], dtype=np.float32
        )

        # 왜곡 계수 생성
        dist_coeffs = np.array([0, 0.5, 0, 0], dtype=np.float32)

        # 왜곡 보정
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

        if is_mask:
            undistorted_image = np.where(undistorted_image == 0, 13, undistorted_image)
            undistorted_image -= 1
        return undistorted_image

    def find_crop_axis(self, image, mask):
        row_s, row_e, col_s, col_e = None, None, None, None
        height, width = image.shape[:2]
        row_blank = np.zeros((width,))
        col_blank = np.zeros((1, height))

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for i, row in enumerate(gray_img):
            if row_s is None and not np.equal(row_blank, row).all():
                row_s = i
            elif row_s is not None and np.equal(row_blank, row).all():
                row_e = i
                break

        for i, col in enumerate(gray_img.T):
            if col_s is None and not np.equal(col_blank, col).all():
                col_s = i
            elif col_s is not None and np.equal(col_blank, col).all():
                col_e = i
                break

        return (image[row_s:row_e, col_s:col_e, :], mask[row_s:row_e, col_s:col_e])

    def __call__(self, image, mask):
        if random.random() < self.p:
            fisheye_image = self.apply_fisheye(image, False)
            fisheye_mask = self.apply_fisheye(mask, True)

            fisheye_image, fisheye_mask = self.find_crop_axis(fisheye_image, fisheye_mask)
        else:
            fisheye_image = image
            fisheye_mask = mask

        return self.transform(image=fisheye_image, mask=fisheye_mask)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"
