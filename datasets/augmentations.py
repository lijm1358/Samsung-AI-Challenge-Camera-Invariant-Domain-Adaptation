import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
import numpy as np

class BaseAugmentation():
    def __init__(self, p=0.5):
        super(BaseAugmentation, self).__init__()
        
        self.transform = A.Compose(
            [   
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()
            ]
        )

    def __call__(self, image, mask):
        return self.transform(image=image, mask=mask)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"
    
class FisheyeAugmentation():
    def __init__(self, p=0.5):
        self.row_s = None
        self.row_e = None
        self.col_s = None
        self.col_e = None
        self.transform = A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(),
                ToTensorV2()
            ]
        )
        
    def apply_fisheye(self, image):
        # 이미지 크기 가져오기
        height, width = image.shape[:2]

        # 카메라 매트릭스 생성
        focal_length = width / 8
        center_x = width / 2
        center_y = height / 2
        camera_matrix = np.array([[width/5, 0, center_x],
                                [0, width/6, center_y],
                                [0, 0, 1]], dtype=np.float32)

        # 왜곡 계수 생성
        dist_coeffs = np.array([0, 0.5, 0, 0], dtype=np.float32)

        # 왜곡 보정
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        return undistorted_image
    
    def find_crop_axis(self, image):
        width, height = image.shape[:2]
        row_blank = np.zeros((width, ))
        col_blank = np.zeros((1, height))
        
        for i, row in enumerate(image):
            if self.row_s is None and not np.equal(row_blank, row).all():
                self.row_s = row
            elif self.row_s is not None and np.equal(row_blank, row).all():
                self.row_e = row
        
    def crop_center(self, image):
        return image[self.row_s:self.row_e, :]

    def __call__(self, image, mask):
        fisheye_image = self.apply_fisheye(image)
        fisheye_mask = self.apply_fisheye(mask)
        
        self.find_crop_axis(fisheye_image)
        fisheye_image = self.crop_center(fisheye_image)
        fisheye_mask = self.crop_center(fisheye_mask)
        return self.transform(image=fisheye_image, mask=fisheye_mask)

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p})"