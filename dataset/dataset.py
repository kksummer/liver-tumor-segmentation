import os
import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image


class LiverDataset(Dataset):
    def __init__(self, data_dir, image_name, mask_name, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.image_name = image_name
        self.mask_name = mask_name

        # 获取并排序文件名列表
        self.filenames = sorted(os.listdir(os.path.join(data_dir, mode, self.image_name)))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.mode, self.image_name, self.filenames[idx])
        mask_path = os.path.join(self.data_dir, self.mode, self.mask_name, self.filenames[idx])

        # 检查文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件未找到：{image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩膜文件未找到：{mask_path}")
        # 尝试读取图像和掩膜
        try:
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, 0)
            if image is None:
                raise ValueError(f"无法读取图像：{image_path}")
            if mask is None:
                raise ValueError(f"无法读取掩膜：{mask_path}")

            # 确保图像和掩膜尺寸匹配
            if image.shape[:2] != mask.shape[:2]:
                raise ValueError(f"图像和掩膜尺寸不匹配：{image_path}，{mask_path}")

            # 数据增强和其他处理
            if self.transform is not None:
                image = Image.fromarray(image)
                mask = Image.fromarray(mask)
                image_np = np.array(image)
                augmented = self.transform(image, mask)
                augmented_image, augmented_mask = augmented
                augmented_image_np = np.array(augmented_image)
                augmented_mask_np = np.array(augmented_mask)

                return augmented_image_np, augmented_mask_np, image_np
            else:
                print("没有经过数据增强")
                return image, mask

        except Exception as e:
            # 这里可以处理读取文件过程中可能发生的其他错误
            print(f"处理文件时发生错误：{e}")
            return None


if __name__ == '__main__':
    pass
