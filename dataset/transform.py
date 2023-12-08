import torchvision.transforms.functional as F
import random


class Preprocess:
    def __init__(self, mode, mean=None, std=None):
        self.mean = mean
        self.std = std
        self.mode = mode

    def apply_transform(self, image, mask):
        """ 应用相同的随机变换到图像和掩膜 """
        if self.mode == 'train':
            # 随机水平翻转
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # 随机垂直翻转
            # if random.random() > 0.5:
            #     image = F.vflip(image)
            #     mask = F.vflip(mask)

            # 随机旋转
            angle = random.uniform(-10, 10)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

            # 随机裁剪
            # i, j, h, w = T.RandomCrop.get_params(image, output_size=(256, 256))
            # image = F.crop(image, i, j, h, w)
            # mask = F.crop(mask, i, j, h, w)

            # 中心裁剪
            # crop_size = 256
            # image = F.center_crop(image, crop_size)
            # mask = F.center_crop(mask, crop_size)

            # 调整大小
            # resize_size = 256
            # image = F.resize(image, resize_size)
            # mask = F.resize(mask, resize_size)

        return image, mask

    def __call__(self, image, mask):
        image, mask = self.apply_transform(image, mask)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)
        mask = F.to_tensor(mask)

        return image, mask
