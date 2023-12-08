
import os
import numpy as np
from PIL import Image


def compute_mean_std(data_dir, image_name, mask_name, mode):
    if data_dir is None:
        raise ValueError("数据目录为空")
    image_channels = 3
    img_dir = os.path.join(data_dir, mode, image_name)
    roi_dir = os.path.join(data_dir, mode, mask_name)
    assert os.path.exists(img_dir), fr"image dir: '{img_dir}' does not exist."
    assert os.path.exists(roi_dir), fr"roi dir: '{roi_dir}' does not exist."

    img_name_list = [i for i in os.listdir(img_dir) if i.endswith(".jpg") or i.endswith("png") or i.endswith("tif")]
    cumulative_mean = np.zeros(image_channels)
    cumulative_std = np.zeros(image_channels)
    for img_name in img_name_list:
        img_path = os.path.join(img_dir, img_name)
        ori_path = os.path.join(roi_dir, img_name)
        img = np.array(Image.open(img_path).convert('RGB')) / 255.
        roi_img = np.array(Image.open(ori_path).convert('L'))

        img = img[roi_img == 255]
        cumulative_mean += img.mean(axis=0)
        cumulative_std += img.std(axis=0)

    mean = cumulative_mean / len(img_name_list)
    std = cumulative_std / len(img_name_list)
    return mean, std