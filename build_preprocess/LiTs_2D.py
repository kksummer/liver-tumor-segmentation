import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from split_dataset import split_data_cls2, split_data_cls3
from os.path import join
import nibabel as nib


class NiiProcessor:
    def __init__(self,
                 root_dir,
                 output_dir,
                 ww=150, wc=70,
                 liver_threshold=100,
                 tumor_threshold=50,
                 target_size=(256, 256)):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.window_width = ww
        self.window_center = wc
        self.liver_threshold = liver_threshold
        self.tumor_threshold = tumor_threshold
        self.total_count_slices = 0
        self.target_size = target_size

    @staticmethod
    def resample_image(image, target_size):
        # 计算重采样因子
        zoom_factor = (1, target_size[0] / image.shape[1], target_size[1] / image.shape[2])
        # 应用重采样
        resampled_image = ndimage.zoom(image, zoom_factor, order=1)  # 双线性插值
        return resampled_image

    @staticmethod
    def windowing(img, window_width, window_center):
        # img 主要是CT图像，window_width和window_center是我们想要的窗宽和窗位
        min_windows = float(window_center) - 0.5 * float(window_width)
        new_img = (img - min_windows) / float(window_width)
        new_img[new_img < 0] = 0  # 二值化处理 抹白
        new_img[new_img > 1] = 1  # 抹黑
        return (new_img * 255).astype('uint8')  # 把数据整理成标准图像格式

    @staticmethod
    def clahe_equalized(img):
        # 输入img的形状必须是3维
        assert (len(img.shape) == 3)
        # 定义均衡化函数
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 新数组存放均衡化后的数据
        img_res = np.zeros_like(img)
        for i in range(len(img)):
            img_res[i, :, :] = clahe.apply(np.array(img[i, :, :], dtype=np.uint8))
        return img_res / 255

    def ct_aug(self, ct):
        # 2. 数据增强
        windowed_ct = self.windowing(ct, self.window_width, self.window_center)
        clahe_ct = self.clahe_equalized(windowed_ct)
        return clahe_ct

    def read_data(self, patient_index):
        ct_path = join(self.root_dir, 'ct', 'volume-{}.nii'.format(patient_index))
        seg_path = join(self.root_dir, 'seg', 'segmentation-{}.nii'.format(patient_index))

        # nib 拿到的是hu值
        ct_hu = nib.load(ct_path).get_fdata().transpose(2, 1, 0)
        seg_hu = nib.load(seg_path).get_fdata().transpose(2, 1, 0)

        # 根据不同病例范围应用不同的翻转操作
        if 0 <= patient_index <= 52:  # [0, 52] 需要上下翻转
            ct_hu = np.flip(ct_hu, axis=1)  # 上下翻转 axis=1代表上下
            seg_hu = np.flip(seg_hu, axis=1)  # 上下翻转
        elif 68 <= patient_index <= 82:  # [68, 82] 需要上下 + 左右翻转
            ct_hu = np.flip(ct_hu, axis=1)  # 上下翻转
            ct_hu = np.flip(ct_hu, axis=2)  # 左右翻转
            seg_hu = np.flip(seg_hu, axis=1)  # 上下翻转
            seg_hu = np.flip(seg_hu, axis=2)  # 左右翻转

        return ct_hu, seg_hu

    def process_ct_liver(self, patient_index):

        # 1. 读取数据
        raw_ct, raw_seg = self.read_data(patient_index)

        # 2. 数据预处理
        # 重采样
        resampled_ct = self.resample_image(raw_ct, self.target_size)
        resampled_seg = self.resample_image(raw_seg, self.target_size)

        # 数据增强
        ct_augmented = self.ct_aug(resampled_ct)

        ct = ct_augmented
        seg = resampled_seg

        # 3. 处理 mask
        # 统计每个切片中肝脏和肿瘤像素的数量  肝脏至少要大于 阈值 肿瘤至少要大于 阈值
        liver_pixel_counts = np.count_nonzero(seg == 1, axis=(1, 2))

        # 获取像素大于阈值的肝脏切片数量
        liver_slices_nums = np.count_nonzero(liver_pixel_counts > self.liver_threshold)

        # 找出满足条件的切片索引
        liver_slice_index = np.nonzero(liver_pixel_counts > self.liver_threshold)[0]

        # 提取这些切片
        patient_slices = ct[liver_slice_index, :, :]
        liver_slices = seg[liver_slice_index, :, :]

        # 二值化
        bin_liver_slices = np.copy(liver_slices)
        bin_liver_slices[bin_liver_slices > 0] = 255

        # 4. 保存
        os.makedirs(join(self.output_dir, 'patient'), exist_ok=True)
        os.makedirs(join(self.output_dir, 'liver'), exist_ok=True)

        total_slices = 0
        for liver_index in range(len(bin_liver_slices)):
            patient_output_path = join(self.output_dir, 'patient', f'{patient_index}_{liver_index}.jpg')
            plt.imsave(patient_output_path, patient_slices[liver_index, :, :], cmap='gray')

            liver_output_path = join(self.output_dir, 'liver', f'{patient_index}_{liver_index}.jpg')
            plt.imsave(liver_output_path, bin_liver_slices[liver_index, :, :], cmap='gray')

            total_slices += 1

        return total_slices

    def process_ct_liver_tumor(self, patient_index):
        # 1. 读取数据
        raw_ct, raw_seg = self.read_data(patient_index)

        # 2. 数据预处理

        # 重采样
        resampled_ct = self.resample_image(raw_ct, self.target_size)
        resampled_seg = self.resample_image(raw_seg, self.target_size)

        # 数据增强
        ct_augmented = self.ct_aug(resampled_ct)

        ct = ct_augmented
        seg = resampled_seg

        # 3. 处理 mask
        # 统计每个切片中肝脏和肿瘤像素的数量  肝脏至少要大于 阈值 肿瘤至少要大于 阈值
        liver_pixel_counts = np.count_nonzero(seg == 1, axis=(1, 2))
        tumor_pixel_counts = np.count_nonzero(seg == 2, axis=(1, 2))

        # 获取像素大于500的肝脏切片数量， 获取像素大于100的肿瘤切片数量
        liver_slices_nums = np.count_nonzero(liver_pixel_counts > self.liver_threshold)
        tumor_slices_nums = np.count_nonzero(tumor_pixel_counts > self.tumor_threshold)

        # 找出满足条件的切片索引
        tumor_slice_index = np.nonzero(np.logical_and(tumor_pixel_counts > self.tumor_threshold,
                                                      liver_pixel_counts > 0))[0]

        # 提取这些切片
        patient_slices = ct[tumor_slice_index, :, :]
        liver_slices = seg[tumor_slice_index, :, :]
        tumor_slices = seg[tumor_slice_index, :, :]

        # 二值化
        bin_liver_slices = np.copy(liver_slices)
        bin_liver_slices[bin_liver_slices > 0] = 255

        bin_tumor_slices = np.copy(tumor_slices)
        bin_tumor_slices[bin_tumor_slices == 2] = 255

        # 4. 保存
        os.makedirs(join(self.output_dir, 'patient'), exist_ok=True)
        os.makedirs(join(self.output_dir, 'liver'), exist_ok=True)
        os.makedirs(join(self.output_dir, 'tumor'), exist_ok=True)

        total_slices = 0
        for tumor_index in range(len(bin_tumor_slices)):
            patient_output_path = join(self.output_dir, 'patient', f'{patient_index}_{tumor_index}.jpg')
            plt.imsave(patient_output_path, patient_slices[tumor_index, :, :], cmap='gray')

            liver_output_path = join(self.output_dir, 'liver', f'{patient_index}_{tumor_index}.jpg')
            plt.imsave(liver_output_path, bin_liver_slices[tumor_index, :, :], cmap='gray')

            tumor_output_path = join(self.output_dir, 'tumor', f'{patient_index}_{tumor_index}.jpg')
            plt.imsave(tumor_output_path, bin_tumor_slices[tumor_index, :, :], cmap='gray')

            total_slices += 1

        return total_slices

    def process_data(self, mode, start, end):
        for patient_id in range(start, end):
            if mode == 'liver':
                total_slices = self.process_ct_liver(patient_id)
            else:
                total_slices = self.process_ct_liver_tumor(patient_id)
                if total_slices is None:
                    continue
            print(f"第 {patient_id} 个病例共有 {total_slices} 张 {mode} 切片")
            self.total_count_slices += total_slices
        print("总共有 {} 张切片".format(self.total_count_slices))


if __name__ == '__main__':
    root = '/home/ian/Project/Datasets/Liver/LiTs_nii'
    output = '/home/ian/Project/Datasets/Liver/LiTs_nii/output'  # 请将此路径更改为您希望存储结果的目录

    processor = NiiProcessor(root, output)
    processor.process_data('tumor', 1, 131)

    # split_data_cls2(output, 'patient', 'liver')
    split_data_cls3(output, 'patient', 'liver', 'tumor')

