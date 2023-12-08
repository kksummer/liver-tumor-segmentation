import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
from split_dataset import split_data_cls2, split_data_cls3
from os.path import join


class DicomProcessor:
    def __init__(self,
                 root_dir,
                 output_dir,
                 ww=100, wc=70,
                 liver_threshold=0,
                 tumor_threshold=0,
                 target_size=(512, 512)):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.window_width = ww
        self.window_center = wc
        self.liver_threshold = liver_threshold
        self.tumor_threshold = tumor_threshold
        self.total_count_slices = 0
        self.target_size = target_size

    @staticmethod
    def read_slices(path):
        slices = [pydicom.dcmread(join(path, image_num)) for image_num in os.listdir(path)]
        slices.sort(key=lambda x: int(x.InstanceNumber))
        slices_np = np.array([i.pixel_array for i in slices])
        return slices_np

    def read_data(self, patient_index):
        patient_path = join(self.root_dir, f'3Dircadb1.{patient_index}', 'PATIENT_DICOM')
        liver_path = join(self.root_dir, f'3Dircadb1.{patient_index}', 'MASKS_DICOM', 'liver')

        liver_slices = self.read_slices(liver_path)
        patient_slices = self.read_slices(patient_path)

        liver_slices_np = np.array(liver_slices)
        patient_slices_np = np.array(patient_slices)
        return patient_slices_np, liver_slices_np

    def read_tumor(self, patient_index):
        patient_slices, _ = self.read_data(patient_index)
        masks_path = join(self.root_dir, f'3Dircadb1.{patient_index}', 'MASKS_DICOM')
        masks_list = os.listdir(masks_path)

        if any('livertumor' in mask for mask in masks_list):
            tumor_paths = [join(masks_path, liver_tumor_path) for liver_tumor_path in masks_list if
                           'livertumor' in liver_tumor_path]
            # 创建一个全零数组来存储所有肿瘤的组合。
            combined_tumor_array = np.zeros((patient_slices.shape[0],
                                             patient_slices.shape[1],
                                             patient_slices.shape[2]), dtype=np.uint16)

            # 遍历每个肿瘤数据，统计并合并所有肿瘤数据。
            for tumor_index, tumor in enumerate(tumor_paths):
                # 读取并排序肿瘤的DICOM数据
                single_tumor_slices = [pydicom.dcmread(os.path.join(tumor, image_num)) for image_num in
                                       os.listdir(tumor)]
                single_tumor_slices.sort(key=lambda x: int(x.InstanceNumber))
                single_tumor_array = np.array([i.pixel_array for i in single_tumor_slices])

                # 使用bitwise_or将当前tumor叠加到combined_tumor_array上 和 patient_slices 相同大小的全零数组，在肿瘤位置上有值。
                combined_tumor_array = np.bitwise_or(combined_tumor_array, single_tumor_array)
            return combined_tumor_array
        return None

    @staticmethod
    def resample(image, target_size):
        zoom_factor = (1, target_size[0] / image.shape[1], target_size[1] / image.shape[2])
        resampled_image = ndimage.zoom(image, zoom_factor, order=1)
        return resampled_image

    @staticmethod
    def windowing(image, window_width, window_center):
        # img 主要是CT图像，window_width和window_center是我们想要的窗宽和窗位
        min_windows = float(window_center) - 0.5 * float(window_width)
        new_img = (image - min_windows) / float(window_width)
        new_img[new_img < 0] = 0  # 二值化处理 抹白
        new_img[new_img > 1] = 1  # 抹黑
        return (new_img * 255).astype('uint8')  # 把数据整理成标准图像格式

    @staticmethod
    def clahe_equalized(image):
        # 输入img的形状必须是3维
        assert (len(image.shape) == 3)
        # 定义均衡化函数
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # 新数组存放均衡化后的数据
        image_res = np.zeros_like(image)
        for i in range(len(image)):
            image_res[i, :, :] = clahe.apply(np.array(image[i, :, :], dtype=np.uint8))
        return image_res

    def process_ct_liver(self, patient_index):
        # 1. 读取数据
        raw_patient_slices, raw_liver_slices = self.read_data(patient_index)

        # 2. 数据预处理
        # 重采样
        resampled_patient_slices = self.resample(raw_patient_slices, self.target_size)
        resampled_liver_slices = self.resample(raw_liver_slices, self.target_size)

        # 3. 数据增强
        windowed_patient_slices = self.windowing(resampled_patient_slices, self.window_width, self.window_center)
        clahe_patient_slices = self.clahe_equalized(windowed_patient_slices)

        patient_slices = clahe_patient_slices
        liver_slices = resampled_liver_slices

        # 4. 处理 mask
        liver_pixel_counts = np.count_nonzero(liver_slices > 0, axis=(1, 2))
        liver_slices_threshold_index = np.nonzero(liver_pixel_counts > self.liver_threshold)[0]

        # 提取切片
        patient_slices_threshold = patient_slices[liver_slices_threshold_index, :, :]
        liver_slices_threshold = liver_slices[liver_slices_threshold_index, :, :]

        # 二值化
        bin_liver_slices_threshold = np.copy(liver_slices_threshold)
        bin_liver_slices_threshold[bin_liver_slices_threshold > 0] = 255

        # 5. 保存
        os.makedirs(join(self.output_dir, 'patient'), exist_ok=True)
        os.makedirs(join(self.output_dir, 'liver'), exist_ok=True)

        total_slices = 0
        for liver_index in range(len(bin_liver_slices_threshold)):
            patient_output_path = join(self.output_dir, 'patient', f'{patient_index}_{liver_index}.jpg')
            plt.imsave(patient_output_path, patient_slices_threshold[liver_index, :, :], cmap='gray')

            liver_output_path = join(self.output_dir, 'liver', f'{patient_index}_{liver_index}.jpg')
            plt.imsave(liver_output_path, bin_liver_slices_threshold[liver_index, :, :], cmap='gray')

            total_slices += 1

        return total_slices

    def process_ct_liver_tumor(self, patient_index):

        # 1. 读取数据
        raw_patient_slices, raw_liver_slices = self.read_data(patient_index)
        raw_tumor_slices = self.read_tumor(patient_index)

        if raw_tumor_slices is None:
            print(f"第 {patient_index} 个病例没有肿瘤数据")
            return None

        # 2. 数据预处理
        # 重采样
        resampled_patient_slices = self.resample(raw_patient_slices, self.target_size)
        resampled_liver_slices = self.resample(raw_liver_slices, self.target_size)
        resampled_tumor_slices = self.resample(raw_tumor_slices, self.target_size)

        # 3. 数据增强
        windowed_patient_slices = self.windowing(resampled_patient_slices, self.window_width, self.window_center)
        clahe_patient_slices = self.clahe_equalized(windowed_patient_slices)

        patient_slices = clahe_patient_slices
        liver_slices = resampled_liver_slices
        tumor_slices = resampled_tumor_slices

        # 4.
        liver_pixel_counts = np.count_nonzero(liver_slices > 0, axis=(1, 2))
        tumor_pixel_counts = np.count_nonzero(tumor_slices > 0, axis=(1, 2))
        tumor_slices_threshold_index = np.nonzero(
            np.logical_and(tumor_pixel_counts > self.tumor_threshold, liver_pixel_counts > 0)
        )[0]
        # 提取切片
        patient_slices_threshold = patient_slices[tumor_slices_threshold_index, :, :]
        liver_slices_threshold = liver_slices[tumor_slices_threshold_index, :, :]
        tumor_slices_threshold = tumor_slices[tumor_slices_threshold_index, :, :]

        # 二值化
        bin_liver_slices_threshold = np.copy(liver_slices_threshold)
        bin_liver_slices_threshold[bin_liver_slices_threshold > 0] = 255

        bin_tumor_slices_threshold = np.copy(tumor_slices_threshold)
        bin_tumor_slices_threshold[bin_tumor_slices_threshold == 255] = 255

        # 5. 保存
        os.makedirs(join(self.output_dir, 'patient'), exist_ok=True)
        os.makedirs(join(self.output_dir, 'liver'), exist_ok=True)
        os.makedirs(join(self.output_dir, 'tumor'), exist_ok=True)

        total_slices = 0
        for tumor_index in range(len(bin_tumor_slices_threshold)):

            patient_output_path = join(self.output_dir, 'patient', f'{patient_index}_{tumor_index}.jpg')
            plt.imsave(patient_output_path, patient_slices_threshold[tumor_index, :, :], cmap='gray')

            liver_output_path = join(self.output_dir, 'liver', f'{patient_index}_{tumor_index}.jpg')
            plt.imsave(liver_output_path, bin_liver_slices_threshold[tumor_index, :, :], cmap='gray')

            tumor_output_path = join(self.output_dir, 'tumor', f'{patient_index}_{tumor_index}.jpg')
            plt.imsave(tumor_output_path, bin_tumor_slices_threshold[tumor_index, :, :], cmap='gray')

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
    root = '/home/ian/Project/Datasets/Liver/IRCAD_RAW/1'
    output = '/home/ian/Project/Datasets/Liver/Ircad'  # 请将此路径更改为您希望存储结果的目录

    processor = DicomProcessor(root, output)
    processor.process_data('liver', 1, 21)

    split_data_cls2(output, 'patient', 'liver')
    # split_data_cls3(output, 'patient', 'liver', 'tumor')


