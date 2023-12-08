
from torch.utils.data import DataLoader
from configs.get_config import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2


def revert_roi(loader, model, roi_folder, device):
    for idx, (images, _, org_images) in enumerate(tqdm(loader)):
        images = images.to(device)
        # 前向传播得到liver mask预测
        with torch.no_grad():
            liver_pred = model(images)
            liver_pred = (liver_pred > 0.5).float()

        # 先进行颜色反转
        liver_pred_np = liver_pred.cpu().numpy().squeeze().astype(np.float32)
        liver_pred_np = np.stack([liver_pred_np] * 3, axis=-1)

        # 获取并翻转原始图像的numpy数组
        org_image_np = np.array(org_images[0])
        inverted_image_np = 255 - org_image_np

        # 然后与 liver_pred 进行“与”操作
        liver_roi_np = (inverted_image_np * liver_pred_np).astype(np.uint8)

        # 使用原始文件名来保存结果
        filename = loader.dataset.image_filenames[idx]
        liver_output_path = os.path.join(roi_folder, filename)
        cv2.imwrite(liver_output_path, liver_roi_np)


def add_pred_mask(loader, model, pred_folder, device):
    for idx, (images, masks, org_images) in enumerate(tqdm(loader)):
        images = images.to(device)

        # 前向传播得到liver mask预测
        with torch.no_grad():
            liver_pred = model(images)
            liver_pred = (liver_pred > 0.5).float()

        # 将预测的mask转换为numpy数组
        liver_pred_np = liver_pred.cpu().numpy().squeeze().astype(np.float32)

        # 获取原始图像的numpy数组
        org_image_np = np.array(org_images[0])
        # 检查是否需要resize

        # # 根据预测结果生成一个蓝色遮罩
        # blue_mask = np.zeros((liver_pred_np.shape[0], liver_pred_np.shape[1], 3), dtype=np.float32)
        # blue_mask[:, :, 2] = liver_pred_np * 255  # 设置蓝色通道

        # # 根据预测结果生成一个绿色遮罩
        # color_mask = np.zeros((liver_pred_np.shape[0], liver_pred_np.shape[1], 3), dtype=np.float32)
        # color_mask[:, :, 1] = liver_pred_np * 255  # 设置绿色通道

        # color_mask = np.zeros((liver_pred_np.shape[0], liver_pred_np.shape[1], 3), dtype=np.float32)
        color_mask = np.zeros_like(org_image_np, dtype=np.float32)
        color_mask[:, :, 0] = liver_pred_np * 255  # 设置红色通道
        color_mask[:, :, 1] = liver_pred_np * 255  # 设置绿色通道

        # color_mask = np.zeros((liver_pred_np.shape[0], liver_pred_np.shape[1], 3), dtype=np.float32)
        # color_mask[:, :, 1] = liver_pred_np * 255  # 设置红色通道
        # color_mask[:, :, 2] = liver_pred_np * 255  # 设置蓝色通道

        # 使用alpha混合来调整蓝色遮罩的不透明度
        alpha = 0.7  # 调整这个值来改变不透明度，范围是[0,1]
        # merged_image_np = org_image_np * (1 - alpha) + color_mask * alpha * liver_pred_resized[..., None]
        merged_image_np = org_image_np * (1 - alpha * liver_pred_np[..., None]) + color_mask * alpha

        merged_image_np = np.clip(merged_image_np, 0, 255).astype(np.uint8)

        # 使用真实掩码（ground truth mask）绘制轮廓
        gt_mask_np = np.array(masks[0]).squeeze().astype(np.uint8)
        contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(merged_image_np, contours, -1, (255, 0, 0), 2)

        # 使用原始文件名来保存结果
        filename = loader.dataset.image_filenames[idx]
        liver_output_path = os.path.join(pred_folder, filename)
        # cv2.imwrite(liver_output_path, merged_image_np)  # 使用OpenCV直接保存图像
        plt.imsave(liver_output_path, merged_image_np)


def pred_mask(loader, model, pred_folder, device):
    for idx, (images, _, _) in enumerate(tqdm(loader)):
        images = images.to(device)
        # 前向传播得到liver mask预测
        with torch.no_grad():
            liver_pred = model(images)
            liver_pred = (liver_pred > 0.5).float()
        liver_pred_np = liver_pred.cpu().numpy().squeeze().astype(np.float32)
        # invert_image = 255 - liver_pred_np
        # 使用原始文件名来保存结果
        filename = loader.dataset.image_filenames[idx]
        liver_output_path = os.path.join(pred_folder, filename)
        plt.imsave(liver_output_path, liver_pred_np, cmap='gray')


def prediction(config):
    # 1. 加载模型和权重
    device = torch.device(config.runtime.device)
    model = config.construct_model()
    model = config.load_model_weights(model)
    model.to(device)
    model.eval()

    model_type = config.model_config['type']
    dataset_path = config.dataset_config['root_dir']
    dataset_type = config.dataset_config['type']
    mask_name = config.dataset_config['mask_name']
    pred_mode = config.runtime_config['pred_mode']

    # 2. 数据加载
    modes = ['train', 'val']
    for mode in modes:
        print("正在处理 {} 数据集".format(mode))

        dataset = config.construct_dataset(mode=mode)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

        output_path = os.path.join(dataset_path, mode)
        output_folder = os.path.join(output_path,
                                     'pred' + "-" +
                                     dataset_type + "-" +
                                     mask_name + "-" +
                                     pred_mode + "-" +
                                     model_type)
        os.makedirs(output_folder, exist_ok=True)
        if pred_mode == 'contour':
            add_pred_mask(loader, model, output_folder, device)
        elif pred_mode == 'roi':
            revert_roi(loader, model, output_folder, device)
        elif pred_mode == 'mask':
            pred_mask(loader, model, output_folder, device)

        print("{}处理完成!生成{}, 路径{}".format(mode, pred_mode, output_folder))


if __name__ == '__main__':
    pass
