import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from sklearn.metrics import jaccard_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries


def compute_surface(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    mask_boundaries = find_boundaries(mask)
    return np.array([prop.centroid for prop in props if mask_boundaries[tuple(map(int, prop.centroid))]])


def compute_dice(predicted, target, smooth=1e-5):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def compute_iou(predicted, target, smooth=1e-5):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum() - intersection

    iou_score = (intersection + smooth) / (union + smooth)
    return iou_score.item()


def compute_hd95(predicted, target):
    # 将预测和目标掩膜转换为二值图像（0或1）
    predicted = predicted.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    # 计算豪斯多夫距离
    hausdorff_forward = directed_hausdorff(predicted, target)[0]
    hausdorff_backward = directed_hausdorff(target, predicted)[0]
    dist = max(hausdorff_forward, hausdorff_backward)
    return dist


def compute_jaccard_voe(predicted, target):
    jaccard = jaccard_score(target.flatten().cpu().numpy(), predicted.flatten().cpu().numpy())
    voe = 1 - jaccard
    return jaccard, voe


def compute_precision_recall(predicted, target):
    precision = precision_score(target.flatten().cpu().numpy(), predicted.flatten().cpu().numpy())
    recall = recall_score(target.flatten().cpu().numpy(), predicted.flatten().cpu().numpy())
    return precision, recall


def compute_roc_curve(predicted_probs, target):
    fpr, tpr, _ = roc_curve(target.flatten().cpu().numpy(), predicted_probs.flatten().cpu().numpy())
    return fpr, tpr, auc(fpr, tpr)


def compute_pr_curve(predicted_probs, target):
    precision, recall, _ = precision_recall_curve(target.flatten().cpu().numpy(), predicted_probs.flatten().cpu().numpy())
    average_precision = auc(recall, precision)  # Note: Use recall as the x-axis for AUC computation for PR curve
    return precision, recall, average_precision


def compute_rvd(predicted, target):
    volume_pred = torch.sum(predicted).item()
    volume_gt = torch.sum(target).item()
    return (volume_pred - volume_gt) / volume_gt


def compute_asd(predicted, target):
    # 移除维度大小为1的维度
    predicted = predicted.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()

    # 如果预测或目标为空，直接返回inf
    if np.sum(predicted) == 0 or np.sum(target) == 0:
        return float('inf')

    # 提取预测和实际掩码的表面体素
    surface_pred = compute_surface(predicted)
    surface_gt = compute_surface(target)

    # 如果预测或目标的表面为空，直接返回inf
    if surface_pred.shape[0] == 0 or surface_gt.shape[0] == 0:
        return float('inf')

    # 计算距离
    distances_pred_to_gt = distance.cdist(surface_pred, surface_gt, 'euclidean')
    distances_gt_to_pred = distance.cdist(surface_gt, surface_pred, 'euclidean')

    asd = (distances_pred_to_gt.mean() + distances_gt_to_pred.mean()) / 2.0
    return asd


