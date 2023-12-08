# visualize.py
import matplotlib.pyplot as plt


def show_pred(pred_liver_masks, gt_liver_masks, pred_tumor_masks=None, gt_tumor_masks=None):
    num_samples = pred_liver_masks.shape[0]
    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 4 * num_samples))

    for i in range(num_samples):
        axes[i, 0].imshow(pred_liver_masks[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 0].set_title('Predicted Mask')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_liver_masks[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # 如果需要显示其他掩膜，可以取消注释下面的代码
        # if pred_tumor_masks is not None:
        #     axes[i, 2].imshow(pred_tumor_masks[i].cpu().numpy().squeeze(), cmap='gray')
        #     axes[i, 2].set_title('Predicted Tumor Mask')
        #     axes[i, 2].axis('off')

        # if gt_tumor_masks is not None:
        #     axes[i, 3].imshow(gt_tumor_masks[i].cpu().numpy().squeeze(), cmap='gray')
        #     axes[i, 3].set_title('Ground Truth Tumor Mask')
        #     axes[i, 3].axis('off')

    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(5)
    plt.close('all')


def show_transformed_images(images, masks):
    num_images = images.size(0)
    fig, axes = plt.subplots(num_images, 2, figsize=(12, num_images * 6))

    for i in range(num_images):
        # 显示图片
        axes[i, 0].imshow(images[i].cpu().numpy().transpose(1, 2, 0))
        axes[i, 0].set_title(f"Transformed Image {i + 1}")
        axes[i, 0].axis('off')

        # 显示mask
        axes[i, 1].imshow(masks[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i, 1].set_title(f"Transformed Mask {i + 1}")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
    plt.pause(5)
    plt.close('all')
