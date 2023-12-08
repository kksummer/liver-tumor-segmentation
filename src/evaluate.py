
from torch.utils.data import DataLoader
from src.train import validate
from utils.metrics import *


def evaluation(config):
    device = torch.device(config.runtime.device)
    is_visual = config.runtime.visual
    model = config.construct_model()
    loss_fn = config.construct_loss()
    model = config.load_model_weights(model)
    model.to(device)
    model.eval()
    val_dataset = config.construct_dataset(mode='val')
    val_loader = DataLoader(val_dataset, batch_size=config.runtime.batch_size,
                            num_workers=config.runtime.num_workers, shuffle=False)
    dice, hd95, jac_iou, precision, recall, voe, rvd = validate(model, loss_fn, val_loader, device,
                                                                metrics_all=True,
                                                                is_visual=is_visual)

    print("Average Result")
    print("Dice: {:.4f}, HD95: {:.4f}, Jaccard_IoU: {:.4f}, "
          "Precision: {:.4f}, Recall: {:.4f}, "
          "VOE: {:.4f}, RVD: {:.4f}"
          .format(dice, hd95, jac_iou, precision, recall, voe, rvd))


if __name__ == "__main__":
    print("预测完成！")
