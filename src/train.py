from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import *
from utils.visualize import show_pred


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, run_device):
    device = run_device
    model.train()
    running_loss = 0.0
    batch_iter = tqdm(train_loader, desc="Iteration", leave=False)
    for images, masks, _ in batch_iter:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        # 4, 1, 512, 512 4, 1, 512, 512
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_iter.set_postfix(epoch=int(epoch + 1), loss=loss.item())
    average_loss = running_loss / len(train_loader)
    return average_loss


def train(config):
    device = torch.device(config.runtime.device)
    mode = 'train'

    model = config.construct_model()
    loss_fn = config.construct_loss()
    optimizer = config.construct_optimizer(model.parameters())
    lr_scheduler = config.construct_lr_scheduler(optimizer)
    early_stopping = config.construct_early_stop()

    bs = config.runtime.batch_size
    workers = config.runtime.num_workers
    epochs = config.runtime.num_epochs

    log_info_list = []
    config.save_info(log_info_list, mode='start')

    train_dataset = config.construct_dataset(mode=mode)
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=workers, shuffle=True)

    val_dataset = config.construct_dataset(mode='val')
    val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=workers, shuffle=False)

    epoch_iter = tqdm(range(epochs), desc="Epoch", leave=True)

    best_dice = 0.0
    model.to(device)
    for epoch in epoch_iter:
        train_loss_info = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
        if lr_scheduler is not None:
            lr_scheduler.step()
        val_loss_info, dice_score = validate(model, loss_fn, val_loader, device, metrics_all=False)

        if dice_score > best_dice:
            best_dice = dice_score
            config.save_best_model(model, epoch)

        epoch_iter.set_postfix(epoch=int(epoch + 1),
                               train_loss=train_loss_info,
                               val_loss=val_loss_info,
                               best_dice=best_dice)

        log_info = {
            'epoch': epoch + 1,
            'train_loss': round(train_loss_info, 4),
            'val_loss': round(val_loss_info, 4),
            'dice_score': round(dice_score, 4),
        }
        epoch_iter.refresh()
        log_info_list.append(log_info)
        config.save_info(log_info_list, mode='train')

        if early_stopping(-dice_score):  # 负dice
            print("早停触发!")
            print("best dice:", round(best_dice, 4))
            break


def validate(model, criterion, val_loader, device, metrics_all=False, is_visual=False):
    model.eval()

    # Initialize the metrics variables
    total_dice = total_hd95 = total_jaccard_iou = 0.0
    total_voe = total_rvd = total_precision = total_recall = 0.0
    num_samples = 0
    running_loss = 0.0

    with torch.no_grad():
        batch_iter = tqdm(val_loader, desc="Validation", leave=False)
        for images, masks, _ in batch_iter:
            batch_size = images.size(0)
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            if is_visual:
                show_pred(outputs, masks)
            for i, (current_predicted_mask, current_mask) in enumerate(zip(outputs, masks)):
                current_predicted_mask = (current_predicted_mask > 0.5).float()
                current_mask = (current_mask > 0.5).float()
                dice = compute_dice(current_predicted_mask, current_mask)
                total_dice += dice

                if metrics_all:
                    # 所有的指标
                    current_jaccard_iou, current_voe = compute_jaccard_voe(current_predicted_mask, current_mask)
                    current_hd95 = compute_hd95(current_predicted_mask, current_mask)
                    current_rvd = compute_rvd(current_predicted_mask, current_mask)
                    current_precision, current_recall = compute_precision_recall(current_predicted_mask, current_mask)

                    # 累加指标
                    total_hd95 += current_hd95
                    total_jaccard_iou += current_jaccard_iou
                    total_voe += current_voe
                    total_rvd += current_rvd
                    total_precision += current_precision
                    total_recall += current_recall

                    print("dice: {:.4f}, jaccadrd/iou: {:.4f}, hd95: {:.4f}, "
                          "voe: {:.4f}, rvd: {:.4f}, precision: {:.4f}, "
                          "recall: {:.4f}"
                          .format(dice, current_jaccard_iou, current_hd95,
                                  current_voe, current_rvd, current_precision,
                                  current_recall))

            num_samples += batch_size

        # 一个批次的指标
        dice_score = total_dice / num_samples
        avg_loss = running_loss / len(val_loader)

        if metrics_all:
            # 计算所有指标
            hd95 = total_hd95 / num_samples
            jaccard_iou = total_jaccard_iou / num_samples
            precision = total_precision / num_samples
            recall = total_recall / num_samples
            voe = total_voe / num_samples
            rvd = total_rvd / num_samples
            return dice_score, hd95, jaccard_iou, precision, recall, voe, rvd
        else:
            # 只返回dice和loss
            return avg_loss, dice_score


