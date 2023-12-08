import torch
from losses.dice_loss import DiceLoss


class HybridLoss(torch.nn.Module):
    def __init__(self, hybrid_rate=0.5):
        super(HybridLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.hybrid_rate = hybrid_rate

    def forward(self, pred, target):
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        loss = self.hybrid_rate * dice_loss + (1 - self.hybrid_rate) * bce_loss
        return loss
