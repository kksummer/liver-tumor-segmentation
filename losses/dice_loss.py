import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, eps=1.0):
        """
        Initializes the Dice Loss with an optional epsilon value.

        Args:
            eps (float, optional): The epsilon value to avoid division by zero. Default is 1.0.
        """
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        Computes the Dice Loss between the input and target.

        Args:
            pred (torch.Tensor): The predicted values.
            target (torch.Tensor): The ground truth values.

        Returns:
            torch.Tensor: The computed Dice Loss.
        """
        # Ensure the input and target have the same shape
        assert pred.size() == target.size(), "Input and Target must have the same shape"

        # Flatten the input and target tensors
        input_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        # Compute the Dice coefficient
        intersection = (input_flat * target_flat).sum()
        union = input_flat.sum() + target_flat.sum()
        dice_coef = (2.0 * intersection + self.eps) / (union + self.eps)

        # Compute the Dice Loss
        loss = 1.0 - dice_coef
        return loss





