import torch.nn as nn
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(_Loss):
    """
    Convenience implementation of CrossEntropyLoss.
    """

    def __init__(self, to_onehot_y=False, ce_params=None):
        """
        Initializes the CELoss class.

        Args:
            to_onehot_y: whether to convert the ``target`` into the one-hot format,
                using the number of classes inferred from `pred` (``pred.shape[1]``). Defaults to False.
            ce_params (dict, optional): Parameters for the CrossEntropyLoss.
        """
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.ce_loss = nn.CrossEntropyLoss(
            **(ce_params if ce_params is not None else {})
        )

    def forward(self, y_pred, y_true):
        """
        Forward pass for calculating the CrossEntropy loss.

        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Ground truth labels.

        Returns:
            The CrossEntropy loss.
        """
        if self.to_onehot_y:
            y_true = one_hot(y_true, num_classes=y_pred.shape[1])
        return self.ce_loss(y_pred, y_true)
