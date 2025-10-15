import torch
import torch.nn as nn


def masked_mse_loss(preds, targets, mask):
    # preds/targets: [B, N, F], mask: [B, N, F] with 1 for valid
    diff = (preds - targets) * mask
    mse = (diff ** 2).sum() / (mask.sum() + 1e-8)
    return mse

def masked_mae_loss(pred, target, mask):
    """
    Mean Absolute Error ignoring masked (0) positions.
    pred, target, mask: [B, N, F]
    """
    diff = torch.abs(pred - target)
    diff = diff * mask
    return diff.sum() / (mask.sum() + 1e-6)

class LossTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum_loss = 0.0
        self.count = 0

    def update(self, loss, n=1):
        self.sum_loss += loss * n
        self.count += n

    def average(self):
        return self.sum_loss / max(self.count, 1)