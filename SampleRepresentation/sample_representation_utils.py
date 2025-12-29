# sample_representation_utils.py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.data_utils import CGMForecastDataset


class CGMSampleContrastiveDataset(Dataset):
    """
    Two-view dataset:
      view1 = days 1–6
      view2 = days 7–12
    """
    def __init__(
        self,
        data,
        patch_size=36,
        stride=6,
        steps_per_day=288,
        split_day=6,
    ):
        self.base = CGMForecastDataset(
            data,
            patch_size=patch_size,
            stride=stride,
            forecast_steps=1,
        )

        starts = torch.tensor(self.base.starts)
        day_idx = starts // steps_per_day

        self.view1_mask = day_idx < split_day
        self.view2_mask = day_idx >= split_day

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        patches, patch_mask, _, _ = self.base[idx]

        return {
            "patches": patches,
            "patch_mask": patch_mask,
            "view1_mask": self.view1_mask.clone(),
            "view2_mask": self.view2_mask.clone(),
        }


def info_nce_loss(z1, z2, temperature=0.1):
    """
    Standard symmetric InfoNCE
    """
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = F.cross_entropy(logits, labels)
    loss += F.cross_entropy(logits.T, labels)
    return 0.5 * loss
