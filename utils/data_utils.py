# utils/data_utils.py
import numpy as np
import torch
from torch.utils.data import Dataset

def fill_nan_lin(x: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of NaNs. If all NaN, returns zeros.
    """
    x = x.copy().astype(np.float32)
    if np.isnan(x).all():
        return np.zeros_like(x, dtype=np.float32)
    isnan = np.isnan(x)
    if not isnan.any():
        return x
    idx = np.arange(len(x))
    good = ~isnan
    # if only single valid point, fill with that value
    if good.sum() == 1:
        x[isnan] = float(x[good][0])
        return x
    x[isnan] = np.interp(idx[isnan], idx[good], x[good])
    return x

class CGMForecastDataset(Dataset):
    """
    Overlapping-patch dataset for forecasting.

    For each sequence (length L) we construct patch windows starting at:
        starts = 0, stride, 2*stride, ..., up to start_max

    For each start:
        patch = seq[start : start + patch_size]
        target = seq[start + patch_size : start + patch_size + forecast_steps]

    We only create starts such that both patch and target are within sequence length:
        start + patch_size + forecast_steps <= seq_len

    Returned per sample:
        patches:        torch.FloatTensor [N, patch_size]
        patch_mask:     torch.BoolTensor  [N]      (True if patch fully in valid region)
        targets:        torch.FloatTensor [N, forecast_steps]
        forecast_mask:  torch.BoolTensor  [N]      (True if target fully in valid region)
    """
    def __init__(self, data, patch_size=36, stride=12, forecast_steps=None):
        """
        data: numpy array shape [num_samples, seq_len] or list-like
        patch_size: length of patch (P)
        stride: hop between starts
        forecast_steps: length of target block. If None, defaults to patch_size.
        """
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("data must be 2D array-like: [num_samples, seq_len]")
        self.data = arr
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.forecast_steps = int(forecast_steps) if forecast_steps is not None else int(patch_size)
        self.seq_len = arr.shape[1]

        # compute starts so patch and target are within sequence length
        max_start = self.seq_len - (self.patch_size + self.forecast_steps)
        if max_start < 0:
            # no possible start for any sample
            self.starts = []
        else:
            self.starts = list(range(0, max_start + 1, self.stride))
        self.num_patches = len(self.starts)

        # build samples (we keep one sample per original sequence)
        self.samples = []
        for seq in self.data:
            # compute L_real (first trailing NaN index) using original sequence
            seq_orig = np.asarray(seq, dtype=np.float32)
            if np.isnan(seq_orig).all():
                # skip entirely-NaN sequences
                self.samples.append(None)  # sentinel for empty sample; kept to preserve indexing
                continue

            L_real = (np.argmax(np.isnan(seq_orig)) if np.isnan(seq_orig).any() else len(seq_orig))
            s = fill_nan_lin(seq_orig)  # interpolated for numeric arrays

            # gather patches and targets using global starts (ensures same N for all samples)
            patches = []
            patch_mask = []
            targets = []
            target_mask = []

            for start in self.starts:
                # slice patch and target (both will be within seq bounds by construction of starts)
                p = s[start: start + self.patch_size].astype(np.float32)
                t = s[start + self.patch_size: start + self.patch_size + self.forecast_steps].astype(np.float32)

                # validity: fully inside the real (non-trailing-NaN) region
                valid_patch = (start + self.patch_size) <= L_real
                valid_target = (start + self.patch_size + self.forecast_steps) <= L_real

                patches.append(p)
                patch_mask.append(valid_patch)
                targets.append(t)
                target_mask.append(valid_target)

            # convert to arrays (N may be zero)
            if len(patches) == 0:
                self.samples.append(None)
            else:
                patches = np.stack(patches, axis=0).astype(np.float32)      # [N, P]
                targets = np.stack(targets, axis=0).astype(np.float32)      # [N, F]
                patch_mask = np.array(patch_mask, dtype=bool)               # [N]
                target_mask = np.array(target_mask, dtype=bool)             # [N]
                self.samples.append((patches, patch_mask, targets, target_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if item is None:
            # return empty tensors for consistency (caller should batch-skip or handle)
            return (torch.zeros((0, self.patch_size), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.bool),
                    torch.zeros((0, self.forecast_steps), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.bool))
        patches, patch_mask, targets, target_mask = item
        return (torch.tensor(patches, dtype=torch.float32),
                torch.tensor(patch_mask, dtype=torch.bool),
                torch.tensor(targets, dtype=torch.float32),
                torch.tensor(target_mask, dtype=torch.bool))
