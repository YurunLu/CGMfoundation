import csv
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset

# ===============================
# Utils
# ===============================
def fill_nan_linear(seq: np.ndarray):
    """Fill NaN values by linear interpolation along time axis (1D or 2D)."""
    if seq.ndim == 1:
        s = pd.Series(seq)
        seq_filled = s.interpolate(limit_direction='both').to_numpy()
    else:
        seq_filled = np.zeros_like(seq)
        for i in range(seq.shape[1]):
            s = pd.Series(seq[:, i])
            seq_filled[:, i] = s.interpolate(limit_direction='both').to_numpy()
    return seq_filled


def pad_and_mask(seq, L_target):
    """
    Pad sequence to length L_target, return (padded_seq, mask)
    mask=1 for valid positions, 0 for padded.
    """
    L = seq.shape[0]
    pad_len = max(0, L_target - L)
    if seq.ndim == 1:
        padded = np.pad(seq, (0, pad_len), constant_values=np.nan)
    else:
        padded = np.pad(seq, ((0, pad_len), (0, 0)), constant_values=np.nan)
    mask = np.zeros(L_target, dtype=float)
    mask[:L] = 1.0
    return padded, mask


def load_data_csv(path):
    """
    Load glucose CSV: id, device_id, 0..L-1 columns as glucose.
    Handles variable-length sequences, fills NaN.
    """
    df = pd.read_csv(path)
    device_ids = df['device_id'].astype(str).tolist()
    ids = df['id'].astype(str).tolist()
    glucose_cols = [str(x) for x in range(3456)]
    glucose_values = df[glucose_cols].values.astype(float)
    glucose_values = fill_nan_linear(glucose_values)
    return ids, device_ids, glucose_values


def load_onehot_csv(path, event_columns, L_expected=None):
    """
    OneHot_all.csv structure:
    device_id, TimeIndex, <event columns...>
    Return dict[device_id] = [L, E]
    """
    df = pd.read_csv(path, index_col=0)
    grouped = {}
    E = len(event_columns)
    for device_id, group in df.groupby(df['device_id']):
        times = group['TimeIndex'].astype(int).values
        events = group[event_columns].values.astype(float)
        max_t = times.max()
        L = max_t + 1 if L_expected is None else max(L_expected, max_t + 1)
        arr = np.zeros((L, E), dtype=float)
        arr[times, :] = events
        grouped[str(device_id)] = arr
    return grouped


# ===============================
# Dataset
# ===============================
class EventDataset(Dataset):
    """
    Dataset returning glucose and event patches with forecast targets and masks.
    """
    def __init__(self, data_csv, onehot_csv, event_columns, patch_size=36, stride=6, forecast_steps=12):
        self.ids, self.device_ids, self.glucose = load_data_csv(data_csv)
        self.L = self.glucose.shape[1]
        self.event_columns = event_columns
        self.events_map = load_onehot_csv(onehot_csv, event_columns, L_expected=self.L)
        self.patch_size = patch_size
        self.stride = stride
        self.forecast_steps = forecast_steps

        # per-device patch start indices
        self.device_patch_starts = []
        for idx, dev in enumerate(self.device_ids):
            starts = list(range(0, self.L - (self.patch_size + self.forecast_steps) + 1, self.stride))
            self.device_patch_starts.append(starts)

        self.device_ids = [str(d) for d in self.device_ids]

    def __len__(self):
        return len(self.device_ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        dev = self.device_ids[idx]
        x = self.glucose[idx]  # [L]
        events = self.events_map.get(dev, np.zeros((self.L, len(self.event_columns)), dtype=float))  # [L, E]

        # --- fill NaN values inside sequence ---
        x = fill_nan_linear(x)
        events = fill_nan_linear(events)

        starts = self.device_patch_starts[idx]

        patches, events_patches, targets, masks = [], [], [], []

        for s in starts:
            # Skip if patch exceeds real length
            if s + self.patch_size + self.forecast_steps > len(x):
                continue

            x_patch = x[s:s + self.patch_size].astype(np.float32)
            ev_patch = events[s:s + self.patch_size + self.forecast_steps, :].astype(np.float32)
            y = x[s + self.patch_size:s + self.patch_size + self.forecast_steps].astype(np.float32)

            # mask invalid/padded values in target
            mask = ~np.isnan(y)
            y[np.isnan(y)] = 0.0

            patches.append(x_patch)
            events_patches.append(ev_patch)
            targets.append(y)
            masks.append(mask.astype(np.float32))

        patches = np.stack(patches, axis=0)
        events_patches = np.stack(events_patches, axis=0)
        targets = np.stack(targets, axis=0)
        masks = np.stack(masks, axis=0)

        return {
            'id': gid,
            'device_id': dev,
            'patches': patches,
            'events': events_patches,
            'targets': targets,
            'masks': masks
        }
    def subset_by_device_indices(self, idx_iter):
        """
        Create a lightweight subset dataset that only includes
        specific device indices (for train/val split).
        """
        indices = list(idx_iter)
        return _SubsetEventDataset(self, indices)



class _SubsetEventDataset(Dataset):
    def __init__(self, parent: EventDataset, indices):
        self.parent = parent
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.parent[self.indices[index]]


# ===============================
# Collate + Masked Loss
# ===============================
def collate_patches(batch):
    """Collate batch of samples with padding."""
    B = len(batch)
    Ns = [item['patches'].shape[0] for item in batch]
    N_max = max(Ns)
    P = batch[0]['patches'].shape[1]
    F = batch[0]['targets'].shape[1]
    E = batch[0]['events'].shape[2]

    patches_b = np.zeros((B, N_max, P), dtype=np.float32)
    events_b = np.zeros((B, N_max, P + F, E), dtype=np.float32)
    targets_b = np.zeros((B, N_max, F), dtype=np.float32)
    mask_b = np.zeros((B, N_max, F), dtype=np.float32)

    for i, item in enumerate(batch):
        n = item['patches'].shape[0]
        patches_b[i, :n, :] = item['patches']
        events_b[i, :n, :, :] = item['events']
        targets_b[i, :n, :] = item['targets']
        mask_b[i, :n, :] = item['masks']

    return (
        torch.from_numpy(patches_b),
        torch.from_numpy(events_b),
        torch.from_numpy(targets_b),
        torch.from_numpy(mask_b)
    )

