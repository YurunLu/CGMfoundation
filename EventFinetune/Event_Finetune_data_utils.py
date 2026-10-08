import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def is_binary_column(arr: np.ndarray):
    vals = arr[~np.isnan(arr)]
    if vals.size == 0:
        return True
    vals = np.unique(np.round(vals.astype(float), 6))
    return np.all(np.isin(vals, [0.0, 1.0]))


def compute_nonzero_mean_std(arrays):
    vals = []
    for arr in arrays:
        x = arr[~np.isnan(arr)]
        x = x[x != 0]
        if x.size > 0:
            vals.append(x)
    if not vals:
        return 0.0, 1.0
    vals = np.concatenate(vals)
    mean = float(vals.mean())
    std = float(vals.std())
    if std == 0.0 or np.isnan(std):
        std = 1.0
    return mean, std


def fill_nan_linear(seq: np.ndarray, keep_edges: bool = False):
    """Fill NaN values by linear interpolation along time axis (1D or 2D)."""
    if seq.ndim == 1:
        s = pd.Series(seq)
        if keep_edges:
            seq_filled = s.interpolate(limit_area="inside").to_numpy()
        else:
            seq_filled = s.interpolate(limit_direction="both").to_numpy()
    else:
        seq_filled = np.zeros_like(seq)
        for i in range(seq.shape[1]):
            s = pd.Series(seq[:, i])
            if keep_edges:
                seq_filled[:, i] = s.interpolate(limit_area="inside").to_numpy()
            else:
                seq_filled[:, i] = s.interpolate(limit_direction="both").to_numpy()
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


def normalize_entity_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def load_data_csv(path, glucose_scale=1.0):
    """
    Load glucose CSV: id, device_id, 0..L-1 columns as glucose.
    Handles variable-length sequences, fills NaN.
    """
    df = pd.read_csv(path)

    if "device_id" in df.columns:
        device_ids = df["device_id"].map(normalize_entity_key).tolist()
    elif "id" in df.columns:
        device_ids = df["id"].map(normalize_entity_key).tolist()
    else:
        device_ids = df.iloc[:, 0].map(normalize_entity_key).tolist()

    ids = (
        df["id"].map(normalize_entity_key).tolist()
        if "id" in df.columns
        else list(device_ids)
    )

    glucose_cols = [c for c in df.columns if str(c).isdigit()]
    glucose_cols = sorted(glucose_cols, key=lambda x: int(x))
    glucose_values = df[glucose_cols].values.astype(float) * float(glucose_scale)
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
    if "device_id" not in df.columns:
        raise ValueError("One-hot CSV must contain a 'device_id' column.")
    device_keys = df["device_id"].map(normalize_entity_key)
    for device_id, group in df.groupby(device_keys):
        times = group["TimeIndex"].astype(int).values
        events = group[event_columns].values.astype(float)
        max_t = times.max()
        L = max_t + 1 if L_expected is None else max(L_expected, max_t + 1)
        arr = np.zeros((L, E), dtype=float)
        arr[times, :] = events
        grouped[normalize_entity_key(device_id)] = arr
    return grouped


class EventDataset(Dataset):
    """
    Dataset returning glucose and event patches with forecast targets and masks.
    """

    def __init__(
        self,
        data_csv,
        onehot_csv,
        event_columns,
        patch_size=36,
        stride=6,
        forecast_steps=12,
        glucose_scale=1.0,
        min_patch_valid_ratio=0.5,
    ):
        ids, device_ids, glucose = load_data_csv(data_csv, glucose_scale=glucose_scale)
        self.event_columns = event_columns
        self.events_map = load_onehot_csv(
            onehot_csv, event_columns, L_expected=glucose.shape[1]
        )
        self.patch_size = patch_size
        self.stride = stride
        self.forecast_steps = forecast_steps
        self.window_size = self.patch_size + self.forecast_steps
        self.min_patch_valid_ratio = float(min_patch_valid_ratio)

        kept_ids = []
        kept_device_ids = []
        kept_glucose = []
        self.valid_lengths = []
        self.device_patch_starts = []

        dense_starts = list(
            range(
                0,
                glucose.shape[1] - self.window_size + 1,
                self.stride,
            )
        )

        for idx, dev in enumerate(device_ids):
            valid = np.flatnonzero(~np.isnan(glucose[idx]))
            valid_len = int(valid[-1]) + 1 if valid.size > 0 else 0
            if valid_len < self.window_size:
                continue

            kept_ids.append(normalize_entity_key(ids[idx]))
            kept_device_ids.append(normalize_entity_key(dev))
            kept_glucose.append(glucose[idx])
            self.valid_lengths.append(valid_len)
            self.device_patch_starts.append(list(dense_starts))

        if not kept_glucose:
            raise ValueError(
                "No valid training devices remain after anchor generation. "
                "Check patch_size, forecast_steps and the source data."
            )

        self.ids = kept_ids
        self.device_ids = kept_device_ids
        self.glucose = np.stack(kept_glucose, axis=0)
        self.L = self.glucose.shape[1]

    def __len__(self):
        return len(self.device_ids)

    def __getitem__(self, idx):
        gid = self.ids[idx]
        dev = self.device_ids[idx]
        x = self.glucose[idx].copy()
        events = self.events_map.get(
            dev, np.zeros((self.L, len(self.event_columns)), dtype=float)
        )

        x_inputs = fill_nan_linear(x)
        events = fill_nan_linear(events)
        starts = self.device_patch_starts[idx]

        patches, events_patches, targets, masks = [], [], [], []

        for s in starts:
            if s + self.window_size > len(x):
                continue

            x_patch = x_inputs[s : s + self.patch_size].astype(np.float32)
            ev_patch = events[s : s + self.window_size, :].astype(np.float32)
            y = x[s + self.patch_size : s + self.window_size].astype(np.float32)

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
        patch_starts = np.asarray(starts[: patches.shape[0]], dtype=np.int64)

        return {
            "id": gid,
            "device_id": dev,
            "patches": patches,
            "events": events_patches,
            "targets": targets,
            "masks": masks,
            "patch_starts": patch_starts,
        }

    def infer_continuous_event_columns(self):
        continuous = []
        for j, col in enumerate(self.event_columns):
            arrays = [arr[:, j] for arr in self.events_map.values()]
            merged = np.concatenate(arrays, axis=0)
            if not is_binary_column(merged):
                continuous.append(col)
        return continuous

    def compute_event_stats(self, device_indices=None, continuous_event_columns=None):
        if continuous_event_columns is None:
            continuous_event_columns = self.infer_continuous_event_columns()
        if device_indices is None:
            device_list = self.device_ids
        else:
            device_list = [self.device_ids[i] for i in device_indices]
        stats = {}
        for col in continuous_event_columns:
            j = self.event_columns.index(col)
            arrays = [
                self.events_map[dev][:, j]
                for dev in device_list
                if dev in self.events_map
            ]
            mean, std = compute_nonzero_mean_std(arrays)
            stats[col] = {"mean": mean, "std": std}
        return stats

    def apply_event_stats(self, stats, device_indices=None):
        if device_indices is None:
            device_list = self.device_ids
        else:
            device_list = [self.device_ids[i] for i in device_indices]
        for dev in device_list:
            if dev not in self.events_map:
                continue
            arr = self.events_map[dev].copy()
            for col, st in stats.items():
                if col not in self.event_columns:
                    continue
                j = self.event_columns.index(col)
                mean, std = st["mean"], st["std"]
                mask = (~np.isnan(arr[:, j])) & (arr[:, j] != 0)
                arr[mask, j] = (arr[mask, j] - mean) / std
            self.events_map[dev] = arr

    def subset_by_device_indices(self, idx_iter):
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

    def compute_event_stats(self, continuous_event_columns=None):
        return self.parent.compute_event_stats(
            device_indices=self.indices,
            continuous_event_columns=continuous_event_columns,
        )

    def apply_event_stats(self, stats):
        self.parent.apply_event_stats(stats, device_indices=self.indices)


def collate_patches(batch):
    """Collate batch of samples with padding."""
    B = len(batch)
    Ns = [item["patches"].shape[0] for item in batch]
    N_max = max(Ns)
    P = batch[0]["patches"].shape[1]
    F = batch[0]["targets"].shape[1]
    E = batch[0]["events"].shape[2]

    patches_b = np.zeros((B, N_max, P), dtype=np.float32)
    events_b = np.zeros((B, N_max, P + F, E), dtype=np.float32)
    targets_b = np.zeros((B, N_max, F), dtype=np.float32)
    mask_b = np.zeros((B, N_max, F), dtype=np.float32)
    patch_starts_b = np.zeros((B, N_max), dtype=np.int64)

    for i, item in enumerate(batch):
        n = item["patches"].shape[0]
        patches_b[i, :n, :] = item["patches"]
        events_b[i, :n, :, :] = item["events"]
        targets_b[i, :n, :] = item["targets"]
        mask_b[i, :n, :] = item["masks"]
        patch_starts_b[i, :n] = item["patch_starts"]

    return (
        torch.from_numpy(patches_b),
        torch.from_numpy(events_b),
        torch.from_numpy(targets_b),
        torch.from_numpy(mask_b),
        torch.from_numpy(patch_starts_b),
    )
