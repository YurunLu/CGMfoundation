# dataset_cgm.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
from utils.data_utils import CGMForecastDataset, fill_nan_lin

class CGMLabelPredictionDataset(Dataset):
    """
    Dataset for CGM label prediction task.
    Each sample -> multiple overlapping patches (from CGMForecastDataset) + one global label.
    """
    def __init__(self, cgm_path, label_path, target = 'duration', type = 'reg',
                 patch_size=36, stride=12, forecast_steps=0):
        # 1. Load data
        cgm_df = pd.read_csv(cgm_path,index_col=0)
        label_df = pd.read_csv(label_path,index_col=0)

        # 2. Merge on sample_id
        cgm_df.loc[:,target] = label_df.loc[cgm_df.index,target]

        if target == 'duration':
            cgm_df['type'] = label_df['type']
            cgm_df= cgm_df[cgm_df['type'] != 'T1D']
        if type == 'reg':
            cgm_df[target] = (cgm_df[target]-cgm_df[target].mean())/cgm_df[target].std()
        # 3. Drop NaN targets
        cgm_df = cgm_df.dropna(subset=[target])

        # 4. Separate target and CGM matrix
        self.sample_ids = cgm_df.index
        self.labels = cgm_df[target].astype(np.float32).values
        cgm_data = cgm_df[[str(x) for x in range(3456)]]

        # 5. Interpolate NaNs (leaving trailing NaN as is)
        for i in cgm_data.index:
            seq = cgm_data.loc[i]
            filled = fill_nan_lin(seq)
            cgm_data.loc[i,:] = filled.astype(np.float32)

        # 6. Patchify via CGMForecastDataset
        self.forecast_ds = CGMForecastDataset(
            cgm_data, patch_size=patch_size, stride=stride, forecast_steps=forecast_steps
        )

        # Keep same order of labels
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __len__(self):
        return len(self.forecast_ds)

    def __getitem__(self, idx):
        patches, patch_mask, targets, target_mask = self.forecast_ds[idx]
        label = self.labels[idx]
        return {
            "patches": patches,               # [N_patches, patch_size]
            "patch_mask": patch_mask,         # [N_patches]
            "label": label,                   # scalar (duration)
            "forecast_mask": target_mask,
            "sample_id": self.forecast_ds.data[idx]
        }


def collate_fn(batch):
    """
    Collate variable-length sequences by padding in time dimension (N).
    - patches padded to max_N using 0. (Assumes model temporal embedding can handle position offsets).
    - forecast_mask padded accordingly.
    - labels stacked where available.
    """
    patches_list = [b['patches'] for b in batch]
    Ns = [p.shape[0] for p in patches_list]
    maxN = max(Ns)
    P = patches_list[0].shape[1]

    batch_patches = torch.zeros((len(batch), maxN, P), dtype=torch.float32)
    batch_mask = torch.zeros((len(batch), maxN), dtype=torch.bool)
    cls_labels = []
    reg_labels = []
    has_cls = any(b['label_cls'] is not None for b in batch)
    has_reg = any(b['label_reg'] is not None for b in batch)

    for i, b in enumerate(batch):
        n = b['patches'].shape[0]
        batch_patches[i, :n] = b['patches']
        batch_mask[i, :n] = b['forecast_mask']
        cls_labels.append(b['label_cls'] if b['label_cls'] is not None else None)
        reg_labels.append(b['label_reg'] if b['label_reg'] is not None else None)

    out = {
        'patches': batch_patches,          # [B, maxN, P]
        'forecast_mask': batch_mask,       # [B, maxN]
    }

    if has_cls:
        # convert None to -1 (ignored index), or keep bool mask for loss
        cls_tensor = torch.tensor([(-1 if v is None else v) for v in cls_labels], dtype=torch.long)
        out['label_cls'] = cls_tensor

    if has_reg:
        # pad regression labels if varying dims; here assume same dim or None
        reg_tensor = torch.stack([ (torch.zeros_like(reg_labels[0]) if v is None else v) for v in reg_labels ], dim=0)
        out['label_reg'] = reg_tensor

    return out

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_pretrained_weights(model, checkpoint_path, strict=False, skip_keys=None):
    """
    Loads weights into `model` from checkpoint_path safely.
    skip_keys: list of key substrings to skip (e.g., forecast_head)
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    if skip_keys:
        sd = {k:v for k,v in sd.items() if not any(sk in k for sk in skip_keys)}

    model.load_state_dict(sd, strict=strict)
    return model
