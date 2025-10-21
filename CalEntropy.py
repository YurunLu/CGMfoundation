import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base_model import CGMBaseModel
from utils.data_utils import CGMForecastDataset
from utils.loss_utils import compute_forecast_metrics, forecast_loss  # must exist in your utils
import math
import matplotlib.pyplot as plt
import pickle as pkl

@torch.no_grad()
def compute_patch_entropy(transformer_output, mask):
    """
    Compute information entropy for each patch position across samples.
    Args:
        transformer_output: Tensor [B, L, D]
    Returns:
        entropy_per_patch: Tensor [L]
    """
    B, L, D = transformer_output.shape

    # Compute per-patch, per-dim variance across samples (B)
    var = transformer_output.var(dim=0, unbiased=False)  # [L, D]

    # Gaussian entropy per patch: H_i = 0.5 * Σ_d log(2πeσ^2)
    entropy = 0.5 * torch.log(2 * math.pi * math.e * var + 1e-8)  # [L, D]

    # Mean over embedding dims to get single entropy per patch
    entropy_per_patch = entropy.mean(dim=1)  # [L]

    return entropy_per_patch

@torch.no_grad()
def compute_patch_shannon_entropy(transformer_output, mask, bins=30, eps=1e-10, device=None):
    """
    Compute Shannon (discrete) entropy per patch across samples.
    Args:
        transformer_output: Tensor [B, L, D]  (transformer outputs after transformer)
        bins: number of histogram bins per-dimension (int)
        eps: small constant to avoid log(0)
        device: torch.device or None (defaults to transformer_output.device)
    Returns:
        entropy_per_patch: Tensor [L]  non-negative entropies (float)
    Notes:
        - Uses torch.histc which is GPU-capable for float tensors.
        - Loop is over L (patches) and D (dims). For typical sizes this is fine;
          if B is very large you may want to chunk or downsample.
    """
    if device is None:
        device = transformer_output.device
    B, L, D = transformer_output.shape
    # ensure float
    x = transformer_output.to(torch.float32)

    entropies = []
    # We'll compute per-patch entropy: for i in [0..L-1]
    for i in range(L):
        valid_idx = mask[:, i]
        Xi = x[valid_idx, i, :]  # [B, D]
        # compute per-dim histogram and entropy
        Hd = torch.empty(D, device=device, dtype=torch.float32)
        # For stability choose per-dim min/max. If min==max, entropy will be ~0.
        mins = Xi.min(dim=0).values
        maxs = Xi.max(dim=0).values
        # Small safety: ensure max>min, otherwise histc fails; set small range
        same_mask = (maxs - mins) <= 1e-6
        maxs = torch.where(same_mask, mins + 1e-6, maxs)

        for d in range(D):
            vmin = float(mins[d].item())
            vmax = float(maxs[d].item())
            vals = Xi[:, d]
            # torch.histc requires float tensors
            hist = torch.histc(vals, bins=bins, min=vmin, max=vmax)
            total = hist.sum()
            if total <= 0:
                Hd[d] = 0.0
                continue
            p = hist / (total + eps)   # probabilities for bins
            p = p.clamp(min=eps)       # avoid zero
            Hd[d] = - (p * torch.log(p)).sum()
        # average over dims
        entropies.append(Hd.mean())
    entropy_per_patch = torch.stack(entropies, dim=0)  # [L]
    return entropy_per_patch
@torch.no_grad()
def CalculateEntropy(model, loader, device):
    model.eval()
    all_entropies = []
    all_embs = []
    all_mask = []
    for patches, patch_mask, targets, forecast_mask in loader:
        patches = patches.to(device)
        embeddings = model(patches, return_embedding=True)  # [B, L, D]
        all_embs.append(embeddings)
        all_mask.append(forecast_mask)
    all_embs = torch.cat(all_embs, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    print(all_embs.shape)
    print(all_mask.shape)
    entropy_all = compute_patch_shannon_entropy(all_embs,all_mask).cpu()
    #entropy_all = torch.stack(all_entropies, dim=0).mean(dim=0)  # [L]
    return entropy_all

def main():
    # ==== Config ====
    csv_path = "data/2022-06_1.csv"
    checkpoint_path = "checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_size = 36
    stride = 6
    forecast_steps = 12
    embed_dim = 128
    batch_size = 64

    # ==== Data ====
    df = pd.read_csv(csv_path, index_col=0)
    df = df[[c for c in df.columns if c.isdigit()]]
    data = df[df.notna().sum(axis=1) > df.shape[1] / 2].to_numpy().astype(np.float32)

    dataset = CGMForecastDataset(data, patch_size=patch_size,
                                 stride=stride, forecast_steps=forecast_steps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ==== Model ====
    model = CGMBaseModel(patch_size=patch_size,
                         forecast_size=forecast_steps,
                         embed_dim=embed_dim,
                         max_seq_len=3456).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt)
    print(f"✅ Loaded checkpoint: {checkpoint_path}")

    entropy = CalculateEntropy(model, loader, device).numpy()
    for i in range(12):
        length = min(i*48+48,entropy.shape[0])-i*48
        plt.plot([x/2 for x in range(length)],entropy[i*48:min(i*48+48,entropy.shape[0])],c='gray')
    plt.xlabel("Time")
    plt.ylabel("Entropy")
    plt.title("Information Entropy Across Samples")
    plt.savefig('./Entropy.png')
    np.save('./Entropy.npy', entropy)

if __name__ == "__main__":
    main()
