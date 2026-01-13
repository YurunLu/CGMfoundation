# calDNB.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from models.base_model import CGMBaseModel
from utils.data_utils import CGMForecastDataset
from utils.calDNB_utils import (
    dnb_sd_batch,
    dnb_pcc_batch,
    lyap_proxy_jacobian_batch,
)


def load_model(ckp_path, device):
    model = CGMBaseModel(
        patch_size=36,
        forecast_size=12,     # must match checkpoint, but head is unused
        embed_dim=128,
        max_seq_len=3456
    ).to(device)

    state = torch.load(ckp_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model

# 2022-06_1
# best_forecast_20251012_140643_dim128_patch36_stride6
# dnb_metrics

@torch.no_grad()
def main(
    data_csv="data/Colas.csv",
    ckp_path="checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt",
    out_csv="Results/Colas_dnb_metrics.csv",
    patch_size=36,
    stride=12,
    batch_size=64,
    num_workers=4,
    ridge=1e-2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # =========================
    # Load data and raw IDs
    # =========================
    df = pd.read_csv(data_csv)

    raw_ids = df["id"].to_numpy()  # 🔑 preserve original sample id
    cgm_cols = [str(i) for i in range(1, 3456)]
    data_all = df[cgm_cols].to_numpy(dtype=np.float32)

    keep = np.where(np.isnan(data_all).sum(axis=1) < 0.9 * data_all.shape[1])[0]
    data = data_all[keep]
    kept_ids = raw_ids[keep]

    # =========================
    # Dataset & DataLoader
    # =========================
    dataset = CGMForecastDataset(
        data,
        patch_size=patch_size,
        stride=stride,
        forecast_steps=patch_size
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    # =========================
    # Model
    # =========================
    model = load_model(ckp_path, device)

    # =========================
    # Batched computation
    # =========================
    records = []
    processed = 0

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    for patches, patch_mask, _, _ in loader:
        if patches.numel() == 0:
            continue

        patches = patches.to(device, non_blocking=True)
        mask = patch_mask.to(device, non_blocking=True)

        emb = model(patches, return_embedding=True)   # [B,T,D]

        sd = dnb_sd_batch(emb, mask)
        pcc = dnb_pcc_batch(emb, mask)
        lyap = lyap_proxy_jacobian_batch(emb, mask, ridge=ridge)

        sd = sd.cpu().numpy()
        pcc = pcc.cpu().numpy()
        lyap = lyap.cpu().numpy()
        n_valid = mask.sum(dim=1).cpu().numpy().astype(int)

        B = len(sd)
        for b in range(B):
            records.append({
                "sample_id": kept_ids[processed + b],   # 🔑 raw CSV id
                "num_patches": int(n_valid[b]),
                "dnb_sd": float(sd[b]),
                "dnb_pcc": float(pcc[b]),
                "lyap_proxy_log_smaxA": float(lyap[b]),
            })

        processed += B
        if processed % (batch_size * 10) == 0:
            print(f"Processed {processed}/{len(kept_ids)} samples")

    # =========================
    # Save results
    # =========================
    out_df = pd.DataFrame(records)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved DNB metrics to {out_csv}")
    print(out_df.head())


if __name__ == "__main__":
    main()
