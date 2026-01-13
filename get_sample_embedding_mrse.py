# get_sample_embedding_mrse.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from utils.data_utils import CGMForecastDataset
from SampleRepresentation.sample_representation_model_mrse import CGMSampleMRSE


def setup_args():
    class Args: pass
    args = Args()

    args.data_csv = "data/2022-06_1.csv"
    args.base_ckp_path = "checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt"
    args.mrse_ckpt = "checkpoints_sample_mrse/mrse.pt"

    args.output_csv = "Results/Emb-MRSE-2022-06_1.csv"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.batch_size = 128
    args.patch_size = 36
    args.stride = 6
    args.embed_dim = 128
    args.max_seq_len = 3456

    return args


@torch.no_grad()
def main(args):
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(args.data_csv, index_col=0)
    data = df[[str(i) for i in range(1, 3456)]]
    data = data[data.notna().sum(axis=1) > 3456 / 2]
    data_np = data.to_numpy().astype(np.float32)

    dataset = CGMForecastDataset(
        data_np,
        patch_size=args.patch_size,
        stride=args.stride,
        forecast_steps=1,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # --- Build model and load MRSE weights ---
    model = CGMSampleMRSE(
        checkpoint_path=args.base_ckp_path,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        finetune_encoder=False,
    ).to(args.device)

    state = torch.load(args.mrse_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    all_emb = []

    for patches, patch_mask, _, _ in loader:
        patches = patches.to(args.device)
        patch_mask = patch_mask.to(args.device)

        mean_emb, resid_emb = model(patches, patch_mask, stride=args.stride, resid_mask=patch_mask)
        emb = torch.cat([mean_emb, resid_emb], dim=1)  # [B, 2D]
        all_emb.append(emb.cpu())

    all_emb = torch.cat(all_emb, dim=0).numpy()

    out_df = pd.DataFrame(
        all_emb,
        index=data.index,
        columns=[f"emb_{i}" for i in range(all_emb.shape[1])]
    )
    out_df.to_csv(args.output_csv)
    print(f"💾 Saved MRSE embeddings: {args.output_csv} | shape={all_emb.shape}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
