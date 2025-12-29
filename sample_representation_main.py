# sample_representation_main.py
import os
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from SampleRepresentation.sample_representation_model import CGMSampleRepresentationModel
from SampleRepresentation.sample_representation_utils import (
    CGMSampleContrastiveDataset,
    info_nce_loss,
)


def setup_args():
    class Args: pass
    args = Args()
    args.data_csv = "data/2022-06_1.csv"
    args.ckp_path = "checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = 128
    args.epochs = 3000
    args.patch_size = 36
    args.stride = 6
    args.embed_dim = 128
    args.lr = 1e-3
    args.save_dir = "checkpoints_sample_rep"
    args.save_name = "best_sample_rep.pt"
    return args


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load data ---
    df = pd.read_csv(args.data_csv, index_col=0)
    data = df[[str(i) for i in range(1, 3456)]]
    data = data[data.notna().sum(axis=1) > 3456 / 2]
    data = data.to_numpy().astype(np.float32)

    dataset = CGMSampleContrastiveDataset(
        data,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # --- Model ---
    model = CGMSampleRepresentationModel(
        checkpoint_path=args.ckp_path,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        finetune_encoder=False,
    ).to(args.device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    print(f"🚀 Start sample-level contrastive pretraining ({len(dataset)} samples)")

    best_loss = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            patches = batch["patches"].to(args.device)
            patch_mask = batch["patch_mask"].to(args.device)
            view1 = batch["view1_mask"].to(args.device)
            view2 = batch["view2_mask"].to(args.device)

            z1, z2 = model(
                patches,
                patch_mask,
                view1,
                view2,
                stride=args.stride,
            )

            loss = info_nce_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {ep}] Contrastive loss = {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, args.save_name)
            torch.save(model.state_dict(), save_path)

    print(f"✅ Finished. Best loss = {best_loss:.4f}")
    print(f"💾 Model saved to {save_path}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
