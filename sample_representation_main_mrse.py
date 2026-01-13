# sample_representation_main_mrse.py
import os
import torch
import torch.nn.functional as F  # IMPORTANT: included as you requested
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from utils.data_utils import CGMForecastDataset
from SampleRepresentation.sample_representation_model_mrse import CGMSampleMRSE
from SampleRepresentation.sample_representation_utils_mrse import (
    split_mask_nonempty,
    residual_consistency_loss,
    variance_regularization
)


def setup_args():
    class Args: pass
    args = Args()

    # data / ckpt
    args.data_csv = "data/2022-06_1.csv"
    args.base_ckp_path = "checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt"

    # training
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.epochs = 50
    args.batch_size = 128
    args.lr = 1e-3
    args.clip_grad = 1.0

    # model params
    args.patch_size = 36
    args.stride = 6
    args.embed_dim = 128
    args.max_seq_len = 3456
    args.finetune_encoder = False
    args.resid_hidden_dim = 256
    args.resid_dropout = 0.0

    # loss
    args.split_p = 0.5
    args.loss_mode = "mse"  # or "cosine"

    # saving
    args.save_dir = "checkpoints_sample_mrse"
    args.save_name = "mrse.pt"

    return args


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Load data (match your previous preprocessing) ---
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

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # --- Model ---
    model = CGMSampleMRSE(
        checkpoint_path=args.base_ckp_path,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        finetune_encoder=args.finetune_encoder,
        resid_hidden_dim=args.resid_hidden_dim,
        resid_dropout=args.resid_dropout,
    ).to(args.device)

    # optimize only trainable params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    print(f"🚀 Train MRSE on {len(dataset)} samples. finetune_encoder={args.finetune_encoder}")

    best_loss = float("inf")
    save_path = os.path.join(args.save_dir, args.save_name)

    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0

        for patches, patch_mask, _, _ in loader:
            patches = patches.to(args.device)
            patch_mask = patch_mask.to(args.device)

            # split mask (two subsets) but keep them non-empty when possible
            m1, m2 = split_mask_nonempty(patch_mask, p=args.split_p)

            # forward: mean uses full patch_mask; resid uses m1/m2
            _, r1 = model(patches, patch_mask, stride=args.stride, resid_mask=m1)
            _, r2 = model(patches, patch_mask, stride=args.stride, resid_mask=m2)

            loss_cons = residual_consistency_loss(r1, r2)
            loss_var = variance_regularization(r1) + variance_regularization(r2)

            loss = loss_cons + loss_var

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.clip_grad is not None and args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        print(f"[Epoch {ep}] Residual consistency loss = {avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)

    print(f"✅ Finished. Best loss = {best_loss:.6f}")
    print(f"💾 Saved to {save_path}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
