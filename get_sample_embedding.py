import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from SampleRepresentation.sample_representation_model import CGMSampleRepresentationModel
from SampleRepresentation.sample_representation_utils import CGMSampleContrastiveDataset

# 2022-06_1.csv
# SIBIONICS_Dup_ND
# SIBIONICS_Dup_HRT2D
# SIBIONICS_Dup_T2D
def setup_args():
    class Args: pass
    args = Args()
    args.data_csv = "data/Zhao_data.csv"   # 2022-06_1.csv Colas Multicenter Zhao_data
    args.model_ckpt = "checkpoints_sample_rep/best_sample_rep.pt"
    args.output_csv = "Results/Emb-Zhao_cont.csv"

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = 128
    args.patch_size = 36
    args.stride = 6
    args.embed_dim = 128

    return args


@torch.no_grad()
def main(args):
    # --------------------------------------------------
    # Load data (EXACTLY same preprocessing as training)
    # --------------------------------------------------
    df = pd.read_csv(args.data_csv, index_col=0)
    data = df[[str(i) for i in range(1, 3456)]]
    data = data[data.notna().sum(axis=1) > 3456 *0.1]
    data_np = data.to_numpy().astype(np.float32)

    dataset = CGMSampleContrastiveDataset(
        data_np,
        patch_size=args.patch_size,
        stride=args.stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"📦 Loaded {len(dataset)} samples")

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = CGMSampleRepresentationModel(
        checkpoint_path=args.model_ckpt,   # encoder ckpt is loaded inside
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        finetune_encoder=False,
    ).to(args.device)

    model.eval()

    # --------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------
    all_embeddings = []

    for batch in loader:
        patches = batch["patches"].to(args.device)
        patch_mask = batch["patch_mask"].to(args.device)
        view1 = batch["view1_mask"].to(args.device)
        view2 = batch["view2_mask"].to(args.device)

        # p1, p2 are normalized projections
        p1, p2 = model(
            patches,
            patch_mask,
            view1,
            view2,
            stride=args.stride,
        )

        # sample embedding = mean of two views (in projection space)
        emb = 0.5 * (p1 + p2)
        all_embeddings.append(emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    print("✅ Embeddings shape:", all_embeddings.shape)

    # --------------------------------------------------
    # Save as CSV
    # --------------------------------------------------
    emb_df = pd.DataFrame(
        all_embeddings,
        index=data.index,  # keep original sample index
        columns=[f"emb_{i}" for i in range(all_embeddings.shape[1])]
    )

    emb_df.to_csv(args.output_csv)
    print(f"💾 Saved embeddings to {args.output_csv}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
