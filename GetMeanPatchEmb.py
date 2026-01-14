import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from models.base_model import CGMBaseModel
from utils.data_utils import CGMForecastDataset


def setup_args():
    class Args: pass
    args = Args()

    args.data_csv = "data/Zhao_data.csv"                                # 2022-06_1.csv Colas Multicenter Zhao_data
    args.ckp_path = "checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt"   # forecast-pretrained checkpoint
    args.output_csv = "Results/Emb-Zhao_meanpool.csv"    #Emb-SIBIONICS-2022-06_1_meanpool Emb-Colas_meanpool Emb-Zhao_meanpool.csv

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.batch_size = 128
    args.patch_size = 36
    args.stride = 6
    args.embed_dim = 128
    args.max_seq_len = 3456
    args.forecast_steps = 12   # match checkpoint
    return args


def load_encoder_only(args):
    """
    Load CGMBaseModel and ignore forecast head mismatch if needed.
    """
    model = CGMBaseModel(
        patch_size=args.patch_size,
        forecast_size=args.forecast_steps,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
    )

    state = torch.load(args.ckp_path, map_location="cpu")

    # handle possible wrappers
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # allow forecast_head mismatch safely
    model.load_state_dict(state, strict=False)

    model.eval()
    return model


@torch.no_grad()
def main(args):
    # --------------------------------------------------
    # Load data (same preprocessing as pretrain)
    # --------------------------------------------------
    df = pd.read_csv(args.data_csv, index_col=0)
    data = df[[str(i) for i in range(1, 3456)]]
    data = data[data.notna().sum(axis=1) > 3456* 0.1]
    data_np = data.to_numpy().astype(np.float32)

    dataset = CGMForecastDataset(
        data_np,
        patch_size=args.patch_size,
        stride=args.stride,
        forecast_steps=args.forecast_steps,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print(f"📦 Loaded {len(dataset)} samples")

    # --------------------------------------------------
    # Load pretrained encoder
    # --------------------------------------------------
    model = load_encoder_only(args).to(args.device)

    # --------------------------------------------------
    # Extract mean patch embeddings
    # --------------------------------------------------
    all_embeddings = []
    print(len(loader))
    for patches, patch_mask, _, _ in loader:
        patches = patches.to(args.device)           # [B, N, P]
        patch_mask = patch_mask.to(args.device)     # [B, N]

        # patch embeddings: [B, N, D]
        patch_emb = model(
            patches,
            stride=args.stride,
            return_embedding=True,
        )
        mask = patch_mask.unsqueeze(-1)              # [B, N, 1]
        summed = (patch_emb * mask).sum(dim=1)       # [B, D]
        counts = mask.sum(dim=1).clamp(min=1.0)      # [B, 1]

        mean_emb = summed / counts                   # [B, D]
        all_embeddings.append(mean_emb.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    print("✅ Embedding shape:", all_embeddings.shape)

    # --------------------------------------------------
    # Save to CSV
    # --------------------------------------------------
    emb_df = pd.DataFrame(
        all_embeddings,
        index=data.index,
        columns=[f"emb_{i}" for i in range(all_embeddings.shape[1])]
    )

    emb_df.to_csv(args.output_csv)
    print(f"💾 Saved mean-patch embeddings to {args.output_csv}")


if __name__ == "__main__":
    args = setup_args()
    main(args)
