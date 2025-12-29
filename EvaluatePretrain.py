import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base_model import CGMBaseModel
from utils.data_utils import CGMForecastDataset
from utils.loss_utils import compute_forecast_metrics, forecast_loss  # must exist in your utils

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_targets, all_masks = [], [], []

    for patches, patch_mask, targets, forecast_mask in loader:
        patches, targets = patches.to(device), targets.to(device)
        forecast_mask = forecast_mask.to(device)
        preds = model(patches)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_masks.append(forecast_mask.cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    loss = forecast_loss(preds, targets, masks)
    # --- Use your metric helper (same as pretrain) ---
    results = compute_forecast_metrics(preds, targets, masks)
    grouped = results["grouped_mae"]
    hyper, hypo = results["hyper"], results["hypo"]

    print(f"[Eval] Loss={loss.item():.4f} "
          f"| MAE(0–10)={grouped.get('0-10min',0):.3f} "
          f"| MAE(10–30)={grouped.get('10-30min',0):.3f} "
          f"| MAE(30–60)={grouped.get('30-60min',0):.3f} "
          f"| Hyper [Acc={hyper['acc']:.3f}, F1={hyper['f1']:.3f}] "
          f"| Hypo [Acc={hypo['acc']:.3f}, F1={hypo['f1']:.3f}]")
    return results

# best_forecast_20251012_140643_dim128_patch36_stride6.pt
# best_forecast_20251106_194359_dim128_patch36_stride6
def main():
    # ==== Config ====
    csv_path = "data/PUHP_CGM.csv"
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

    # ==== Evaluate ====
    results = evaluate_model(model, loader, device)
    '''
    # ==== Save ====
    result_path = os.path.join(os.path.dirname(checkpoint_path), "evaluation_summary.csv")
    pd.DataFrame([{
        "Loss": results["loss"],
        "MAE_0_10": results["mae_groups"].get("0-10min", 0),
        "MAE_10_30": results["mae_groups"].get("10-30min", 0),
        "MAE_30_60": results["mae_groups"].get("30-60min", 0),
        "Hyper_Acc": results["hyper"]["acc"],
        "Hyper_F1": results["hyper"]["f1"],
        "Hypo_Acc": results["hypo"]["acc"],
        "Hypo_F1": results["hypo"]["f1"]
    }]).to_csv(result_path, index=False)
    print(f"💾 Saved summary to {result_path}")
    '''


if __name__ == "__main__":
    main()
