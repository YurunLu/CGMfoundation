import os, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime

from models.base_model import CGMBaseModel
from utils.data_utils import CGMForecastDataset
from utils.loss_utils import forecast_loss, compute_forecast_metrics
from utils.train_utils import WarmupCosineScheduler


def setup_args():
    class Args: pass
    args = Args()
    args.data_csv = "data/2022-06_1.csv"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.epochs = 3000
    args.batch_size = 128
    args.patch_size = 36
    args.stride = 6
    args.forecast_steps = 12
    args.embed_dim = 128
    args.save_dir = "checkpoints"
    args.lr = 1e-3
    args.patience = 50
    return args


def run_pretrain_forecast(args=None):
    if args is None: args = setup_args()

    os.makedirs(args.save_dir, exist_ok=True)
    data = pd.read_csv(args.data_csv, index_col=0)[[str(x) for x in range(1, 3456)]]
    data = data[data.notna().sum(axis=1) > 3456 / 2].to_numpy().astype(np.float32)
    dataset = CGMForecastDataset(data, patch_size=args.patch_size,
                                 stride=args.stride, forecast_steps=args.forecast_steps)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CGMBaseModel(patch_size=args.patch_size, forecast_size=args.forecast_steps,
                         embed_dim=args.embed_dim, max_seq_len=3456).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = WarmupCosineScheduler(opt, warmup_steps=500,
                                      max_steps=args.epochs * len(loader))

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.save_dir, f"log_{time_tag}.csv")
    log_df = []

    best_loss, no_improve = float("inf"), 0
    best_state = None

    print(f"🚀 Start Forecast Pretraining on {len(dataset)} samples.")

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for patches, patch_mask, targets, forecast_mask in loader:
            patches, targets = patches.to(args.device), targets.to(args.device)
            forecast_mask = forecast_mask.to(args.device)

            pred_forecast = model(patches)
            loss = forecast_loss(pred_forecast, targets, forecast_mask)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        # --- Per-epoch evaluation ---
        model.eval()
        with torch.no_grad():
            patches, patch_mask, targets, forecast_mask = next(iter(loader))
            patches, targets, forecast_mask = (
                patches.to(args.device), targets.to(args.device), forecast_mask.to(args.device)
            )
            pred_forecast = model(patches)
            metrics = compute_forecast_metrics(pred_forecast, targets, forecast_mask)

        grouped = metrics["grouped_mae"]
        hyper, hypo = metrics["hyper"], metrics["hypo"]

        print(f"[Epoch {ep}] Loss={avg_loss:.4f} "
              f"| MAE(0–10)={grouped.get('0-10min',0):.3f} "
              f"| MAE(10–30)={grouped.get('10-30min',0):.3f} "
              f"| MAE(30–60)={grouped.get('30-60min',0):.3f} "
              f"| Hyper [Acc={hyper['acc']:.3f}, F1={hyper['f1']:.3f}] "
              f"| Hypo [Acc={hypo['acc']:.3f}, F1={hypo['f1']:.3f}]")

        # --- Early stopping tracking ---
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"⏹️ Early stop at epoch {ep}. Best loss {best_loss:.4f}")
                break

        # Log metrics
        log_df.append({
            "epoch": ep, "loss": avg_loss,
            **{f"mae_{k}": v for k, v in grouped.items()},
            **{f"hyper_{k}": v for k, v in hyper.items()},
            **{f"hypo_{k}": v for k, v in hypo.items()}
        })
        pd.DataFrame(log_df).to_csv(log_file, index=False)

    # --- Save best model and show final metrics ---
    print("✅ Training finished. Evaluating best model...")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        patches, patch_mask, targets, forecast_mask = next(iter(loader))
        patches, targets, forecast_mask = (
            patches.to(args.device), targets.to(args.device), forecast_mask.to(args.device)
        )
        pred_forecast = model(patches)
        metrics = compute_forecast_metrics(pred_forecast, targets, forecast_mask)

    grouped = metrics["grouped_mae"]
    hyper, hypo = metrics["hyper"], metrics["hypo"]

    print(f"🏁 Final Evaluation:")
    print(f"MAE(0–10)={grouped.get('0-10min',0):.3f} | "
          f"MAE(10–30)={grouped.get('10-30min',0):.3f} | "
          f"MAE(30–60)={grouped.get('30-60min',0):.3f}")
    print(f"Hyper [Acc={hyper['acc']:.3f}, Prec={hyper['prec']:.3f}, "
          f"Rec={hyper['rec']:.3f}, F1={hyper['f1']:.3f}]")
    print(f"Hypo [Acc={hypo['acc']:.3f}, Prec={hypo['prec']:.3f}, "
          f"Rec={hypo['rec']:.3f}, F1={hypo['f1']:.3f}]")

    # --- Save best model ---
    best_path = os.path.join(
        args.save_dir,
        f"best_forecast_{time_tag}_dim{args.embed_dim}_patch{args.patch_size}_stride{args.stride}.pt"
    )
    torch.save(best_state, best_path)
    print(f"💾 Best model saved at {best_path}")

    return model, best_path


if __name__ == "__main__":
    args = setup_args()
    run_pretrain_forecast(args)
