import os, torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

from models.base_model import CGMBaseModel
from utils.data_utils import CGMForecastDataset
from utils.loss_utils import forecast_loss
from utils.train_utils import WarmupCosineScheduler

def run_pretrain_forecast(data_csv="data/2022-06_1.csv",
                          device="cuda",
                          epochs=300,
                          batch_size=128,
                          patch_size=36,
                          stride=6,
                          forecast_steps=6,
                          embed_dim=128,
                          save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)
    data = pd.read_csv(data_csv, index_col=0)[[str(x) for x in range(1, 3456)]]
    data = data[data.notna().sum(axis=1) > 3456/2]
    data = data.to_numpy().astype(np.float32)
    dataset = CGMForecastDataset(data, patch_size=patch_size, stride=stride, forecast_steps=forecast_steps)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CGMBaseModel(patch_size=patch_size, forecast_size=forecast_steps, embed_dim=embed_dim,max_seq_len=3456)
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(opt, warmup_steps=500, max_steps=epochs * len(loader))

    print(f"Start Forecast Pretraining on {len(dataset)} samples.")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for patches, patch_mask, targets, forecast_mask in loader:
            patches, targets = patches.to(device), targets.to(device)
            forecast_mask = forecast_mask.to(device)

            pred_forecast = model(patches)
            loss = forecast_loss(pred_forecast, targets, forecast_mask)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); scheduler.step()
            total_loss += loss.item()

        print(f"[Epoch {ep}] avg loss: {total_loss / len(loader):.6f}")

    ckpt_path = os.path.join(save_dir, f"forecast_pretrain_ep{epochs}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Model saved to {ckpt_path}")
    return model, ckpt_path


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_pretrain_forecast(device=device)
