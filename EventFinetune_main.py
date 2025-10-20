import os
import argparse
import math
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from EventFinetune.Event_Finetune_data_utils import EventDataset, collate_patches
from EventFinetune.Event_Finetune_model import EventFinetuneModel
from EventFinetune.Event_loss_utils import LossTracker, masked_mse_loss, masked_mae_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', type=str, default='./data/data_withevent_top1k.csv')
    parser.add_argument('--onehot_csv', type=str, default='./data/OneHot_all_top1k.csv')
    parser.add_argument('--pretrain_ckpt', type=str, default='./checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt')
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patch_size', type=int, default=36)
    parser.add_argument('--stride', type=int, default=6)
    parser.add_argument('--forecast_steps', type=int, default=12)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--event_cols', type=str, nargs='+', default=["Metformin","Insulin","Diet","Exercise"]) #,"GLP-1 receptor agonists","Calories"
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    return args


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for patches, events, targets, mask in dataloader:
        patches, events, targets, mask = patches.to(device), events.to(device), targets.to(device), mask.to(device)
        preds = model(patches, events, start_idx=0, stride=6)
        loss = masked_mae_loss(preds, targets, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * patches.shape[0]
        n += patches.shape[0]
    return total_loss / max(n,1)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in dataloader:
            patches, events, targets, mask = batch
            patches = patches.to(device)
            events = events.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            preds = model(patches, events)
            loss = masked_mae_loss(preds, targets, mask)
            total_loss += loss.item() * patches.shape[0]
            n += patches.shape[0]
    return total_loss / max(n,1)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # dataset
    dataset = EventDataset(args.data_csv, args.onehot_csv, event_columns=args.event_cols,
                           patch_size=args.patch_size, stride=args.stride, forecast_steps=args.forecast_steps)
    # simple split by index
    n = len(dataset.device_ids)
    train_idx = range(0, int(0.8*n))
    val_idx = range(int(0.8*n), n)

    train_set = dataset.subset_by_device_indices(train_idx)
    val_set = dataset.subset_by_device_indices(val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_patches, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_patches, num_workers=args.num_workers)

    device = torch.device(args.device)

    # model

    model = EventFinetuneModel(
        pretrained_ckpt_path=args.pretrain_ckpt,
        num_event_types=len(args.event_cols),
        patch_size=args.patch_size,
        forecast_size=args.forecast_steps,
    ).to(device)

    model.to(device)

    with torch.no_grad():
        total_loss = 0
        for patches, events, targets, mask in train_loader:
            patches = patches.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            preds = model.base(patches)
            total_loss += masked_mae_loss(preds, targets, mask).item()
        print(f"Raw loss on Train {total_loss / len(train_loader):6f}")
        total_loss = 0
        for patches, events, targets, mask in val_loader:
            patches = patches.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            preds = model.base(patches)
            total_loss += masked_mae_loss(preds, targets, mask).item()
        print(f"Raw loss on Validation {total_loss / len(val_loader):6f}")

    for name, p in model.base.named_parameters():
        p.requires_grad = False


    freeze_epochs = 50
    optimizer = torch.optim.AdamW([
        {'params': model.base.parameters(), 'lr': 1e-6},
        {'params': list(model.event_emb.parameters()) +
                   list(model.past_event_proj.parameters()) +
                   list(model.future_event_proj.parameters()) +
                   list(model.future_mlp.parameters()) +
                   list(model.past_attn_score.parameters()) +
                   [model.alpha_past, model.alpha_future],
         'lr': 5e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float('inf')
    print("\n[INFO] Starting fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        # Unfreeze base model after a few epochs
        if epoch == freeze_epochs + 1:
            for p in model.base.parameters():
                p.requires_grad = True
            print(f"[INFO] Unfroze base model at epoch {epoch}")

        model.train()
        total_loss, n = 0.0, 0
        for patches, events, targets, mask in train_loader:
            patches, events, targets, mask = (
                patches.to(device), events.to(device), targets.to(device), mask.to(device)
            )
            optimizer.zero_grad(set_to_none=True)
            preds = model(patches, events)
            loss = masked_mae_loss(preds, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * patches.shape[0]
            n += patches.shape[0]
        train_loss = total_loss / max(n, 1)
        scheduler.step()

        if epoch % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:4d} | train={train_loss:.5f} | val={val_loss:.5f} "
                  f"| α_past={model.alpha_past.item():.4f} α_future={model.alpha_future.item():.4f}")
            if val_loss < best_val:
                best_val = val_loss
        else:
            print(f"Epoch {epoch:4d} | train={train_loss:.5f}")

    print(f"[INFO] Best validation loss: {best_val:.5f}")

    '''
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        if epoch % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                #ckpt_path = Path(args.out_dir) / f"best_model_epoch{epoch}.pt"
                #torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)
                #print(f"Saved best model to {ckpt_path}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}")
    '''


if __name__ == '__main__':
    main()