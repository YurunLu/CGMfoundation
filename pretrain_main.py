import os
from pathlib import Path
import yaml
import argparse
import math 
import torch
import torch.nn as nn
import pandas as pd
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from datetime import datetime, timedelta

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed, DistributedDataParallelKwargs

from models.base_model import CGMBaseModel
from utils.loss_utils import forecast_loss, compute_forecast_metrics, event_based_metrics
from utils.train_utils import WarmupCosineScheduler

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "config.yaml"


class OnTheFlyDataset(Dataset):
    def __init__(self, hf_dataset, patch_size, stride, forecast_steps):
        self.hf_dataset = hf_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.forecast_steps = forecast_steps
        self.window_size = patch_size + forecast_steps

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        seq = item['seq_filled']
        mask = item['orig_mask']
        
        seq_windows = seq.unfold(0, self.window_size, self.stride).contiguous()
        mask_windows = mask.unfold(0, self.window_size, self.stride).contiguous()
        
        return {
            "patches": seq_windows[:, :self.patch_size],
            "patch_mask": mask_windows[:, :self.patch_size],
            "targets": seq_windows[:, self.patch_size:],
            "forecast_mask": mask_windows[:, self.patch_size:]
        }

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_loss_kwargs(config):
    loss_cfg = config.get("loss", {})
    return {
        "hypo_thr": loss_cfg.get("hypo_thr", 3.9),
        "hyper_thr": loss_cfg.get("hyper_thr", 10.0),
        "hypo_weight": loss_cfg.get("hypo_weight", 2.0),
        "hyper_weight": loss_cfg.get("hyper_weight", 1.2),
        "normal_weight": loss_cfg.get("normal_weight", 1.0),
    }

def run_pretrain_forecast(config):
    set_seed(config['training'].get('seed', 42))
    loss_kwargs = get_loss_kwargs(config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, process_group_kwargs])

    if accelerator.is_main_process:
        wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'], config=config)

    save_dir = config['training']['save_dir']
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    hf_dir = config['data']['hf_dataset_dir']
    if accelerator.is_main_process:
        print(f"Loading the preprocessed dataset...")
    
    hf_datasets = load_from_disk(hf_dir).with_format("torch")
    
    train_ds = OnTheFlyDataset(hf_datasets['train'], config['model']['patch_size'], config['model']['stride'], config['model']['forecast_steps'])
    val_ds = OnTheFlyDataset(hf_datasets['validation'], config['model']['patch_size'], config['model']['stride'], config['model']['forecast_steps'])
    test_ds = OnTheFlyDataset(hf_datasets['test'], config['model']['patch_size'], config['model']['stride'], config['model']['forecast_steps'])

    num_workers = config['training'].get('num_workers', 8) 
    
    train_loader = DataLoader(
        train_ds, batch_size=config['training']['batch_size'], shuffle=True, 
        num_workers=num_workers, drop_last=True, pin_memory=True, prefetch_factor=4       
    )
    val_loader = DataLoader(
        val_ds, batch_size=config['training']['batch_size'], shuffle=False, 
        num_workers=num_workers, drop_last=True, pin_memory=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=config['training']['batch_size'], shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    model = CGMBaseModel(
        patch_size=config['model']['patch_size'],
        forecast_size=config['model']['forecast_steps'],
        embed_dim=config['model']['embed_dim'],
        max_seq_len=config['model']['max_seq_len'],
        depth=config['model'].get('depth', 4),
        num_heads=config['model'].get('num_heads', 8),
        hidden_dim=config['model'].get('hidden_dim', 128),
        dropout=config['model'].get('dropout', 0.2),
        use_cnn=config['model'].get('use_cnn', True), 
        use_transformer=config['model'].get('use_transformer', True),
        use_temporal=config['model'].get('use_temporal', True),
        use_positional=config['model'].get('use_positional', True)
    )

    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\n" + "=" * 60)
        print(f"Model parameter counts")
        print(f"   Total parameters:     {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Approximate size:     {total_params / 1e6:.2f} M")
        print("=" * 60 + "\n")

    weight_decay = config['training'].get('weight_decay', 0.01)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'norm']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    num_processes = accelerator.num_processes 
    scaled_lr = config['training']['lr'] * math.sqrt(num_processes)
    
    if accelerator.is_main_process:
        print(f"[Automatic LR scaling] Base LR for one GPU: {config['training']['lr']} --> {num_processes}-GPU scaled LR: {scaled_lr:.6f}\n")

    opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=scaled_lr)
    
    model, opt, train_loader, val_loader, test_loader = accelerator.prepare(
        model, opt, train_loader, val_loader, test_loader
    )

    warmup_steps = config['training'].get('warmup_steps', 500)
    total_steps = config['training']['epochs'] * len(train_loader) 
    scheduler = WarmupCosineScheduler(opt, warmup_steps=warmup_steps, max_steps=total_steps)

    scheduler = accelerator.prepare(scheduler)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"log_{time_tag}.csv")
    log_df = []

    best_val_loss, no_improve = float("inf"), 0
    best_state = None
    best_path = None

    if accelerator.is_main_process:
        print(f"Starting pretraining...")

    for ep in range(1, config['training']['epochs'] + 1):
        model.train()
        total_train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{config['training']['epochs']} [Train]", disable=not accelerator.is_main_process)
        for batch in pbar:
            patches = batch['patches']
            forecast_mask = batch['forecast_mask']
            targets = batch['targets']

            pred_forecast = model(patches, stride=config['model']['stride'])
            loss = forecast_loss(pred_forecast, targets, forecast_mask, **loss_kwargs)

            opt.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            scheduler.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        all_preds, all_targets, all_masks = [], [], []

        pbar_val = tqdm(val_loader, desc=f"Epoch {ep}/{config['training']['epochs']} [Val]", leave=False, disable=not accelerator.is_main_process)
        
        with torch.no_grad():
            for batch in pbar_val:
                pred_forecast = model(batch['patches'], stride=config['model']['stride'])
                pred_gathered = accelerator.gather_for_metrics(pred_forecast)
                targets_gathered = accelerator.gather_for_metrics(batch['targets'])
                mask_gathered = accelerator.gather_for_metrics(batch['forecast_mask'])

                all_preds.append(pred_gathered.cpu())
                all_targets.append(targets_gathered.cpu())
                all_masks.append(mask_gathered.cpu())

        accelerator.wait_for_everyone()
        should_stop = False 

        if accelerator.is_main_process:
            avg_train_loss = total_train_loss / len(train_loader)
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            all_masks = torch.cat(all_masks, dim=0)

            val_loss = forecast_loss(all_preds, all_targets, all_masks, **loss_kwargs)
            avg_val_loss = val_loss.item()
            
            val_metrics = compute_forecast_metrics(all_preds, all_targets, all_masks)
            grouped = val_metrics["grouped_mae"]
            ceg = val_metrics["ceg"]
            val_event_hyper = event_based_metrics(all_preds, all_targets, all_masks, thr=10.0, is_hypo=False)
            val_event_hypo = event_based_metrics(all_preds, all_targets, all_masks, thr=3.9, is_hypo=True)

            print(f"\n[Epoch {ep}] Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f}")
            wandb.log({
                "epoch": ep, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                "lr": opt.param_groups[0]['lr'],
                **{f"val_mae_{k}": v for k, v in grouped.items()},
                "val_ceg_A": ceg.get('A', 0),
                "val_ceg_B": ceg.get('B', 0),
                "val_hypo_f1": val_event_hypo.get('f1', 0),
                "val_hyper_f1": val_event_hyper.get('f1', 0)
            })

            unwrapped_model = accelerator.unwrap_model(model)
            current_model_state = {k: v.cpu() for k, v in unwrapped_model.state_dict().items()}
            
            # Include architecture flags in the checkpoint metadata.
            checkpoint_dict = {
                'epoch': ep,
                'model_hparams': {
                    'patch_size': config.get('model', {}).get('patch_size'),
                    'stride': config.get('model', {}).get('stride'),
                    'forecast_steps': config.get('model', {}).get('forecast_steps'),
                    'embed_dim': config.get('model', {}).get('embed_dim'),
                    'max_seq_len': config.get('model', {}).get('max_seq_len'),
                    'depth': config.get('model', {}).get('depth'),
                    'num_heads': config.get('model', {}).get('num_heads'),
                    'hidden_dim': config.get('model', {}).get('hidden_dim'),
                    'dropout': config.get('model', {}).get('dropout'),
                    # Store ablation flags so evaluation can reconstruct the model architecture.
                    'use_cnn': config.get('model', {}).get('use_cnn', True),
                    'use_transformer': config.get('model', {}).get('use_transformer', True),
                    'use_temporal': config.get('model', {}).get('use_temporal', True),
                    'use_positional': config.get('model', {}).get('use_positional', True),
                },
                'model_state_dict': current_model_state,
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                best_state = current_model_state 
                best_path = os.path.join(save_dir, f"best_forecast_{time_tag}_dim{config['model']['embed_dim']}_patch{config['model']['patch_size']}_stride{config['model']['stride']}.pt")
                torch.save(checkpoint_dict, best_path) 
            else:
                no_improve += 1
                if no_improve >= config['training']['patience']:
                    print(f"Early stopping triggered at epoch {ep}. Best validation loss: {best_val_loss:.4f}")
                    should_stop = True

            if ep % 10 == 0:
                latest_path = os.path.join(save_dir, f"latest_checkpoint_{ep}.pt")
                torch.save(checkpoint_dict, latest_path)

            log_df.append({
                "epoch": ep, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
                **{f"mae_{k}": v for k, v in grouped.items()},
                "ceg_A": ceg.get('A', 0),
                "ceg_B": ceg.get('B', 0),
                "hypo_f1": val_event_hypo.get('f1', 0),
                "hyper_f1": val_event_hyper.get('f1', 0)
            })
            pd.DataFrame(log_df).to_csv(log_file, index=False)

        stop_tensor = torch.tensor(1 if should_stop else 0, dtype=torch.int, device=accelerator.device)
        stop_tensor = accelerator.reduce(stop_tensor, reduction="max")
        if stop_tensor.item() == 1:
            break

    # Synchronize the best model weights before distributed test evaluation.
    accelerator.wait_for_everyone()
    sync_path = os.path.join(save_dir, "best_model_sync_temp.pt")

    # Every rank sees the same unsharded split length before distributed testing.
    if len(test_ds) == 0:
        if accelerator.is_main_process:
            print("\nTest split is empty; skipping test evaluation.")
            if best_path is not None:
                print(f"Best model saved to: {best_path}")
            if os.path.exists(sync_path):
                os.remove(sync_path)
            wandb.finish()
        return
    
    if best_state is not None and accelerator.is_main_process:
        torch.save(best_state, sync_path)
        
    accelerator.wait_for_everyone() 

    if os.path.exists(sync_path):
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(torch.load(sync_path, map_location=accelerator.device))
        model.eval()
        
        all_preds_test, all_targets_test, all_masks_test = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", disable=not accelerator.is_main_process):
                preds = model(batch['patches'], stride=config['model']['stride'])
                pred_gathered = accelerator.gather_for_metrics(preds)
                targets_gathered = accelerator.gather_for_metrics(batch['targets'])
                mask_gathered = accelerator.gather_for_metrics(batch['forecast_mask'])
                all_preds_test.append(pred_gathered.cpu())
                all_targets_test.append(targets_gathered.cpu())
                all_masks_test.append(mask_gathered.cpu())

        if accelerator.is_main_process:
            test_preds = torch.cat(all_preds_test, dim=0)
            test_targets = torch.cat(all_targets_test, dim=0)
            test_masks = torch.cat(all_masks_test, dim=0)
            test_metrics = compute_forecast_metrics(test_preds, test_targets, test_masks)
            print(f"\nBest model saved to: {best_path}")
            if os.path.exists(sync_path):
                os.remove(sync_path)
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CGM Pretraining FrameWork")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to the training configuration YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pretrain_forecast(cfg)
