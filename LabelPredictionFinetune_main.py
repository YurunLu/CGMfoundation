# train_finetune.py
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import time
from models.base_model import CGMBaseModel  # adjust import path
from LabelPredictionFinetune.Prediction_Finetune_model import FinetuneModel
from LabelPredictionFinetune.Prediction_Finetune_utils import CGMLabelPredictionDataset,collate_fn,save_checkpoint,load_pretrained_weights
from utils.train_utils import WarmupCosineScheduler

import os
pd.options.mode.chained_assignment = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics_class(y_true, y_pred_logits):
    """
    y_true: np array shape [B] with labels (or -1 for missing)
    y_pred_logits: np array shape [B, C] or [B] (if binary with single logit)
    """
    mask = (y_true != -1)
    if mask.sum() == 0:
        return {}
    y_true_f = y_true[mask]
    if y_pred_logits.ndim == 1 or y_pred_logits.shape[1] == 1:
        # binary
        probs = torch.sigmoid(torch.tensor(y_pred_logits[mask])).numpy().ravel()
        preds = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_true_f, probs)
        acc = accuracy_score(y_true_f, preds)
        return {'acc': acc, 'auc': auc}
    else:
        probs = torch.softmax(torch.tensor(y_pred_logits[mask]), dim=1).numpy()
        preds = probs.argmax(axis=1)
        acc = accuracy_score(y_true_f, preds)
        # multiclass AUROC requires one-vs-rest; skip if not applicable
        return {'acc': acc}

def compute_metrics_reg(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}

def train_loop(model, train_loader, val_loader=None, epochs=20, lr=1e-4, out_dir='./checkpoints',
               num_classes=None, regression_dim=None):
    """
    Unified training loop for finetuning CGM foundation model on label prediction.
    Works with datasets that provide:
      batch = {'patches': ..., 'forecast_mask': ..., 'label': ...}
    """
    model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = optim.AdamW(params, lr=lr, weight_decay=1e-7)
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=100,
                                      max_steps=epochs * len(train_loader))
    # Define losses
    cls_loss_fn = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else \
        torch.nn.CrossEntropyLoss(ignore_index=-1) if num_classes is not None else None
    reg_loss_fn = torch.nn.MSELoss() if regression_dim is not None else None

    best_val = float('inf')
    os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        all_cls_true, all_cls_pred = [], []
        all_reg_true, all_reg_pred = [], []
        losses = []

        for batch in train_loader:
            patches = batch['patches'].to(device)
            forecast_mask = batch['forecast_mask'].to(device)
            labels = batch['label'].to(device)

            out = model(patches, forecast_mask=forecast_mask)
            loss = torch.tensor(0.0, device=device)

            # --- classification ---
            if cls_loss_fn is not None and 'logits' in out:
                logits = out['logits']
                if num_classes == 1 and logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.view(-1)
                if num_classes == 1:
                    mask = (labels != -1)
                    if mask.sum() > 0:
                        l = cls_loss_fn(logits[mask].float(), labels[mask].float())
                        loss += l
                else:
                    l = cls_loss_fn(logits, labels.long())
                    loss += l
                all_cls_true.append(labels.cpu().numpy())
                all_cls_pred.append(logits.detach().cpu().numpy())

            # --- regression ---
            if reg_loss_fn is not None and 'regression' in out:
                preds = out['regression']
                if preds.ndim == 2 and preds.shape[1] == 1:
                    preds = preds.view(-1)
                if labels.ndim == 2 and labels.shape[1] == 1:
                    labels = labels.view(-1)
                l = reg_loss_fn(preds, labels)
                loss += l
                all_reg_true.append(labels.cpu().numpy())
                all_reg_pred.append(preds.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # --- compute metrics ---
        avg_train_loss = float(np.mean(losses)) if losses else 0.0

        # Regression metrics
        reg_mae, reg_r2, pcc = None, None,None
        if all_reg_true:
            y_true = np.concatenate(all_reg_true, axis=0)
            y_pred = np.concatenate(all_reg_pred, axis=0)
            reg_mae = mean_absolute_error(y_true, y_pred)
            reg_r2 = r2_score(y_true, y_pred)
            pcc = pearsonr(y_true, y_pred)[0]

        # Classification metrics
        cls_acc, cls_auc = None, None
        if all_cls_true:
            y_true = np.concatenate(all_cls_true, axis=0)
            y_pred = np.concatenate(all_cls_pred, axis=0)
            if num_classes == 1:
                # binary
                y_pred_bin = (1 / (1 + np.exp(-y_pred))) > 0.5
                cls_acc = accuracy_score(y_true, y_pred_bin)
                cls_auc = roc_auc_score(y_true, 1 / (1 + np.exp(-y_pred)))
            else:
                cls_acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))

        print(f"Epoch {epoch}: train loss={avg_train_loss:.4f}, "
              f"reg_MAE={reg_mae}, reg_R2={reg_r2}, pcc={pcc}, "
              f"cls_acc={cls_acc}, cls_auc={cls_auc}")

        # --- validation ---
        if val_loader is not None:
            val_metrics, val_loss = evaluate(model, val_loader, cls_loss_fn, reg_loss_fn, num_classes)
            print(f"  Val loss {val_loss:.4f}  metrics: {val_metrics}")
            scheduler.step()
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, f"{out_dir}/best.pt")
        else:
            scheduler.step()


def evaluate(model, loader, cls_loss_fn, reg_loss_fn, num_classes=None):
    model.eval()
    all_true = []
    all_pred = []
    losses = []

    with torch.no_grad():
        for batch in loader:
            patches = batch['patches'].to(device)
            forecast_mask = batch['forecast_mask'].to(device)
            labels = batch['label'].to(device)

            out = model(patches, forecast_mask=forecast_mask)
            loss = torch.tensor(0.0, device=device)

            if cls_loss_fn is not None and 'logits' in out:
                logits = out['logits']
                if num_classes == 1:
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        logits = logits.view(-1)
                    mask = (labels != -1)
                    if mask.sum() > 0:
                        l = cls_loss_fn(logits[mask].float(), labels[mask].float())
                        loss = loss + l
                else:
                    l = cls_loss_fn(logits, labels.long())
                    loss = loss + l
                all_true.append(labels.cpu().numpy())
                all_pred.append(logits.cpu().numpy())

            if reg_loss_fn is not None:
                target = batch['label'].to(device)
                preds = out.get('regression')
                if preds.ndim == 2 and preds.shape[1] == 1:
                    preds = preds.view(-1)
                if target.ndim == 2 and target.shape[1] == 1:
                    target = target.view(-1)
                l = reg_loss_fn(preds, target)
                loss = loss + l
                all_true.append(target.cpu().numpy())
                all_pred.append(preds.cpu().numpy())

            losses.append(loss.item())

    avg_loss = float(np.mean(losses)) if losses else 0.0
    metrics = {}
    if cls_loss_fn is not None:
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        if num_classes == 1:
            from sklearn.metrics import roc_auc_score
            metrics["AUC"] = float(roc_auc_score(y_true, torch.sigmoid(torch.tensor(y_pred)).numpy()))
        else:
            from sklearn.metrics import accuracy_score
            metrics["acc"] = float(accuracy_score(y_true, np.argmax(y_pred, axis=1)))
    if reg_loss_fn is not None:
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        from sklearn.metrics import r2_score
        from scipy.stats import pearsonr
        metrics["R2"] = float(r2_score(y_true, y_pred))
        metrics["pearsonr"] = pearsonr(y_true, y_pred)[0]
        metrics["mae"] = mean_absolute_error(y_true, y_pred)

    return metrics, avg_loss

# --- add at the bottom of your script ---
from sklearn.model_selection import KFold

def cross_validate(dataset, base_model,
                   n_splits=5, leave_one_out=False,
                   num_classes=None, regression_dim=None, epochs=20, lr=1e-4):
    """
    dataset: CGMLabelPredictionDataset
    n_splits: 5 for 5-fold
    leave_one_out: if True, use leave-one-out instead of KFold
    Returns mean PCC (regression) or mean accuracy (classification)
    """
    if leave_one_out:
        n_splits = len(dataset)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset))), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

        # reload a fresh finetune model for each fold
        finetune_fold = FinetuneModel(base_model=base_model,
                                      t_head_dim=128,
                                      t_head_depth=2,
                                      t_head_heads=4,
                                      mlp_hidden=128,
                                      num_classes=num_classes,
                                      regression_dim=regression_dim,
                                      freeze_base=False).to(device)

        train_loop(finetune_fold, train_loader, val_loader=val_loader,
                   epochs=epochs, lr=lr,
                   num_classes=num_classes,
                   regression_dim=regression_dim)

        # final evaluation on validation set
        cls_loss_fn = torch.nn.BCEWithLogitsLoss() if num_classes == 1 else \
                      torch.nn.CrossEntropyLoss(ignore_index=-1) if num_classes is not None else None
        reg_loss_fn = torch.nn.MSELoss() if regression_dim is not None else None
        metrics, _ = evaluate(finetune_fold, val_loader, cls_loss_fn, reg_loss_fn, num_classes=num_classes)
        print(f"Fold {fold} metrics: {metrics}")
        all_metrics.append(metrics)

    # compute mean metrics across folds
    mean_metrics = {}
    if num_classes is not None:
        acc_list = [m.get('acc', 0.0) for m in all_metrics]
        mean_metrics['mean_acc'] = np.mean(acc_list)
        if num_classes == 1:
            auc_list = [m.get('auc', 0.0) for m in all_metrics]
            mean_metrics['mean_auc'] = np.mean(auc_list)
        print(acc_list, f"Mean={mean_metrics}")
    if regression_dim is not None:
        for key in ['R2','pearsonr','mae']:
            m_list = [m.get(key, 0.0) for m in all_metrics]
            mean_metrics[key] = np.mean(m_list)
            print(key, m_list, f"Mean={np.mean(m_list)}")
    return mean_metrics

if __name__ == "__main__":
    # create base pretrained model
    cgm_path = "data/Zhao_data.csv"
    label_path = "data/Summary_Zhao.csv"
    target = 'cp120'
    num_classes = None
    regression_dim = 1
    type = 'reg'
    base_model = CGMBaseModel(patch_size=36, forecast_size=12, embed_dim=128, depth=4)
    checkpoint_path = "./checkpoints/best_forecast_20251012_140643_dim128_patch36_stride6.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_model.load_state_dict(checkpoint.get('model_state', checkpoint.get('state_dict', checkpoint)))

    base_model.to(device)
    base_model.eval()

    for param in base_model.parameters():
        param.requires_grad = False


    ds = CGMLabelPredictionDataset(
        cgm_path=cgm_path,
        label_path=label_path,
        target = target,
        type = type
    )
    '''
    finetune = FinetuneModel(base_model=base_model,
                             t_head_dim=128,
                             t_head_depth=1,
                             t_head_heads=4,
                             mlp_hidden=64,
                             num_classes=num_classes,
                             regression_dim=regression_dim,
                             freeze_base=False)

    loader = DataLoader(ds, batch_size=128, shuffle=True)
    train_loop(finetune, loader, val_loader=None, epochs=3000, lr=1e-4,
               num_classes=num_classes,
               regression_dim=regression_dim,
               )
    '''
    mean_metrics = cross_validate(ds, base_model = base_model,
                                  n_splits=5, leave_one_out=False,
                                  num_classes=num_classes,
                                  regression_dim=regression_dim,
                                  epochs=1000,
                                  lr=5e-5)
