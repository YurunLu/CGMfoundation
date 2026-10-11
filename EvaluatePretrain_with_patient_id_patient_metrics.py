'''
python EvaluatePretrain.py \
  --checkpoint_path /workspace/JTTCGMformer/CGMfoundation_gai/output/ckp/your_all_ckp.pt \
  --csv_path /workspace/JTTCGMformer/CGMformer_loss/data/shanghai_data.csv \
  --output_dir /workspace/JTTCGMformer/CGMfoundation_gai/output/fanhuaxing/weitai1wall \
  > /workspace/JTTCGMformer/CGMfoundation_gai/output/fanhuaxing/weitai1wall/shanghai.log 2>&1 &
'''
import os
import torch
import numpy as np
import pandas as pd
import argparse
import re
from torch.utils.data import DataLoader, Dataset

from models.base_model import CGMBaseModel
from utils.loss_utils import forecast_loss
from utils.data_cleaning import clean_cgm_array, clean_cgm_dataframe

# =========================================================================
# Standalone forecast evaluation metrics.
# =========================================================================
def clarke_error_grid_analysis(ref, pred):
    """Tensor-based Clarke error grid analysis (1987)."""
    n = ref.numel()
    if n == 0: 
        return {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
    
    ref_mg = ref * 18.0
    pred_mg = pred * 18.0
    
    zone_A = (torch.abs(pred_mg - ref_mg) <= 0.2 * ref_mg) | ((pred_mg <= 70) & (ref_mg <= 70))
    zone_E = ((ref_mg <= 70) & (pred_mg >= 180)) | ((ref_mg >= 180) & (pred_mg <= 70))
    zone_C = (((ref_mg >= 70) & (ref_mg <= 290)) & (pred_mg >= ref_mg + 110)) | \
             (((ref_mg >= 130) & (ref_mg <= 180)) & (pred_mg <= (7/5) * ref_mg - 182))
    zone_D = ((ref_mg >= 240) & (pred_mg >= 70) & (pred_mg <= 180)) | \
             ((ref_mg <= 70) & (pred_mg >= 70) & (pred_mg <= 180))
             
    zone_E = zone_E & ~zone_A
    zone_C = zone_C & ~zone_A & ~zone_E
    zone_D = zone_D & ~zone_A & ~zone_E & ~zone_C
    zone_B = ~(zone_A | zone_E | zone_C | zone_D)
    
    return {
        "A": (zone_A.sum().item() / n) * 100,
        "B": (zone_B.sum().item() / n) * 100,
        "C": (zone_C.sum().item() / n) * 100,
        "D": (zone_D.sum().item() / n) * 100,
        "E": (zone_E.sum().item() / n) * 100
    }

def dts_error_grid_analysis(ref, pred):
    """DTS point error grid analysis (2024 Surveillance Error Grid)."""
    n = ref.numel()
    if n == 0: 
        return {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        
    x = ref * 18.0
    y = pred * 18.0
    
    def above_upper(y_cut, x_top):
        m = (600.0 - y_cut) / (x_top - 50.0) if x_top != 50.0 else 1e9
        return ((x <= 50) & (y > y_cut)) | ((x > 50) & (y > m * (x - 50) + y_cut))
        
    def below_lower(x_cut, y_right):
        m = (y_right - 50.0) / (600.0 - x_cut) if x_cut != 600.0 else 1e9
        return ((y <= 50) & (x > x_cut)) | ((y > 50) & (y < m * (x - x_cut) + 50))
        
    out_B_upper = above_upper(60.0, 500.0)
    out_C_upper = above_upper(86.5, 347.0)
    out_D_upper = above_upper(124.0, 241.0)
    out_E_upper = above_upper(179.0, 167.0)
    
    out_B_lower = below_lower(62.5, 480.0)
    out_C_lower = below_lower(97.5, 307.0)
    out_D_lower = below_lower(153.0, 197.0)
    out_E_lower = below_lower(238.0, 126.0)
    
    zone_E = out_E_upper | out_E_lower
    zone_D = (out_D_upper | out_D_lower) & ~zone_E
    zone_C = (out_C_upper | out_C_lower) & ~zone_E & ~zone_D
    zone_B = (out_B_upper | out_B_lower) & ~zone_E & ~zone_D & ~zone_C
    zone_A = ~(zone_E | zone_D | zone_C | zone_B)
    
    return {
        "A": (zone_A.sum().item() / n) * 100,
        "B": (zone_B.sum().item() / n) * 100,
        "C": (zone_C.sum().item() / n) * 100,
        "D": (zone_D.sum().item() / n) * 100,
        "E": (zone_E.sum().item() / n) * 100
    }

def dts_trend_accuracy_matrix(pred, true, mask, step_mins=5, gap_steps=3):
    """DTS trend accuracy matrix (2024)."""
    if pred.shape[-1] <= gap_steps: 
        return None
        
    dt = gap_steps * step_mins
    true_roc = (true[:, :, gap_steps:] - true[:, :, :-gap_steps]) * 18.0 / dt
    pred_roc = (pred[:, :, gap_steps:] - pred[:, :, :-gap_steps]) * 18.0 / dt
    true_abs_mg = true[:, :, gap_steps:] * 18.0 
    
    roc_mask = mask[:, :, gap_steps:] * mask[:, :, :-gap_steps]
    valid = roc_mask > 0.5
    if not valid.any(): 
        return None
        
    t_roc = true_roc[valid]
    p_roc = pred_roc[valid]
    t_abs = true_abs_mg[valid]
    
    def discretize_roc_pt(roc):
        bins = torch.zeros_like(roc, dtype=torch.long)
        bins[(roc >= -2.0) & (roc < -1.0)] = 1
        bins[(roc >= -1.0) & (roc <= 1.0)] = 2
        bins[(roc > 1.0) & (roc <= 2.0)] = 3
        bins[roc > 2.0] = 4
        return bins
        
    t_bins = discretize_roc_pt(t_roc)
    p_bins = discretize_roc_pt(p_roc)
    
    map_low = torch.tensor([[1, 2, 2, 4, 4], [3, 1, 2, 4, 4], [5, 3, 1, 2, 2], [5, 5, 3, 1, 2], [5, 5, 5, 3, 1]], device=pred.device)
    map_normal = torch.tensor([[1, 2, 2, 4, 4], [3, 1, 2, 4, 4], [3, 3, 1, 2, 2], [5, 3, 3, 1, 2], [5, 3, 3, 3, 1]], device=pred.device)
    map_high = torch.tensor([[1, 2, 2, 4, 4], [3, 1, 2, 4, 4], [3, 3, 1, 2, 2], [4, 3, 3, 1, 2], [4, 3, 3, 3, 1]], device=pred.device)
    
    risks = torch.zeros_like(t_bins)
    mask_low = t_abs < 100.0
    mask_high = t_abs > 180.0
    mask_normal = ~(mask_low | mask_high)
    
    if mask_low.any(): risks[mask_low] = map_low[p_bins[mask_low], t_bins[mask_low]]
    if mask_normal.any(): risks[mask_normal] = map_normal[p_bins[mask_normal], t_bins[mask_normal]]
    if mask_high.any(): risks[mask_high] = map_high[p_bins[mask_high], t_bins[mask_high]]
    
    total = t_bins.numel()
    return {
        1: (torch.sum(risks==1).item() / total) * 100, 
        2: (torch.sum(risks==2).item() / total) * 100,
        3: (torch.sum(risks==3).item() / total) * 100, 
        4: (torch.sum(risks==4).item() / total) * 100,
        5: (torch.sum(risks==5).item() / total) * 100
    }

def eval_event_based_metrics(pred, true, mask, thr, is_hypo=True):
    """Evaluate window alerts using consecutive observed target points."""
    if pred.shape[-1] < 2: 
        return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
        
    if is_hypo:
        p_flag = (pred < thr) & (mask > 0.5)
        t_flag = (true < thr) & (mask > 0.5)
    else:
        p_flag = (pred > thr) & (mask > 0.5)
        t_flag = (true > thr) & (mask > 0.5)
        
    t_consec = t_flag[:, :, :-1] & t_flag[:, :, 1:]
    t_event = t_consec.any(dim=-1).float()
    p_event = p_flag.any(dim=-1).float()
    
    window_mask = (mask.sum(dim=-1) >= 2).float() 
    tp = (p_event * t_event * window_mask).sum().item()
    tn = ((1 - p_event) * (1 - t_event) * window_mask).sum().item()
    fp = (p_event * (1 - t_event) * window_mask).sum().item()
    fn = ((1 - p_event) * t_event * window_mask).sum().item()
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

def eval_relaxed_event_metrics(pred, true, mask, thr, is_hypo=True):
    """Evaluate window alerts using any observed threshold crossing."""
    if pred.shape[-1] < 1: 
        return {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0}
        
    if is_hypo:
        p_flag = (pred < thr) & (mask > 0.5)
        t_flag = (true < thr) & (mask > 0.5)
    else:
        p_flag = (pred > thr) & (mask > 0.5)
        t_flag = (true > thr) & (mask > 0.5)
        
    t_event = t_flag.any(dim=-1).float()
    p_event = p_flag.any(dim=-1).float()
    
    window_mask = (mask.sum(dim=-1) >= 1).float() 
    tp = (p_event * t_event * window_mask).sum().item()
    tn = ((1 - p_event) * (1 - t_event) * window_mask).sum().item()
    fp = (p_event * (1 - t_event) * window_mask).sum().item()
    fn = ((1 - p_event) * t_event * window_mask).sum().item()
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

@torch.no_grad()
def compute_advanced_eval_metrics(pred, true, mask, last_true=None, last_mask=None, hyper_thr=10.0, hypo_thr=3.9):
    """Compute forecast metrics across the available horizons."""
    if last_true is not None and last_mask is not None:
        trend_p = torch.cat([last_true, pred], dim=-1)
        trend_t = torch.cat([last_true, true], dim=-1)
        trend_m = torch.cat([last_mask, mask], dim=-1)
        offset = 1 
    else:
        trend_p = pred
        trend_t = true
        trend_m = mask
        offset = 0

    def pearson_corr_flat(pred_values, true_values, eps=1e-12):
        if pred_values.numel() < 2:
            return 0.0

        p = pred_values.float()
        t = true_values.float()
        p_centered = p - p.mean()
        t_centered = t - t.mean()
        denom = torch.sqrt((p_centered ** 2).sum() * (t_centered ** 2).sum())
        if denom <= eps:
            return 0.0
        return ((p_centered * t_centered).sum() / denom).item()
    
    def compute_slice(start_idx, end_idx):
        if end_idx > pred.shape[-1]: 
            end_idx = pred.shape[-1]
        if start_idx >= end_idx: 
            return None
            
        p_slice = pred[:, :, start_idx:end_idx]
        t_slice = true[:, :, start_idx:end_idx]
        m_slice = mask[:, :, start_idx:end_idx]
        
        valid = m_slice > 0.5
        if not valid.any(): 
            return None
            
        t_val = t_slice[valid]
        p_val = p_slice[valid]
        
        ceg = clarke_error_grid_analysis(t_val, p_val)
        dts_pt = dts_error_grid_analysis(t_val, p_val)
        pearson_r = pearson_corr_flat(p_val, t_val)
        
        t_start = max(0, start_idx + offset - 3)
        t_end = end_idx + offset
        tr_p_slice = trend_p[:, :, t_start:t_end]
        tr_t_slice = trend_t[:, :, t_start:t_end]
        tr_m_slice = trend_m[:, :, t_start:t_end]
        
        dts_tr = dts_trend_accuracy_matrix(tr_p_slice, tr_t_slice, tr_m_slice, gap_steps=3)
        mae = ((torch.abs(p_slice - t_slice) * m_slice).sum() / m_slice.sum().clamp_min(1.0)).item()
        
        ev_hyper = eval_event_based_metrics(p_slice, t_slice, m_slice, thr=hyper_thr, is_hypo=False)
        ev_hypo = eval_event_based_metrics(p_slice, t_slice, m_slice, thr=hypo_thr, is_hypo=True)
        rel_hyper = eval_relaxed_event_metrics(p_slice, t_slice, m_slice, thr=hyper_thr, is_hypo=False)
        rel_hypo = eval_relaxed_event_metrics(p_slice, t_slice, m_slice, thr=hypo_thr, is_hypo=True)
        
        return {
            "mae": mae, 
            "pearson_r": pearson_r,
            "valid_points": int(valid.sum().item()),
            "ceg": ceg, 
            "dts_point": dts_pt, 
            "dts_trend": dts_tr, 
            "ev_hyper": ev_hyper, 
            "ev_hypo": ev_hypo, 
            "rel_hyper": rel_hyper, 
            "rel_hypo": rel_hypo
        }

    max_steps = pred.shape[-1]
    step_mins = 5
    results_dict = {}

    # Short forecast horizons.
    if max_steps >= 3: 
        results_dict["Very short-term forecast (0-15 min)"] = compute_slice(0, 3)
    if max_steps >= 6: 
        results_dict["Short-term forecast (0-30 min)"] = compute_slice(0, 6)
    if max_steps >= 12: 
        results_dict["Forecast interval (30-60 min)"] = compute_slice(6, 12)
        # Include the cumulative 0-60 minute horizon.
        results_dict["1h forecast horizon (0-60 min)"] = compute_slice(0, 12) 
    
    # Longer forecast horizons.
    if max_steps > 12:
        results_dict["Forecast interval (1h - 2h)"] = compute_slice(12, 24)
        results_dict["Cumulative horizon (0 - 2h)"] = compute_slice(0, 24)
    if max_steps > 24:
        results_dict["Forecast interval (2h - 3h)"] = compute_slice(24, 36)
        results_dict["Cumulative horizon (0 - 3h)"] = compute_slice(0, 36)
        
    results_dict[f"Full forecast horizon (0 - {(max_steps*step_mins)//60}h)"] = compute_slice(0, max_steps)
    
    return results_dict

class ConsistentEvalDataset(Dataset):
    def __init__(self, df, patch_size, stride, forecast_steps):
        self.df = df
        self.patch_size = patch_size
        self.stride = stride
        self.forecast_steps = forecast_steps
        self.window_size = patch_size + forecast_steps

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_raw = clean_cgm_array(self.df.iloc[idx].to_numpy(dtype=np.float32))
        orig_mask = (~np.isnan(seq_raw)).astype(np.float32)
        
        s = pd.Series(seq_raw)
        seq_filled = s.ffill().fillna(5.5).to_numpy(dtype=np.float32)
        
        seq_t = torch.tensor(seq_filled)
        mask_t = torch.tensor(orig_mask)
        
        seq_windows = seq_t.unfold(0, self.window_size, self.stride).contiguous()
        mask_windows = mask_t.unfold(0, self.window_size, self.stride).contiguous()
        
        return (
            seq_windows[:, :self.patch_size], 
            mask_windows[:, :self.patch_size], 
            seq_windows[:, self.patch_size:], 
            mask_windows[:, self.patch_size:]
        )

@torch.no_grad()
def evaluate_model(model, loader, device, stride):
    model.eval()
    all_preds, all_targets, all_masks, all_t0_true, all_t0_mask = [], [], [], [], [] 
    
    for patches, patch_mask, targets, forecast_mask in loader:
        patches = patches.to(device)
        targets = targets.to(device)
        forecast_mask = forecast_mask.to(device)
        patch_mask = patch_mask.to(device)
        
        preds = model(patches, stride=stride) 
        
        t0_true = patches[:, :, -1:]
        t0_mask = patch_mask[:, :, -1:] 
        
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_masks.append(forecast_mask.cpu())
        all_t0_true.append(t0_true.cpu())
        all_t0_mask.append(t0_mask.cpu()) 
    
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    t0_true = torch.cat(all_t0_true, dim=0)
    t0_mask = torch.cat(all_t0_mask, dim=0)
    
    # Aggregate errors at each patch position.
    se = torch.abs(preds - targets) * masks
    rse = (torch.abs(preds - targets) / targets.clamp_min(1e-5)) * masks
    mae_per_patch = se.sum(dim=(0, 2)) / masks.sum(dim=(0, 2)).clamp_min(1.0)
    rmae_per_patch = rse.sum(dim=(0, 2)) / masks.sum(dim=(0, 2)).clamp_min(1.0)
    
    loss = forecast_loss(preds, targets, masks)
    results_adv = compute_advanced_eval_metrics(preds, targets, masks, last_true=t0_true, last_mask=t0_mask)
    
    overall_mae = (torch.abs(preds - targets) * masks).sum() / masks.sum().clamp_min(1.0)
    overall_mard = (torch.abs(preds - targets) / targets.clamp_min(1e-5) * masks).sum() / masks.sum().clamp_min(1.0) * 100.0
    
    valid_preds = preds[masks > 0.5].numpy()
    valid_targets = targets[masks > 0.5].numpy()
    pearson_corr = np.corrcoef(valid_preds, valid_targets)[0, 1] if len(valid_preds) > 1 else 0.0
    
    print(f"\n[Eval Result] Overall Loss: {loss.item():.4f}")
    print(f"📊 Global Performance: MAE={overall_mae:.3f} | MARD={overall_mard:.2f}% | R={pearson_corr:.4f}")
    
    print("\n" + "="*85)
    print("🌟 [Nature-Standard] CGM Foundation Model Multi-Dimensional Evaluation Report 🌟")
    print("="*85)
    
    for name, s in results_adv.items():
        if not s: continue
        print(f"\n⏳ [{name}]")
        print(f"  ▶ Absolute error (MAE): {s['mae']:.3f}")
        print(f"  ▶ Pooled correlation (Pearson R): {s['pearson_r']:.4f} | Valid points: {s['valid_points']}")
        
        ceg = s["ceg"]
        print(f"  ▶ [CEG 1987] Clarke Error Grid:")
        print(f"    A: {ceg['A']:>6.2f}% | B: {ceg['B']:>6.2f}% | C: {ceg['C']:>6.2f}% | D: {ceg['D']:>6.2f}% | E: {ceg['E']:>6.2f}%")
        
        d_pt = s["dts_point"]
        print(f"  ▶ [DTS 2024] Point Accuracy:")
        print(f"    A: {d_pt['A']:>6.2f}% | B: {d_pt['B']:>6.2f}% | C: {d_pt['C']:>6.2f}% | D: {d_pt['D']:>6.2f}% | E: {d_pt['E']:>6.2f}%")
        
        if s["dts_trend"]:
            d_tr = s["dts_trend"]
            print(f"  ▶ [DTS 2024] Trend Accuracy:")
            print(f"    Cat 1: {d_tr[1]:>6.2f}% | Cat 2: {d_tr[2]:>6.2f}% | Cat 3: {d_tr[3]:>6.2f}% | Cat 4: {d_tr[4]:>6.2f}% | Cat 5: {d_tr[5]:>6.2f}%")
        
        rel_hypo = s["rel_hypo"]
        rel_hyper = s["rel_hyper"]
        print(f"  ▶ Clinical Alert (Relaxed):")
        print(f"    Hypo(<3.9) F1: {rel_hypo['f1']:>5.3f} (Rec: {rel_hypo['rec']*100:.1f}%) | Hyper(>10) F1: {rel_hyper['f1']:>5.3f} (Rec: {rel_hyper['rec']*100:.1f}%)")
        
        ev_hypo = s["ev_hypo"]
        ev_hyper = s["ev_hyper"]
        print(f"  ▶ Clinical Alert (Strict):")
        print(f"    Hypo(<3.9) F1: {ev_hypo['f1']:>5.3f} (Rec: {ev_hypo['rec']*100:.1f}%) | Hyper(>10) F1: {ev_hyper['f1']:>5.3f} (Rec: {ev_hyper['rec']*100:.1f}%)")
    
    print("="*85 + "\n")

    time_slice_metrics = []
    for name, s in results_adv.items():
        if not s:
            continue
        row = {
            "Time_Slice": name,
            "MAE": s["mae"],
            "Pearson_R": s["pearson_r"],
            "Valid_Points": s["valid_points"],
        }

        for prefix, metric_dict in [
            ("CEG", s["ceg"]),
            ("DTS_Point", s["dts_point"]),
            ("Event_Hypo", s["ev_hypo"]),
            ("Event_Hyper", s["ev_hyper"]),
            ("Relaxed_Hypo", s["rel_hypo"]),
            ("Relaxed_Hyper", s["rel_hyper"]),
        ]:
            for k, v in metric_dict.items():
                row[f"{prefix}_{k}"] = v

        if s["dts_trend"]:
            for k, v in s["dts_trend"].items():
                row[f"DTS_Trend_Cat{k}"] = v

        time_slice_metrics.append(row)
    
    # Return patch metrics and raw predictions.
    return {
        'mae_per_patch': mae_per_patch.numpy(), 
        'rmae_per_patch': rmae_per_patch.numpy(), 
        'time_slice_metrics': time_slice_metrics,
        'raw_preds': preds, 
        'raw_targets': targets, 
        'raw_masks': masks
    }


# Patient-level metric helpers.
# Summarize each patient row alongside the global evaluation.
# These tables support patient-level filtering, plots, and case analysis.
# Outputs:
#   patient_level_metrics.csv: one row per patient with overall and horizon metrics.
#   patient_time_slice_metrics.csv: detailed metrics for each patient and horizon.

def _safe_pearson_torch(pred_values, true_values, eps=1e-12):
    """Compute Pearson R safely for one-dimensional tensors."""
    if pred_values.numel() < 2 or true_values.numel() < 2:
        return 0.0
    p = pred_values.float()
    t = true_values.float()
    p_centered = p - p.mean()
    t_centered = t - t.mean()
    denom = torch.sqrt((p_centered ** 2).sum() * (t_centered ** 2).sum())
    if denom <= eps:
        return 0.0
    return ((p_centered * t_centered).sum() / denom).item()


def _basic_slice_metrics(p_slice, t_slice, m_slice):
    """Compute basic regression metrics for one patient and forecast horizon."""
    valid = m_slice > 0.5
    valid_points = int(valid.sum().item())

    if valid_points == 0:
        return {
            "Valid_Points": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MARD": np.nan,
            "Pearson_R": np.nan,
            "Bias": np.nan,
            "True_Mean": np.nan,
            "Pred_Mean": np.nan,
            "True_SD": np.nan,
            "Pred_SD": np.nan,
        }, None, None

    t_val = t_slice[valid]
    p_val = p_slice[valid]
    err = p_val - t_val
    abs_err = torch.abs(err)

    return {
        "Valid_Points": valid_points,
        "MAE": abs_err.mean().item(),
        "RMSE": torch.sqrt((err ** 2).mean()).item(),
        "MARD": (abs_err / t_val.clamp_min(1e-5)).mean().item() * 100.0,
        "Pearson_R": _safe_pearson_torch(p_val, t_val),
        "Bias": err.mean().item(),
        "True_Mean": t_val.mean().item(),
        "Pred_Mean": p_val.mean().item(),
        "True_SD": t_val.std(unbiased=False).item() if valid_points > 1 else 0.0,
        "Pred_SD": p_val.std(unbiased=False).item() if valid_points > 1 else 0.0,
    }, t_val, p_val


def _add_prefixed_metrics(row, prefix, metrics):
    """Add metrics to a summary row with a shared prefix."""
    for k, v in metrics.items():
        row[f"{prefix}_{k}"] = v


def compute_patient_level_metrics(preds, targets, masks, patient_ids, hyper_thr=10.0, hypo_thr=3.9):
    """
    preds/targets/masks: torch.Tensor, shape = [num_patients, num_windows, forecast_steps]
    patient_ids: patient IDs in the same order as the first tensor dimension
    """
    preds = preds.cpu()
    targets = targets.cpu()
    masks = masks.cpu()

    num_patients, num_windows, num_steps = preds.shape

    # Match the horizons used by compute_advanced_eval_metrics.
    time_slices = []
    if num_steps >= 3:
        time_slices.append(("0-15min", 0, 3))
    if num_steps >= 6:
        time_slices.append(("0-30min", 0, 6))
    if num_steps >= 12:
        time_slices.append(("30-60min", 6, 12))
        time_slices.append(("0-60min", 0, 12))
    if num_steps > 12:
        time_slices.append(("1h-2h", 12, min(24, num_steps)))
        time_slices.append(("0-2h", 0, min(24, num_steps)))
    if num_steps > 24:
        time_slices.append(("2h-3h", 24, min(36, num_steps)))
        time_slices.append(("0-3h", 0, min(36, num_steps)))
    time_slices.append((f"0-{num_steps * 5}min_full", 0, num_steps))

    patient_rows = []
    patient_time_slice_rows = []

    for i in range(num_patients):
        pid = patient_ids[i]
        p_i = preds[i:i+1]
        t_i = targets[i:i+1]
        m_i = masks[i:i+1]

        row = {
            "Patient_ID": pid,
            "Patient_Row_Index": i,
            "Num_Windows": num_windows,
            "Forecast_Steps": num_steps,
        }

        # Overall metrics across all forecast steps for this patient.
        overall_metrics, overall_t_val, overall_p_val = _basic_slice_metrics(p_i, t_i, m_i)
        _add_prefixed_metrics(row, "Overall", overall_metrics)

        if overall_t_val is not None:
            ceg = clarke_error_grid_analysis(overall_t_val, overall_p_val)
            dts_pt = dts_error_grid_analysis(overall_t_val, overall_p_val)
            ev_hypo = eval_event_based_metrics(p_i, t_i, m_i, thr=hypo_thr, is_hypo=True)
            ev_hyper = eval_event_based_metrics(p_i, t_i, m_i, thr=hyper_thr, is_hypo=False)
            rel_hypo = eval_relaxed_event_metrics(p_i, t_i, m_i, thr=hypo_thr, is_hypo=True)
            rel_hyper = eval_relaxed_event_metrics(p_i, t_i, m_i, thr=hyper_thr, is_hypo=False)
        else:
            ceg = {k: np.nan for k in ["A", "B", "C", "D", "E"]}
            dts_pt = {k: np.nan for k in ["A", "B", "C", "D", "E"]}
            ev_hypo = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}
            ev_hyper = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}
            rel_hypo = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}
            rel_hyper = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}

        _add_prefixed_metrics(row, "Overall_CEG", ceg)
        _add_prefixed_metrics(row, "Overall_DTS_Point", dts_pt)
        _add_prefixed_metrics(row, "Overall_Event_Hypo", ev_hypo)
        _add_prefixed_metrics(row, "Overall_Event_Hyper", ev_hyper)
        _add_prefixed_metrics(row, "Overall_Relaxed_Hypo", rel_hypo)
        _add_prefixed_metrics(row, "Overall_Relaxed_Hyper", rel_hyper)

        # Metrics for each forecast horizon of this patient.
        for slice_name, start_idx, end_idx in time_slices:
            if start_idx >= num_steps:
                continue
            end_idx = min(end_idx, num_steps)

            p_slice = p_i[:, :, start_idx:end_idx]
            t_slice = t_i[:, :, start_idx:end_idx]
            m_slice = m_i[:, :, start_idx:end_idx]

            slice_metrics, t_val, p_val = _basic_slice_metrics(p_slice, t_slice, m_slice)

            ts_row = {
                "Patient_ID": pid,
                "Patient_Row_Index": i,
                "Time_Slice": slice_name,
                "Start_Step_1Based": start_idx + 1,
                "End_Step_1Based": end_idx,
                "Start_Min": start_idx * 5,
                "End_Min": end_idx * 5,
                "Num_Windows": num_windows,
                "Forecast_Steps": num_steps,
                **slice_metrics,
            }

            if t_val is not None:
                slice_ceg = clarke_error_grid_analysis(t_val, p_val)
                slice_dts = dts_error_grid_analysis(t_val, p_val)
                slice_ev_hypo = eval_event_based_metrics(p_slice, t_slice, m_slice, thr=hypo_thr, is_hypo=True)
                slice_ev_hyper = eval_event_based_metrics(p_slice, t_slice, m_slice, thr=hyper_thr, is_hypo=False)
                slice_rel_hypo = eval_relaxed_event_metrics(p_slice, t_slice, m_slice, thr=hypo_thr, is_hypo=True)
                slice_rel_hyper = eval_relaxed_event_metrics(p_slice, t_slice, m_slice, thr=hyper_thr, is_hypo=False)
            else:
                slice_ceg = {k: np.nan for k in ["A", "B", "C", "D", "E"]}
                slice_dts = {k: np.nan for k in ["A", "B", "C", "D", "E"]}
                slice_ev_hypo = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}
                slice_ev_hyper = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}
                slice_rel_hypo = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}
                slice_rel_hyper = {k: np.nan for k in ["acc", "prec", "rec", "f1"]}

            _add_prefixed_metrics(ts_row, "CEG", slice_ceg)
            _add_prefixed_metrics(ts_row, "DTS_Point", slice_dts)
            _add_prefixed_metrics(ts_row, "Event_Hypo", slice_ev_hypo)
            _add_prefixed_metrics(ts_row, "Event_Hyper", slice_ev_hyper)
            _add_prefixed_metrics(ts_row, "Relaxed_Hypo", slice_rel_hypo)
            _add_prefixed_metrics(ts_row, "Relaxed_Hyper", slice_rel_hyper)

            patient_time_slice_rows.append(ts_row)

            # Also add basic horizon metrics to the patient summary table.
            safe_slice_name = slice_name.replace("-", "_").replace(" ", "").replace("/", "_")
            _add_prefixed_metrics(row, safe_slice_name, slice_metrics)

        patient_rows.append(row)

    return pd.DataFrame(patient_rows), pd.DataFrame(patient_time_slice_rows)
# =================================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_outputs")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--forecast_steps", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--use_cnn", action="store_true", default=None)
    parser.add_argument("--use_transformer", action="store_true", default=None)
    args = parser.parse_args()

    # Load the checkpoint and resolve model settings.
    ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    h = ckpt.get("model_hparams", {}) if isinstance(ckpt, dict) else {}
    
    def pick_int(arg_val, key, default): 
        return int(arg_val) if arg_val is not None else int(h.get(key, default))
        
    def pick_flag(arg_val, key, default=True): 
        return bool(arg_val) if arg_val is not None else (bool(h[key]) if key in h else default)

    patch_size = pick_int(args.patch_size, "patch_size", 144)
    stride = pick_int(args.stride, "stride", 3)
    forecast_steps = pick_int(args.forecast_steps, "forecast_steps", 12)
    embed_dim = pick_int(args.embed_dim, "embed_dim", 256)
    
    use_cnn = pick_flag(args.use_cnn, "use_cnn", True)
    use_transformer = pick_flag(args.use_transformer, "use_transformer", True)
    use_temporal = pick_flag(None, "use_temporal", True)
    use_positional = pick_flag(None, "use_positional", True)

    # Log model settings, including temporal and positional encoding.
    print("\n" + "="*85)
    print(f"✅ Loaded foundation model weights: {os.path.basename(args.checkpoint_path)}")
    print(f"   Path: {args.checkpoint_path}")
    print("-" * 85)
    print(f"🛠️  Model parameters:")
    print(f"    - Patch Size:     {patch_size:>5} | Stride:         {stride:>5}")
    print(f"    - Forecast Steps: {forecast_steps:>5} | Embed Dim:      {embed_dim:>5}")
    print(f"    - Device:         {args.device.upper()}")
    print("-" * 85)
    print(f"⚙️  Architecture flags:")
    print(f"    - CNN Encoder:      {use_cnn}")
    print(f"    - Transformer:      {use_transformer}")
    print(f"    - Positional Embed: {use_positional}")
    print(f"    - Temporal Embed:   {use_temporal}")
    print("-" * 85)

    model = CGMBaseModel(
        patch_size=patch_size, 
        forecast_size=forecast_steps, 
        embed_dim=embed_dim, 
        max_seq_len=3456, 
        depth=pick_int(None, "depth", 12), 
        num_heads=pick_int(None, "num_heads", 8), 
        hidden_dim=pick_int(None, "hidden_dim", 1024), 
        use_cnn=use_cnn, 
        use_transformer=use_transformer,
        use_temporal=use_temporal,
        use_positional=use_positional
    ).to(args.device)
    
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total model parameters: {total_params / 1e6:.2f} M ({total_params} parameters)")
    print("="*85 + "\n")

    # Prepare the evaluation data.
    df_raw = pd.read_csv(args.csv_path, index_col=0)

    # Preserve patient IDs before selecting CGM columns.
    # Keep patient IDs for raw and flattened prediction outputs.
    # Accept either of these patient ID formats:
    #   Prefer an explicit "id" column when present.
    #   Otherwise, use the first CSV column loaded as the DataFrame index.
    if "id" in df_raw.columns:
        patient_ids_raw = df_raw["id"].to_numpy()
        df = df_raw.drop(columns=["id"])
    else:
        patient_ids_raw = df_raw.index.to_numpy()
        df = df_raw
    # =================================================================================================

    col_map = {col: int(re.findall(r'\d+', str(col))[0]) for col in df.columns if re.findall(r'\d+', str(col))}
    sorted_cols = [k for k, v in sorted(col_map.items(), key=lambda item: item[1])]
    df = df[sorted_cols]

    standard_cols = [str(x) for x in range(1, 3456 + 1)]
    df = df.iloc[:, :3456] if len(df.columns) > 3456 else df
    df.columns = [str(i+1) for i in range(len(df.columns))]
    df = df.reindex(columns=standard_cols)
    df = clean_cgm_dataframe(df)

    min_req = patch_size + forecast_steps
    valid_patient_mask = df.notna().sum(axis=1) >= min_req
    df_filtered = df[valid_patient_mask]

    # Apply the same patient filter to IDs and CGM rows.
    patient_ids_filtered = np.asarray(patient_ids_raw, dtype=object)[valid_patient_mask.to_numpy()]
    # =================================================================================================

    if len(df_filtered) == 0: 
        raise ValueError("🚨 Error: No valid sequences meet the evaluation requirements!")

    loader = DataLoader(
        ConsistentEvalDataset(df_filtered, patch_size, stride, forecast_steps), 
        batch_size=args.batch_size, 
        shuffle=False
    )

    # Run evaluation.
    results = evaluate_model(model, loader, args.device, stride=stride)
    
    # Save evaluation outputs.
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results['mae_per_patch'], columns=['MAE']).to_csv(os.path.join(args.output_dir, 'mae_per_patch.csv'), index=False)
    pd.DataFrame(results['rmae_per_patch'], columns=['RMAE']).to_csv(os.path.join(args.output_dir, 'rmae_per_patch.csv'), index=False)
    pd.DataFrame(results['time_slice_metrics']).to_csv(os.path.join(args.output_dir, 'time_slice_metrics.csv'), index=False)
    
    # Save filtered patient IDs with the prediction tensors.
    torch.save(
        {
            'preds': results['raw_preds'], 
            'targets': results['raw_targets'], 
            'masks': results['raw_masks'],
            'patient_ids': patient_ids_filtered.tolist()
        }, 
        os.path.join(args.output_dir, 'raw_predictions.pt')
    )

    # Map the first tensor dimension to patient IDs.
    pd.DataFrame({
        'Patient_Row_Index': np.arange(len(patient_ids_filtered)),
        'Patient_ID': patient_ids_filtered
    }).to_csv(os.path.join(args.output_dir, 'patient_ids.csv'), index=False)
    # =================================================================================================

    # Export valid predictions with patient, window, and step identifiers.
    # Tensor layout and export rules:
    #   raw_preds/raw_targets/raw_masks have shape [patients, windows, forecast steps].
    #   Export only observed target points with mask > 0.5.
    #   Include patient IDs and window/step indices for traceability.
    raw_preds_np = results['raw_preds'].numpy()
    raw_targets_np = results['raw_targets'].numpy()
    raw_masks_np = results['raw_masks'].numpy()

    num_patients, num_windows, num_steps = raw_preds_np.shape
    valid_idx_np = raw_masks_np > 0.5

    patient_id_grid = np.repeat(np.asarray(patient_ids_filtered, dtype=object), num_windows * num_steps)
    window_idx_grid = np.tile(np.repeat(np.arange(num_windows), num_steps), num_patients)
    step_idx_grid = np.tile(np.arange(1, num_steps + 1), num_patients * num_windows)

    valid_flat = valid_idx_np.reshape(-1)
    pd.DataFrame({
        'Patient_ID': patient_id_grid[valid_flat],
        'Window_Index': window_idx_grid[valid_flat],
        'Step_Index': step_idx_grid[valid_flat],
        'True_Value': raw_targets_np.reshape(-1)[valid_flat], 
        'Predicted_Value': raw_preds_np.reshape(-1)[valid_flat],
        'Mask': raw_masks_np.reshape(-1)[valid_flat]
    }).to_csv(os.path.join(args.output_dir, 'flattened_predictions.csv'), index=False)
    # =================================================================================================


    # Save patient-level metric tables.
    # patient_level_metrics.csv: one row per patient for sorting and filtering.
    # patient_time_slice_metrics.csv: one row per patient and horizon for plots and cases.
    patient_level_df, patient_time_slice_df = compute_patient_level_metrics(
        preds=results['raw_preds'],
        targets=results['raw_targets'],
        masks=results['raw_masks'],
        patient_ids=patient_ids_filtered,
        hyper_thr=10.0,
        hypo_thr=3.9
    )

    patient_level_df.to_csv(os.path.join(args.output_dir, 'patient_level_metrics.csv'), index=False)
    patient_time_slice_df.to_csv(os.path.join(args.output_dir, 'patient_time_slice_metrics.csv'), index=False)

    print(f"📌 Patient-level metrics saved: {os.path.join(args.output_dir, 'patient_level_metrics.csv')}")
    print(f"📌 Patient-horizon metrics saved: {os.path.join(args.output_dir, 'patient_time_slice_metrics.csv')}")
    # =================================================================================================

    print(f"💾 All evaluation outputs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
