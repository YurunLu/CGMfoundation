import torch
import torch.nn.functional as F
import numpy as np

def forecast_loss(pred_forecast, true_forecast, forecast_mask, 
                  hypo_thr=3.9, hyper_thr=10.0, 
                  hypo_weight=2.0, hyper_weight=1.2, normal_weight=1.0):
    """
    Clinical weighted L1 loss.
    """
    mask = forecast_mask
    base_loss = torch.abs(pred_forecast - true_forecast)
    
    weights = torch.full_like(true_forecast, normal_weight)
    weights[true_forecast < hypo_thr] = hypo_weight
    weights[true_forecast > hyper_thr] = hyper_weight
    
    weighted_loss = base_loss * weights
    loss_mae = torch.sum(weighted_loss * mask) / mask.sum().clamp_min(1.0)
    
    return loss_mae


def clarke_error_grid_analysis(ref_values, pred_values):
    """
    Compute the Clarke error grid (CEG) distribution.
    Inputs are in mmol/L and converted to mg/dL internally.
    """
    ref = np.asarray(ref_values) * 18.0
    pred = np.asarray(pred_values) * 18.0
    
    n = len(ref)
    if n == 0:
        return {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    # Vectorized zone classification.
    zone_A = (np.abs(pred - ref) <= 0.2 * ref) | ((pred <= 70) & (ref <= 70))
    zone_E = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    zone_C = (((ref >= 70) & (ref <= 290)) & (pred >= ref + 110)) | \
             (((ref >= 130) & (ref <= 180)) & (pred <= (7/5) * ref - 182))
    zone_D = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | \
             ((ref <= 70) & (pred >= 70) & (pred <= 180))

    # Keep zones mutually exclusive in A, E, C, D, B priority order.
    zone_E = zone_E & ~zone_A
    zone_C = zone_C & ~zone_A & ~zone_E
    zone_D = zone_D & ~zone_A & ~zone_E & ~zone_C
    zone_B = ~(zone_A | zone_E | zone_C | zone_D)

    return {
        "A": (np.sum(zone_A) / n) * 100,
        "B": (np.sum(zone_B) / n) * 100,
        "C": (np.sum(zone_C) / n) * 100,
        "D": (np.sum(zone_D) / n) * 100,
        "E": (np.sum(zone_E) / n) * 100
    }


@torch.no_grad()
def compute_forecast_metrics(pred, true, forecast_mask,
                             hyper_thr=10.0, hypo_thr=3.9):
    """
    Compute detailed metrics for CGM forecast evaluation.
    Includes CEG across forecast horizons and basic pointwise evaluation.
    """
    mask = forecast_mask

    # --- 1. CEG across forecast horizons ---
    # CEG over the full forecast horizon.
    valid_indices = mask > 0.5
    ceg_results = clarke_error_grid_analysis(true[valid_indices].cpu().numpy(), pred[valid_indices].cpu().numpy())

    # 10 min CEG (first 2 points).
    ceg_10min = None
    if pred.shape[-1] >= 2:
        mask_10 = mask[:, :, :2]
        valid_10 = mask_10 > 0.5
        ceg_10min = clarke_error_grid_analysis(true[:, :, :2][valid_10].cpu().numpy(), pred[:, :, :2][valid_10].cpu().numpy())

    # 30 min CEG (first 6 points).
    ceg_30min = None
    if pred.shape[-1] >= 6:
        mask_30 = mask[:, :, :6]
        valid_30 = mask_30 > 0.5
        ceg_30min = clarke_error_grid_analysis(true[:, :, :6][valid_30].cpu().numpy(), pred[:, :, :6][valid_30].cpu().numpy())

    # --- 2. Per-step & Grouped MAE ---
    abs_err = torch.abs(pred - true) * mask
    mae_per_step = torch.sum(abs_err, dim=(0,1)) / mask.sum(dim=(0,1)).clamp_min(1.0)
    
    step = 5  # 5 min per point
    groups = {
        "0-10min": (0, 10 // step),
        "10-30min": (10 // step, 30 // step),
        "30-60min": (30 // step, 60 // step)
    }
    grouped_mae = {}
    for name, (a,b) in groups.items():
        if b <= pred.shape[-1]:
            grouped_mae[name] = mae_per_step[a:b].mean().item()

    # --- 3. Pointwise abnormal glucose detection ---
    pred_hyper = (pred > hyper_thr).float()
    true_hyper = (true > hyper_thr).float()
    pred_hypo = (pred < hypo_thr).float()
    true_hypo = (true < hypo_thr).float()

    def prf(pred_bin, true_bin):
        tp = (pred_bin * true_bin * mask).sum().item()
        tn = ((1 - pred_bin) * (1 - true_bin) * mask).sum().item()
        fp = (pred_bin * (1 - true_bin) * mask).sum().item()
        fn = ((1 - pred_bin) * true_bin * mask).sum().item()
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return acc, prec, rec, f1

    hyper_acc, hyper_prec, hyper_rec, hyper_f1 = prf(pred_hyper, true_hyper)
    hypo_acc, hypo_prec, hypo_rec, hypo_f1 = prf(pred_hypo, true_hypo)

    return {
        "mae_per_step": mae_per_step.cpu().numpy(),
        "grouped_mae": grouped_mae,
        "ceg": ceg_results,
        "ceg_10min": ceg_10min,
        "ceg_30min": ceg_30min,
        "hyper": {"acc": hyper_acc, "prec": hyper_prec, "rec": hyper_rec, "f1": hyper_f1},
        "hypo": {"acc": hypo_acc, "prec": hypo_prec, "rec": hypo_rec, "f1": hypo_f1}
    }


def event_based_metrics(pred, true, mask, thr, is_hypo=True):
    """
    Window-level clinical event detection.
    True events require two consecutive abnormal points; alerts require one.
    """
    if is_hypo:
        p_flag = (pred < thr) & (mask > 0.5)
        t_flag = (true < thr) & (mask > 0.5)
    else:
        p_flag = (pred > thr) & (mask > 0.5)
        t_flag = (true > thr) & (mask > 0.5)
        
    # A true event requires two consecutive abnormal points.
    t_consec = t_flag[:, :, :-1] & t_flag[:, :, 1:]
    t_event = t_consec.any(dim=-1).float()
    
    # Trigger an alert if any predicted point crosses the threshold.
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