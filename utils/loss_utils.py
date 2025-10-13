import torch
import torch.nn.functional as F

def forecast_loss(pred_forecast, true_forecast, forecast_mask, delta=0.2):
    """
    Smooth L1 (Huber) loss or MAE loss on forecast predictions.
    pred_forecast, true_forecast: [B, N, P]
    forecast_mask: [B, N, P]
    """
    mask = forecast_mask.unsqueeze(-1).expand_as(true_forecast)
    loss_mae = F.l1_loss(pred_forecast * mask, true_forecast * mask, reduction='sum') / mask.sum().clamp_min(1.0)
    return loss_mae


@torch.no_grad()
def compute_forecast_metrics(pred, true, forecast_mask,
                             hyper_thr=10.0, hypo_thr=3.9):
    """
    Compute detailed metrics for CGM forecast evaluation.
    Args:
        pred, true: [B, N, P]
        forecast_mask: [B, N, P]
    Returns:
        dict with per-step MAE, grouped MAE, and hyper/hypo detection metrics.
    """
    mask = forecast_mask.unsqueeze(-1).expand_as(true)
    pred, true = pred * mask, true * mask

    # --- Per-step MAE ---
    mae_per_step = torch.sum(torch.abs(pred - true), dim=(0,1)) / mask.sum(dim=(0,1)).clamp_min(1.0)

    # --- Grouped MAE ---
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

    # --- Hyper/Hypo detection ---
    pred_hyper = (pred > hyper_thr).float()
    true_hyper = (true > hyper_thr).float()
    pred_hypo = (pred < hypo_thr).float()
    true_hypo = (true < hypo_thr).float()

    def prf(pred_bin, true_bin):
        tp = (pred_bin * true_bin).sum().item()
        tn = ((1 - pred_bin) * (1 - true_bin)).sum().item()
        fp = (pred_bin * (1 - true_bin)).sum().item()
        fn = ((1 - pred_bin) * true_bin).sum().item()
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
        "hyper": {"acc": hyper_acc, "prec": hyper_prec, "rec": hyper_rec, "f1": hyper_f1},
        "hypo": {"acc": hypo_acc, "prec": hypo_prec, "rec": hypo_rec, "f1": hypo_f1}
    }
