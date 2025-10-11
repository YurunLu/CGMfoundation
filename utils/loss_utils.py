import torch
import torch.nn.functional as F

def forecast_loss(pred_forecast, true_forecast, forecast_mask, delta=0.2):
    """
    Smooth L1 (Huber) loss on forecast predictions.
    pred_forecast, true_forecast: [B, N, P]
    forecast_mask: [B, N, P]
    """
    #diff = pred_forecast - true_forecast
    #mask = forecast_mask.bool()
    #loss = F.smooth_l1_loss(pred_forecast[mask], true_forecast[mask], beta=delta)
    #loss = F.l1_loss(pred_forecast[mask], true_forecast[mask])
    mask = forecast_mask.unsqueeze(-1).expand_as(true_forecast)
    loss = F.l1_loss(pred_forecast * mask, true_forecast * mask, reduction='sum') / mask.sum().clamp_min(1.0)
    return loss
