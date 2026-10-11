import os
import torch
import numpy as np
import pandas as pd
import argparse
import re
from tqdm import tqdm
from scipy.stats import pearsonr
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.base_model import CGMBaseModel
from utils.data_cleaning import clean_cgm_array
'''
nohup env CUDA_VISIBLE_DEVICES=4 python Eval_AR_24h_FullContext_clamped.py \
  --csv_path /your/data.csv \
  --checkpoint_path /your/model.pt \
  --output_dir /your/output \
  --ar_step_size 3 \
  --min_history_hours 72 > ar_eval.log 2>&1 &
  
nohup python Eval_AR_24h_FullContext_clamped.py \
  --csv_path /workspace/JTTCGMformer/CGMfoundation_gai/data/shanghai_data.csv \
  --checkpoint_path /workspace/JTTCGMformer/CGMfoundation_gai/output/ckp/your_model.pt \
  --output_dir /workspace/JTTCGMformer/CGMfoundation_gai/output/ar_24h_eval/shanghai \
  --stride 3 \
  --ar_step_size 12 \
  --workers 4 \
  --min_history_hours 72 \
  --min_history_valid_ratio 0.7 \
  > /workspace/JTTCGMformer/CGMfoundation_gai/output/ar_24h_eval/shanghai/ar_eval.log 2>&1 &


'''

# =========================================================================
# Clinical distribution metrics.
# =========================================================================
def calc_clinical_metrics(seq, mask):
    valid_seq = seq[mask > 0.5]
    if len(valid_seq) == 0:
        return {"TIR": np.nan, "TAR": np.nan, "TBR": np.nan, "SB": np.nan, "Mean": np.nan, "CV": np.nan, "SD": np.nan}
    
    n = len(valid_seq)
    tir = np.sum((valid_seq >= 3.9) & (valid_seq <= 10.0)) / n * 100.0
    tar = np.sum(valid_seq > 10.0) / n * 100.0
    tbr = np.sum(valid_seq < 3.9) / n * 100.0
    sb = np.sum(valid_seq < 3.0) / n * 100.0
    mean_val = np.mean(valid_seq)
    sd_val = np.std(valid_seq)
    cv = (sd_val / mean_val * 100.0) if mean_val > 0 else 0.0
    
    return {"TIR": tir, "TAR": tar, "TBR": tbr, "SB": sb, "Mean": mean_val, "CV": cv, "SD": sd_val}

def calc_window_metrics(pred, true, mask):
    """Compute MAE and correlation for one patient and forecast window."""
    valid_p = pred[mask > 0.5]
    valid_t = true[mask > 0.5]
    if len(valid_p) < 2: return np.nan, np.nan
    mae = np.mean(np.abs(valid_p - valid_t))
    r = pearsonr(valid_p, valid_t)[0] if np.std(valid_p) > 1e-5 and np.std(valid_t) > 1e-5 else 0.0
    return mae, r

# =========================================================================
# Per-patient worker.
# =========================================================================
@torch.inference_mode()
def process_single_patient(pid, row, mask, model, args, patch_size, target_24h_pts, max_seq_len, required_min_len):
    valid_indices = np.where(mask == 1)[0]
    if len(valid_indices) == 0:
        return None, None, "no_valid_points"
    last_idx = valid_indices[-1]
    
    if last_idx + 1 < required_min_len:
        return None, None, "insufficient_total_length"
        
    available_history = last_idx + 1 - target_24h_pts
    start_idx = max(0, available_history - max_seq_len)
    end_idx = last_idx + 1
    
    row_segment = row[start_idx:end_idx]
    mask_segment = mask[start_idx:end_idx]
    
    row_filled = pd.Series(row_segment).ffill().fillna(5.5).to_numpy(dtype=np.float32)
    
    true_future = row_filled[-target_24h_pts:]
    true_future_mask = mask_segment[-target_24h_pts:]
    if np.sum(true_future_mask) < (target_24h_pts * 0.5):
        return None, None, "insufficient_target_valid_ratio"

    history_mask = mask_segment[:-target_24h_pts]
    history_valid_ratio = float(np.mean(history_mask)) if len(history_mask) > 0 else 0.0
    if history_valid_ratio < args.min_history_valid_ratio:
        return None, None, "insufficient_history_valid_ratio"

    context_seq = row_filled[:-target_24h_pts].tolist()
    
    # Trim the leading remainder to align the last patch with the context end.
    rem = (len(context_seq) - patch_size) % args.stride
    if rem != 0:
        context_seq = context_seq[rem:]
        mask_segment = mask_segment[rem:]  # Keep the mask aligned with the trimmed context.
        start_idx += rem 

    # Move the history to the selected device.
    gpu_context = torch.tensor(context_seq, dtype=torch.float32, device=args.device)
    
    # Determine the number of AR iterations from the rollout step size.
    ar_steps = target_24h_pts // args.ar_step_size 
    # ==================================================================================================
    original_phys_start = start_idx 
    
    pred_future_list = []

    # Roll out predictions autoregressively.
    for step in range(ar_steps):
        recent_context = gpu_context[-max_seq_len:]
        
        # Use the model patch stride when unfolding the context.
        patches = recent_context.unfold(0, patch_size, args.stride).contiguous()
        # ====================================================================================================
        inp = patches.unsqueeze(0) 
        
        current_start_idx = original_phys_start + max(0, len(gpu_context) - max_seq_len)
        
        # Pass the patch stride for position and time encoding.
        pred_all = model(inp, start_idx=current_start_idx, stride=args.stride) 
        # ==============================================================================================================
        
        # Append the first ar_step_size predictions from the last patch.
        step_pred = pred_all[0, -1, :args.ar_step_size]
        # ==============================================================================================================

        # Clip predictions before feeding them into the next AR iteration.
        # Bound each predicted step before adding it to the context.
        step_pred = torch.clamp(step_pred, min=args.clip_min, max=args.clip_max)
        # ==============================================================================================================
        
        gpu_context = torch.cat([gpu_context, step_pred], dim=0)
        pred_future_list.extend(step_pred.cpu().tolist())
        
    # Clip the final 24-hour prediction curve.
    # Apply the same bounds to the saved and evaluated predictions.
    pred_future = np.array(pred_future_list, dtype=np.float32)[:target_24h_pts]
    pred_future = np.clip(pred_future, args.clip_min, args.clip_max)
    # ==============================================================================================================
    
    patient_res = {"patient_id": pid, "History_Valid_Ratio": history_valid_ratio}
    
    # Compute MAE and correlation for each patient.
    windows = {"30min": 6, "1h": 12, "2h": 24, "4h": 48, "6h": 72, "12h": 144, "24h": 288}
    for name, pts in windows.items():
        mae, r = calc_window_metrics(pred_future[:pts], true_future[:pts], true_future_mask[:pts])
        patient_res[f"Indiv_MAE_{name}"] = mae
        patient_res[f"Indiv_R_{name}"] = r
    
    true_clin = calc_clinical_metrics(true_future, true_future_mask)
    pred_clin = calc_clinical_metrics(pred_future, true_future_mask)
    
    for k in ["TIR", "TAR", "TBR", "SB", "SD", "Mean", "CV"]:
        patient_res[f"True_{k}"] = true_clin[k]
        patient_res[f"Pred_{k}"] = pred_clin[k]
        patient_res[f"Delta_{k}"] = pred_clin[k] - true_clin[k]
        
    raw_curve = {"true": true_future, "pred": pred_future, "mask": true_future_mask}
    return patient_res, {pid: raw_curve}, "kept"

# =========================================================================
# Command-line entry point.
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="24h AR Digital Twin Evaluation (Dual Correlation)")
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ar_24h_results")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Configure the patch stride and AR rollout step separately.
    parser.add_argument("--stride", type=int, default=3, help="Patch stride used during model training")
    parser.add_argument("--ar_step_size", type=int, default=3, help="Number of forecast steps appended per autoregressive iteration")
    # ========================================================================================
    
    parser.add_argument("--workers", type=int, default=1, help="Use 1 worker for GPU inference; multiple threads make patients compete for the same GPU")
    parser.add_argument("--min_history_hours", type=int, default=24, help="Minimum observed history required before forecasting (hours)")
    parser.add_argument("--min_history_valid_ratio", type=float, default=0.7, help="Minimum fraction of observed points in the historical context; patients below this threshold are skipped")

    # Prediction clipping bounds.
    # Default bounds for the 24-hour AR predictions: 2.22-33.33 mmol/L.
    parser.add_argument("--clip_min", type=float, default=2.22, help="Lower bound for clipping predicted glucose values (mmol/L)")
    parser.add_argument("--clip_max", type=float, default=33.33, help="Upper bound for clipping predicted glucose values (mmol/L)")
    # ========================================================================================

    args = parser.parse_args()

    # Validate the clipping bounds.
    if args.clip_min >= args.clip_max:
        raise ValueError(f"clip_min must be less than clip_max; current clip_min={args.clip_min}, clip_max={args.clip_max}")
    # ========================================================================================

    # Restore model settings and architecture flags.
    ckpt = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
    h = ckpt.get("model_hparams", {}) if isinstance(ckpt, dict) else {}
    def pick_int(key, default): return int(h.get(key, default)) if h.get(key) is not None else default
    def pick_flag(key, default=True): return bool(h[key]) if key in h else default

    patch_size = pick_int("patch_size", 144)
    forecast_steps = pick_int("forecast_steps", 12)
    max_seq_len = pick_int("max_seq_len", 3456)
    embed_dim = pick_int("embed_dim", 256)
    depth = pick_int("depth", 12)
    num_heads = pick_int("num_heads", 8)
    hidden_dim = pick_int("hidden_dim", 1024)
    dropout = float(h.get("dropout", 0.0)) if h.get("dropout") is not None else 0.0
    ckpt_stride = pick_int("stride", args.stride)

    use_cnn = pick_flag("use_cnn", True)
    use_transformer = pick_flag("use_transformer", True)
    use_temporal = pick_flag("use_temporal", True)
    use_positional = pick_flag("use_positional", True)
    
    print("\n" + "="*85)
    print(f"🚀 CGM Foundation Model - Full-Context 24H Fast Concurrent Autoregressive Forecasting (Dual Correlation Evaluation)")
    print(f"📥 Loading checkpoint file: {os.path.basename(args.checkpoint_path)}")
    print(f"📍 Checkpoint path: {args.checkpoint_path}")
    print("-" * 85)
    print("🛠️  CKP Model Parameter Audit (Important Parameters):")
    print(f"    - Patch Size:     {patch_size:>5} | CKP Stride:     {ckpt_stride:>5} | Eval Stride: {args.stride:>5}")
    print(f"    - Forecast Steps: {forecast_steps:>5} | Embed Dim:      {embed_dim:>5}")
    print(f"    - Max Seq Len:    {max_seq_len:>5} | Depth:          {depth:>5}")
    print(f"    - Num Heads:      {num_heads:>5} | Hidden Dim:     {hidden_dim:>5}")
    print(f"    - Dropout(CKP):   {dropout}")
    print("-" * 85)
    print("⚙️  Network Architecture Flags:")
    print(f"    - CNN Encoder:      {use_cnn}")
    print(f"    - Transformer:      {use_transformer}")
    print(f"    - Positional Embed: {use_positional}")
    print(f"    - Temporal Embed:   {use_temporal}")
    print("-" * 85)
    if args.stride != ckpt_stride:
        print(f"⚠️  Warning: Eval Stride={args.stride} differs from CKP Stride={ckpt_stride}; verify that this setting is intentional.")
    print(f"🔥 Concurrent threads: {args.workers}  |  AR rollout step size: {args.ar_step_size}")
    print(f"📊 Minimum history valid ratio: {args.min_history_valid_ratio:.2f}")
    # Log the prediction clipping bounds.
    print(f"🧪 Prediction clipping range: [{args.clip_min:.2f}, {args.clip_max:.2f}] mmol/L")
    # ========================================================================================
    print("="*85)

    model = CGMBaseModel(
        patch_size=patch_size, forecast_size=forecast_steps, embed_dim=embed_dim,
        max_seq_len=max_seq_len, depth=depth, num_heads=num_heads, 
        hidden_dim=hidden_dim, dropout=0.0, use_cnn=use_cnn, 
        use_transformer=use_transformer, use_temporal=use_temporal, 
        use_positional=use_positional
    ).to(args.device)
    
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Total model parameters: {total_params / 1e6:.2f} M ({total_params} parameters)")

    # Prepare the patient data.
    df = pd.read_csv(args.csv_path, index_col=0)
    patient_ids = df["id"].to_numpy() if "id" in df.columns else df.index.to_numpy()
    if "id" in df.columns: df = df.drop(columns=["id"])

    col_map = {int(re.findall(r'\d+', str(col))[0]): col for col in df.columns if re.findall(r'\d+', str(col))}
    df = df[[col_map[k] for k in sorted(col_map.keys())]]
    
    target_24h_pts = 288
    
    min_history_pts = args.min_history_hours * 12 
    actual_min_history = max(patch_size, min_history_pts) 
    required_min_len = actual_min_history + target_24h_pts

    results = []
    raw_curves_dict = {} 
    skip_counts = {
        "no_valid_points": 0,
        "insufficient_total_length": 0,
        "insufficient_target_valid_ratio": 0,
        "insufficient_history_valid_ratio": 0,
    }

    # Evaluate patients with the thread pool.
    futures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i in range(len(df)):
            pid = patient_ids[i]
            row = clean_cgm_array(df.iloc[i].to_numpy(dtype=np.float32))
            mask = (~np.isnan(row)).astype(np.float32)
            futures.append(executor.submit(
                process_single_patient, pid, row, mask, model, args, 
                patch_size, target_24h_pts, max_seq_len, required_min_len
            ))
            
        for future in tqdm(as_completed(futures), total=len(futures), desc="⚡ A100 dual-correlation forecasting"):
            res, curve, status = future.result()
            if res is not None:
                results.append(res)
                raw_curves_dict.update(curve)
            elif status in skip_counts:
                skip_counts[status] += 1

    total_samples = len(df)
    qualified_samples = len(results)
    skipped_samples = total_samples - qualified_samples

    print("\n" + "="*85)
    print("📋 Sample Quality Control Summary (AR 24h Evaluation)")
    print(f"   ▶ Total input samples: {total_samples}")
    print(f"   ▶ Eligible samples successfully forecast: {qualified_samples}")
    print(f"   ▶ Skipped samples: {skipped_samples}")
    print(f"      - No observed points: {skip_counts['no_valid_points']}")
    print(f"      - Insufficient total length: {skip_counts['insufficient_total_length']}")
    print(f"      - Fewer than 50% observed points in the final 24h: {skip_counts['insufficient_target_valid_ratio']}")
    print(f"      - Historical observed-point ratio below the threshold: {skip_counts['insufficient_history_valid_ratio']}")
    print("="*85)

    if not results:
        print("🚨 Error: No valid patient sequences found.")
        return
        
    res_df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ===================================================================
    # Pool valid points across patients to compute global metrics.
    # ===================================================================
    windows = {"30min": 6, "1h": 12, "2h": 24, "4h": 48, "6h": 72, "12h": 144, "24h": 288}
    global_time_metrics = []
    
    for name, pts in windows.items():
        all_true = []
        all_pred = []
        for pid, curve in raw_curves_dict.items():
            t = curve["true"][:pts]
            p = curve["pred"][:pts]
            m = curve["mask"][:pts]
            all_true.extend(t[m > 0.5])
            all_pred.extend(p[m > 0.5])
            
        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        
        global_mae = np.mean(np.abs(all_true - all_pred))
        global_r = pearsonr(all_true, all_pred)[0] if len(all_true) > 1 and np.std(all_true)>1e-5 and np.std(all_pred)>1e-5 else 0.0
        
        # Compare pooled metrics with mean patient-level metrics.
        global_time_metrics.append({
            "Horizon": name, 
            "Global_MAE": global_mae, 
            "Global_R": global_r,                     
            "Avg_Indiv_MAE": res_df[f"Indiv_MAE_{name}"].mean(), 
            "Avg_Indiv_R": res_df[f"Indiv_R_{name}"].mean()      
        })

    global_time_df = pd.DataFrame(global_time_metrics)

    # ===================================================================
    
    print("\n" + "="*85)
    print("🔬 Population Clinical Steady-State Consistency (Population Metrics Correlation)")
    metric_corr_results = []
    for k in ["TIR", "TAR", "TBR", "SB", "SD", "Mean", "CV"]:
        valid_data = res_df[[f"True_{k}", f"Pred_{k}"]].dropna()
        if len(valid_data) > 1 and valid_data[f"True_{k}"].std() > 1e-5 and valid_data[f"Pred_{k}"].std() > 1e-5:
            r_val, _ = pearsonr(valid_data[f"True_{k}"], valid_data[f"Pred_{k}"])
        else:
            r_val = np.nan
        print(f"   ▶ {k:<4} Predicted vs. observed clinical correlation (R): {r_val:.4f}")
        metric_corr_results.append({"Metric": k, "Pearson_R": r_val})
        
    # Save metric tables and prediction curves.
    # Include ar_step_size in output names.
    pd.DataFrame(metric_corr_results).to_csv(os.path.join(args.output_dir, f"Pop_Metric_Corr_step{args.ar_step_size}.csv"), index=False)
    global_time_df.to_csv(os.path.join(args.output_dir, f"Global_vs_Indiv_TimeDecay_step{args.ar_step_size}.csv"), index=False)
    res_df.to_csv(os.path.join(args.output_dir, f"AR_24h_step_{args.ar_step_size}_clin_metrics.csv"), index=False)
    torch.save(raw_curves_dict, os.path.join(args.output_dir, f"AR_24h_step_{args.ar_step_size}_curves.pt"))
    
    print("\n📉 [Multi-Scale Time-Decay Tracking (Global vs Avg Individual)]:")
    for m in global_time_metrics:
        print(f"   ⏳ {m['Horizon']:>3} horizon | Global R: {m['Global_R']:.4f} (MAE: {m['Global_MAE']:.3f}) | Mean Individual R: {m['Avg_Indiv_R']:.4f} (MAE: {m['Avg_Indiv_MAE']:.3f})")
        
    print(f"\n✅ Evaluation complete! Successfully forecast samples: {len(results)} | Results saved to: {args.output_dir}")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()
