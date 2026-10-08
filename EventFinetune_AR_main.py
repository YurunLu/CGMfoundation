from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import EventFinetune_main as event_main
from EventFinetune import export_event_prepost_prediction_excel as helpers
from EventFinetune import export_event_rollout24_compact as compact
from EventFinetune.Event_Finetune_model import EventFinetuneModel
from EventFinetune.Event_Finetune_data_utils import fill_nan_linear, normalize_entity_key

ROW_TIME = "时间轴"
ROW_INPUT_VALUE = "输入值"
ROW_EVENT = "事件"

TIME_INTERVAL_MINUTES = getattr(helpers, "TIME_INTERVAL_MINUTES", 5)
COL_CONTEXT_EVENT_COUNT = "full_context_event_count"
COL_CONTEXT_EVENT_SUMMARY = "full_context_event_summary"


def calc_clinical_metrics(seq, mask):
    valid_seq = seq[mask > 0.5]
    if len(valid_seq) == 0:
        return {
            "TIR": np.nan,
            "TAR": np.nan,
            "TBR": np.nan,
            "SB": np.nan,
            "Mean": np.nan,
            "CV": np.nan,
            "SD": np.nan,
        }

    n = len(valid_seq)
    tir = np.sum((valid_seq >= 3.9) & (valid_seq <= 10.0)) / n * 100.0
    tar = np.sum(valid_seq > 10.0) / n * 100.0
    tbr = np.sum(valid_seq < 3.9) / n * 100.0
    sb = np.sum(valid_seq < 3.0) / n * 100.0
    mean_val = np.mean(valid_seq)
    sd_val = np.std(valid_seq)
    cv = (sd_val / mean_val * 100.0) if mean_val > 0 else 0.0

    return {
        "TIR": tir,
        "TAR": tar,
        "TBR": tbr,
        "SB": sb,
        "Mean": mean_val,
        "CV": cv,
        "SD": sd_val,
    }


def calc_window_metrics(pred, true, mask):
    valid_p = pred[mask > 0.5]
    valid_t = true[mask > 0.5]
    if len(valid_p) < 2:
        return np.nan, np.nan
    mae = np.mean(np.abs(valid_p - valid_t))
    r = (
        pearsonr(valid_p, valid_t)[0]
        if np.std(valid_p) > 1e-5 and np.std(valid_t) > 1e-5
        else 0.0
    )
    return mae, r


def calc_patch_metrics(pred, true, mask):
    valid = mask > 0.5
    if not np.any(valid):
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan}
    diff = pred[valid] - true[valid]
    mse = np.mean(diff**2)
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(diff)),
    }


def calc_global_series_metrics(true_values, pred_values):
    true_values = np.asarray(true_values, dtype=np.float32)
    pred_values = np.asarray(pred_values, dtype=np.float32)
    if len(true_values) == 0:
        return {"mae": np.nan, "r": np.nan}
    mae = float(np.mean(np.abs(true_values - pred_values)))
    r = (
        float(pearsonr(true_values, pred_values)[0])
        if len(true_values) > 1
        and np.std(true_values) > 1e-5
        and np.std(pred_values) > 1e-5
        else 0.0
    )
    return {"mae": mae, "r": r}


def build_parser():
    parser = argparse.ArgumentParser(
        description="24h autoregressive full-context evaluation for EventFinetuneModel."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--summary_path", type=str, default=None)
    source_group.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Required when --config is used. Overrides summary checkpoint when provided.",
    )
    parser.add_argument(
        "--checkpoint_kind",
        type=str,
        choices=["auto", "best", "final"],
        default="auto",
        help="Used when --summary_path is provided.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="val",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--ar_step_size",
        type=int,
        default=None,
        help="Autoregressive jump size. Default uses forecast_steps from the checkpoint.",
    )
    parser.add_argument(
        "--rollout_horizon_steps",
        type=int,
        default=288,
        help="Default 288 = 24h for 5-minute CGM.",
    )
    parser.add_argument(
        "--min_history_hours",
        type=int,
        default=72,
        help="Minimum real-history hours before the forecast window.",
    )
    parser.add_argument(
        "--min_history_valid_ratio",
        type=float,
        default=0.7,
        help="Minimum observed ratio inside the history context.",
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        default=None,
        help="Sliding window stride in 5-min points. When set, evaluates multiple "
        "windows per patient by sliding across the full history instead of only the "
        "tail. E.g., --window_stride 72 gives 6h spacing between evaluation windows. "
        "Default: None (single window at the tail only).",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Default: <run_tag>_event_compare_compact_fullcontext24_<split>.csv",
    )
    parser.add_argument(
        "--metrics_output_name",
        type=str,
        default=None,
        help="Default: <run_tag>_event_compare_compact_fullcontext24_<split>_metrics.csv",
    )
    return parser


def resolve_config_from_path(config_path: Path, device_override: str | None):
    cli_args = argparse.Namespace(
        config=str(config_path),
        device=device_override,
        epochs=None,
        batch_size=None,
        out_dir=None,
        run_name=None,
        prediction_split=None,
        num_workers=None,
    )
    return event_main.resolve_config(cli_args)


def load_json_compat(path: Path):
    if hasattr(helpers, "load_json"):
        return helpers.load_json(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def load_torch_checkpoint_compat(path: Path):
    if hasattr(helpers, "load_torch_checkpoint"):
        return helpers.load_torch_checkpoint(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def maybe_round_float_compat(value, digits: int = 6):
    if hasattr(helpers, "maybe_round_float"):
        return helpers.maybe_round_float(value, digits=digits)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value):
        return None
    rounded = round(value, digits)
    if abs(rounded) < 10 ** (-digits):
        rounded = 0.0
    return rounded


def build_patch_metric_dict_compat(preds, targets, mask):
    if hasattr(helpers, "build_patch_metric_dict"):
        return helpers.build_patch_metric_dict(preds, targets, mask)
    valid = mask.astype(bool)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {"valid_points": 0, "mae": None, "rmse": None, "bias": None, "max_ae": None}
    diff = np.asarray(preds)[valid] - np.asarray(targets)[valid]
    mse = float(np.mean(diff**2))
    return {
        "valid_points": valid_count,
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(diff)),
        "max_ae": float(np.max(np.abs(diff))),
    }


def format_timestamp_compat(ts):
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).isoformat()


def add_minutes_compat(ts, offset_minutes: int):
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts) + pd.Timedelta(minutes=int(offset_minutes))


def load_time_metadata_compat(data_csv: Path):
    if hasattr(helpers, "load_time_metadata"):
        return helpers.load_time_metadata(data_csv)
    header = pd.read_csv(data_csv, nrows=0)
    available_cols = set(str(col) for col in header.columns)
    usecols = [col for col in ("device_id", "id", "starttime", "endtime") if col in available_cols]
    if not usecols:
        return {}
    df = pd.read_csv(data_csv, usecols=usecols)
    if "device_id" not in df.columns and "id" not in df.columns:
        return {}
    if "starttime" in df.columns:
        df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce")
    else:
        df["starttime"] = pd.NaT
    if "endtime" in df.columns:
        df["endtime"] = pd.to_datetime(df["endtime"], errors="coerce")
    else:
        df["endtime"] = pd.NaT
    mapping = {}
    for _, row in df.iterrows():
        keys = []
        if "device_id" in row and not pd.isna(row["device_id"]):
            keys.append(normalize_entity_key(row["device_id"]))
        if "id" in row and not pd.isna(row["id"]):
            keys.append(normalize_entity_key(row["id"]))
        payload = {"starttime": row["starttime"], "endtime": row["endtime"]}
        for key in keys:
            mapping[key] = payload
    return mapping


def resolve_time_info_compat(time_meta, sample_id: str, device_id: str, index_value: int | None = None):
    if hasattr(helpers, "resolve_time_info"):
        return helpers.resolve_time_info(time_meta, sample_id, device_id, index_value)
    meta = (
        time_meta.get(normalize_entity_key(device_id))
        or time_meta.get(normalize_entity_key(sample_id))
        or {}
    )
    starttime = meta.get("starttime")
    endtime = meta.get("endtime")
    absolute_time = None
    if index_value is not None and starttime is not None and not pd.isna(starttime):
        absolute_time = add_minutes_compat(starttime, index_value * TIME_INTERVAL_MINUTES)
    return {
        "series_start_time": format_timestamp_compat(starttime),
        "series_end_time": format_timestamp_compat(endtime),
        "absolute_time": format_timestamp_compat(absolute_time),
    }


def resolve_training_config_compat(summary, summary_path: Path):
    if hasattr(helpers, "resolve_training_config"):
        return helpers.resolve_training_config(summary, summary_path)
    resolved_config_path = summary.get("resolved_config_path")
    if resolved_config_path:
        path = Path(str(resolved_config_path)).expanduser()
        if not path.is_absolute():
            path = (summary_path.parent / path).resolve()
        if path.exists():
            return load_json_compat(path)
    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Cannot resolve config for summary: {summary_path}")
    return dict(config)


def resolve_checkpoint_path_compat(summary, summary_path: Path, checkpoint_kind: str):
    if hasattr(helpers, "resolve_checkpoint_path"):
        return helpers.resolve_checkpoint_path(summary, summary_path, checkpoint_kind)

    def normalize(raw_value):
        if not raw_value:
            return None
        path = Path(str(raw_value)).expanduser()
        if not path.is_absolute():
            path = (summary_path.parent / path).resolve()
        return path

    best_path = normalize(summary.get("best_checkpoint_path"))
    final_path = normalize(summary.get("final_checkpoint_path"))

    if checkpoint_kind == "best":
        if best_path is None:
            raise ValueError(f"Summary does not contain best_checkpoint_path: {summary_path}")
        return best_path, "best"
    if checkpoint_kind == "final":
        if final_path is None:
            raise ValueError(f"Summary does not contain final_checkpoint_path: {summary_path}")
        return final_path, "final"
    if best_path is not None:
        return best_path, "best"
    if final_path is not None:
        return final_path, "final"
    raise ValueError(f"No checkpoint path found in summary: {summary_path}")


def build_key_to_index_map_compat(dataset):
    mapping = {}
    for idx, (sample_id, device_id) in enumerate(zip(dataset.ids, dataset.device_ids)):
        key = (normalize_entity_key(sample_id), normalize_entity_key(device_id))
        if key in mapping:
            raise ValueError(f"Duplicate dataset key encountered: {key}")
        mapping[key] = idx
    return mapping


def resolve_selected_indices_compat(dataset, config, summary, split_name: str):
    if hasattr(helpers, "resolve_selected_indices"):
        try:
            return helpers.resolve_selected_indices(dataset, config, summary, split_name)
        except Exception:
            pass

    split_export = summary.get("split_export")
    key_to_index = build_key_to_index_map_compat(dataset)

    if isinstance(split_export, dict) and split_export.get("assignment_path"):
        assignment_path = Path(str(split_export["assignment_path"])).expanduser()
        if not assignment_path.is_absolute():
            assignment_path = (Path(config["out_dir"]).expanduser().resolve() / assignment_path).resolve()
        if assignment_path.exists():
            assignment_df = pd.read_csv(assignment_path)
            split_indices = {"train": [], "val": [], "all": []}
            missing_rows = []
            for _, row in assignment_df.iterrows():
                key = (
                    normalize_entity_key(row["id"]),
                    normalize_entity_key(row["device_id"]),
                )
                dataset_index = key_to_index.get(key)
                if dataset_index is None:
                    raw_index = row.get("dataset_index")
                    if pd.notna(raw_index):
                        raw_index = int(raw_index)
                        if 0 <= raw_index < len(dataset):
                            dataset_index = raw_index
                if dataset_index is None:
                    missing_rows.append(key)
                    continue
                row_split = str(row["split"])
                if row_split in ("train", "val"):
                    split_indices[row_split].append(dataset_index)
                split_indices["all"].append(dataset_index)
            if missing_rows:
                raise RuntimeError(
                    f"Failed to remap {len(missing_rows)} split rows from assignment file: {assignment_path}"
                )
            for name in split_indices:
                split_indices[name] = sorted(split_indices[name])
            return split_indices[split_name], split_indices

    split_map = event_main.split_indices(dataset, float(config["train_ratio"]))
    normalized = {name: [int(idx) for idx in values] for name, values in split_map.items()}
    return normalized[split_name], normalized


def apply_saved_event_stats_compat(dataset, summary, all_indices):
    if hasattr(helpers, "apply_saved_event_stats"):
        try:
            return helpers.apply_saved_event_stats(dataset, summary, all_indices)
        except Exception:
            pass
    event_stats = summary.get("event_stats")
    if isinstance(event_stats, dict) and event_stats:
        dataset.apply_event_stats(event_stats, device_indices=all_indices)


def build_models_compat(config, device: torch.device):
    if hasattr(helpers, "build_models"):
        return helpers.build_models(config, device)
    model = EventFinetuneModel(
        pretrained_ckpt_path=config["pretrain_ckpt"],
        num_event_types=len(config["event_cols"]),
        patch_size=int(config["patch_size"]),
        forecast_size=int(config["forecast_steps"]),
        embed_dim=int(config["embed_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        hidden_dim=int(config["hidden_dim"]),
        max_seq_len=int(config["max_seq_len"]),
        dropout=float(config["dropout"]),
        use_cnn=bool(config.get("use_cnn", True)),
        use_transformer=bool(config.get("use_transformer", True)),
        use_temporal=bool(config.get("use_temporal", True)),
        use_positional=bool(config.get("use_positional", True)),
    ).to(device)
    model.eval()
    model.base.eval()
    return model


def load_post_model_state_compat(model, checkpoint_path: Path):
    if hasattr(helpers, "load_post_model_state"):
        return helpers.load_post_model_state(model, checkpoint_path)
    checkpoint = load_torch_checkpoint_compat(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload: {checkpoint_path}")
    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint)),
    )
    result = model.load_state_dict(state_dict, strict=False)
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}. missing={missing}, unexpected={unexpected}"
        )


def resolve_run_context(args):
    summary = None
    resolved_ckpt_kind = "explicit"

    if args.summary_path:
        summary_path = Path(args.summary_path).expanduser().resolve()
        summary = load_json_compat(summary_path)
        config = resolve_training_config_compat(summary, summary_path)
        checkpoint_path, resolved_ckpt_kind = resolve_checkpoint_path_compat(
            summary=summary,
            summary_path=summary_path,
            checkpoint_kind=args.checkpoint_kind,
        )
        if args.checkpoint_path:
            checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        default_output_dir = summary_path.parent / "ar_24h_fullcontext"
    else:
        config_path = Path(args.config).expanduser().resolve()
        config = resolve_config_from_path(config_path, args.device)
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when --config is used.")
        checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
        default_output_dir = checkpoint_path.parent / "ar_24h_fullcontext"

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_dir.resolve()
    )
    return config, summary, checkpoint_path, output_dir, resolved_ckpt_kind


def resolve_run_tag(config, summary, checkpoint_path: Path):
    if isinstance(summary, dict) and summary.get("run_tag"):
        return str(summary["run_tag"])
    if isinstance(config, dict) and config.get("run_name"):
        return str(config["run_name"])
    return checkpoint_path.stem


def resolve_event_array(events_map, device_id: str, sample_id: str):
    candidates = []
    for raw in (device_id, sample_id):
        normalized = normalize_entity_key(raw)
        if normalized and normalized not in candidates:
            candidates.append(normalized)
        raw_text = str(raw).strip()
        if raw_text and raw_text not in candidates:
            candidates.append(raw_text)
    for key in candidates:
        if key in events_map:
            return np.asarray(events_map[key], dtype=np.float32), key
    return None, None


def build_event_timeline_compat(
    full_events,
    event_cols,
    sample_id,
    device_id,
    time_meta,
    event_content_cache,
    time_text_cache,
    start,
    length,
    patch_size,
    is_forecast,
):
    params = inspect.signature(compact.build_event_timeline).parameters
    kwargs = {
        "full_events": full_events,
        "event_cols": event_cols,
        "sample_id": sample_id,
        "device_id": device_id,
        "time_meta": time_meta,
        "start": start,
        "length": length,
        "patch_size": patch_size,
        "is_forecast": is_forecast,
    }
    if "event_content_cache" in params:
        kwargs["event_content_cache"] = event_content_cache
    if "time_text_cache" in params:
        kwargs["time_text_cache"] = time_text_cache
    return compact.build_event_timeline(**kwargs)


def format_event_content_compat(row_values: np.ndarray, event_cols: list[str]) -> str:
    if hasattr(compact, "format_event_content"):
        return compact.format_event_content(row_values, event_cols)
    parts = []
    for event_idx, event_name in enumerate(event_cols):
        raw_value = row_values[event_idx]
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if np.isnan(value) or value == 0.0:
            continue
        rounded = maybe_round_float_compat(value)
        parts.append(f"{event_name}={rounded}")
    return "; ".join(parts)


def build_context_event_summary(
    full_events: np.ndarray,
    event_cols: list[str],
    sample_id: str,
    device_id: str,
    time_meta: dict,
    start: int,
    end: int,
    max_items: int = 32,
):
    summary_parts = []
    count = 0
    upper = min(int(end), int(full_events.shape[0]))
    for abs_index in range(int(start), upper):
        content = format_event_content_compat(np.asarray(full_events[abs_index]), event_cols)
        if not content:
            continue
        count += 1
        if len(summary_parts) >= max_items:
            continue
        time_text = resolve_time_info_compat(
            time_meta,
            sample_id,
            device_id,
            abs_index,
        )["absolute_time"] or f"index={abs_index}"
        summary_parts.append(f"{abs_index}@{time_text}:{content}")
    summary = " | ".join(summary_parts)
    if count > max_items:
        summary = f"{summary} | ... ({count - max_items} more)"
    return count, summary


def build_full_context_axis_columns(max_seq_len: int, rollout_horizon_steps: int):
    columns = [
        "sample_id",
        "device_id",
        "window_index",
        "context_start_index",
        "forecast_start_index",
        "context_length",
        "forecast_horizon",
        "history_valid_ratio",
        COL_CONTEXT_EVENT_COUNT,
        COL_CONTEXT_EVENT_SUMMARY,
        "row_type",
    ]
    columns.extend([f"ctx_{idx + 1}" for idx in range(max_seq_len)])
    columns.extend([f"t+{idx + 1}" for idx in range(rollout_horizon_steps)])
    return columns


def fill_right_aligned_columns(row: dict, column_names: list[str], values, formatter=None):
    values = list(values)
    if len(values) > len(column_names):
        values = values[-len(column_names) :]
    start = len(column_names) - len(values)
    for idx, value in enumerate(values):
        if formatter is not None:
            value = formatter(value)
        row[column_names[start + idx]] = value


def fill_forecast_time_columns(
    row: dict,
    sample_id: str,
    device_id: str,
    time_meta: dict,
    forecast_start: int,
    rollout_horizon_steps: int,
):
    for idx in range(rollout_horizon_steps):
        abs_index = int(forecast_start + idx)
        time_value = resolve_time_info_compat(
            time_meta,
            sample_id,
            device_id,
            abs_index,
        )["absolute_time"]
        row[f"t+{idx + 1}"] = time_value or f"index={abs_index}"


def fill_input_time_columns(
    row: dict,
    input_column_names: list[str],
    sample_id: str,
    device_id: str,
    time_meta: dict,
    input_start: int,
):
    for idx, column_name in enumerate(input_column_names):
        abs_index = int(input_start + idx)
        time_value = resolve_time_info_compat(
            time_meta,
            sample_id,
            device_id,
            abs_index,
        )["absolute_time"]
        row[column_name] = time_value or f"index={abs_index}"


def apply_eval_event_stats(model_dataset, config, summary, split_map):
    if summary is not None:
        apply_saved_event_stats_compat(
            dataset=model_dataset,
            summary=summary,
            all_indices=list(split_map["all"]),
        )
        return

    if not bool(config.get("normalize_continuous_events", False)):
        return

    continuous_cols = model_dataset.infer_continuous_event_columns()
    if not continuous_cols:
        return
    event_stats = model_dataset.compute_event_stats(
        device_indices=list(split_map["train"]),
        continuous_event_columns=continuous_cols,
    )
    model_dataset.apply_event_stats(event_stats, device_indices=list(split_map["all"]))


def evaluate_selected_patch_losses(model_dataset, selected_indices, pre_base, post_model, config, device):
    if not selected_indices:
        return {
            "selected_devices": 0,
            "pre_patch_mae": np.nan,
            "post_patch_mae": np.nan,
        }

    subset = model_dataset.subset_by_device_indices(selected_indices)
    loader = event_main.build_loader(
        subset=subset,
        batch_size=int(config["batch_size"]),
        num_workers=0,
        shuffle=False,
    )
    model_stride = int(config.get("model_stride", config["stride"]))
    pre_patch_mae = event_main.evaluate_base(pre_base, loader, device, model_stride)
    post_patch_mae = event_main.evaluate(post_model, loader, device, model_stride)
    return {
        "selected_devices": int(len(selected_indices)),
        "pre_patch_mae": float(pre_patch_mae),
        "post_patch_mae": float(post_patch_mae),
    }


def prepare_rollout_state(
    raw_glucose,
    model_events,
    patch_size,
    forecast_steps,
    model_stride,
    max_seq_len,
    target_horizon_steps,
    min_history_valid_ratio,
    required_min_len,
    target_end_idx=None,
):
    raw_glucose = np.asarray(raw_glucose, dtype=np.float32)
    raw_mask = (~np.isnan(raw_glucose)).astype(np.float32)
    valid_indices = np.flatnonzero(raw_mask > 0.5)
    if len(valid_indices) == 0:
        return None, "no_valid_points"

    if target_end_idx is None:
        target_end_idx = int(valid_indices[-1]) + 1
    if target_end_idx < required_min_len:
        return None, "insufficient_total_length"

    target_start = target_end_idx - target_horizon_steps
    context_abs_start = max(0, target_start - max_seq_len)

    row_segment = raw_glucose[context_abs_start : target_end_idx]
    mask_segment = raw_mask[context_abs_start : target_end_idx]
    row_filled = pd.Series(row_segment).ffill().fillna(5.5).to_numpy(dtype=np.float32)

    true_future = row_filled[-target_horizon_steps:]
    true_future_mask = mask_segment[-target_horizon_steps:]
    if np.sum(true_future_mask) < (target_horizon_steps * 0.5):
        return None, "insufficient_target_valid_ratio"

    history_mask = mask_segment[:-target_horizon_steps]
    history_valid_ratio = float(np.mean(history_mask)) if len(history_mask) > 0 else 0.0
    if history_valid_ratio < min_history_valid_ratio:
        return None, "insufficient_history_valid_ratio"

    context_seq = row_filled[:-target_horizon_steps]
    if len(context_seq) < patch_size:
        return None, "insufficient_total_length"

    rem = (len(context_seq) - patch_size) % model_stride
    if rem != 0:
        context_seq = context_seq[rem:]
        context_abs_start += rem

    if len(context_seq) < patch_size:
        return None, "insufficient_total_length"

    model_events = np.asarray(model_events, dtype=np.float32)
    model_events = fill_nan_linear(model_events.copy()).astype(np.float32)

    return {
        "context_seq": context_seq.astype(np.float32),
        "context_abs_start": int(context_abs_start),
        "true_future": true_future.astype(np.float32),
        "true_future_mask": true_future_mask.astype(np.float32),
        "history_valid_ratio": history_valid_ratio,
        "model_events": model_events,
        "input_patch": context_seq[-patch_size:].astype(np.float32),
        "input_start": int(context_abs_start + len(context_seq) - patch_size),
        "window_index": int((len(context_seq) - patch_size) // model_stride),
        "forecast_start": int(target_start),
        "forecast_end": int(target_start + target_horizon_steps),
    }, "kept"


@torch.inference_mode()
def rollout_base_future(
    base_model,
    context_seq,
    context_abs_start,
    patch_size,
    forecast_steps,
    model_stride,
    max_seq_len,
    target_horizon_steps,
    ar_step_size,
    device,
):
    gpu_context = torch.as_tensor(context_seq, dtype=torch.float32, device=device)
    pred_future_list = []

    while len(pred_future_list) < target_horizon_steps:
        recent_context = gpu_context[-max_seq_len:]
        current_abs_start = context_abs_start + max(0, len(gpu_context) - max_seq_len)

        patches = recent_context.unfold(0, patch_size, model_stride).contiguous()
        num_tokens = int(patches.shape[0])
        patch_starts = (
            torch.arange(num_tokens, dtype=torch.long, device=device) * model_stride
            + int(current_abs_start)
        )

        pred_all = base_model(
            patches.unsqueeze(0),
            start_idx=patch_starts.view(1, -1),
            stride=model_stride,
        )
        take = min(ar_step_size, target_horizon_steps - len(pred_future_list))
        step_pred = pred_all[0, -1, :take]
        gpu_context = torch.cat([gpu_context, step_pred], dim=0)
        pred_future_list.extend(step_pred.cpu().tolist())

    return np.asarray(pred_future_list[:target_horizon_steps], dtype=np.float32)


@torch.inference_mode()
def rollout_event_future(
    model,
    context_seq,
    context_abs_start,
    model_events,
    patch_size,
    forecast_steps,
    model_stride,
    max_seq_len,
    target_horizon_steps,
    ar_step_size,
    device,
):
    event_dim = int(model_events.shape[1])
    event_pad = np.zeros((patch_size + forecast_steps, event_dim), dtype=np.float32)
    full_events_gpu = torch.as_tensor(
        np.concatenate([model_events, event_pad], axis=0),
        dtype=torch.float32,
        device=device,
    )
    event_offsets = torch.arange(
        patch_size + forecast_steps,
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    gpu_context = torch.as_tensor(context_seq, dtype=torch.float32, device=device)
    pred_future_list = []

    while len(pred_future_list) < target_horizon_steps:
        recent_context = gpu_context[-max_seq_len:]
        current_abs_start = context_abs_start + max(0, len(gpu_context) - max_seq_len)

        patches = recent_context.unfold(0, patch_size, model_stride).contiguous()
        num_tokens = int(patches.shape[0])
        patch_starts = (
            torch.arange(num_tokens, dtype=torch.long, device=device) * model_stride
            + int(current_abs_start)
        )
        event_tensor = full_events_gpu[patch_starts.view(-1, 1) + event_offsets]

        pred_all = model(
            patches.unsqueeze(0),
            event_tensor.unsqueeze(0),
            start_idx=patch_starts.view(1, -1),
            stride=model_stride,
        )
        take = min(ar_step_size, target_horizon_steps - len(pred_future_list))
        step_pred = pred_all[0, -1, :take]
        gpu_context = torch.cat([gpu_context, step_pred], dim=0)
        pred_future_list.extend(step_pred.cpu().tolist())

    return np.asarray(pred_future_list[:target_horizon_steps], dtype=np.float32)


@torch.inference_mode()
def direct_base_future(
    base_model,
    context_seq,
    context_abs_start,
    patch_size,
    forecast_steps,
    model_stride,
    max_seq_len,
    direct_horizon_steps,
    device,
):
    gpu_context = torch.as_tensor(context_seq, dtype=torch.float32, device=device)
    recent_context = gpu_context[-max_seq_len:]
    current_abs_start = context_abs_start + max(0, len(gpu_context) - max_seq_len)

    patches = recent_context.unfold(0, patch_size, model_stride).contiguous()
    num_tokens = int(patches.shape[0])
    patch_starts = (
        torch.arange(num_tokens, dtype=torch.long, device=device) * model_stride
        + int(current_abs_start)
    )
    pred_all = base_model(
        patches.unsqueeze(0),
        start_idx=patch_starts.view(1, -1),
        stride=model_stride,
    )
    take = min(direct_horizon_steps, forecast_steps)
    return pred_all[0, -1, :take].cpu().numpy().astype(np.float32)


@torch.inference_mode()
def direct_event_future(
    model,
    context_seq,
    context_abs_start,
    model_events,
    patch_size,
    forecast_steps,
    model_stride,
    max_seq_len,
    direct_horizon_steps,
    device,
):
    event_dim = int(model_events.shape[1])
    event_pad = np.zeros((patch_size + forecast_steps, event_dim), dtype=np.float32)
    full_events_gpu = torch.as_tensor(
        np.concatenate([model_events, event_pad], axis=0),
        dtype=torch.float32,
        device=device,
    )
    event_offsets = torch.arange(
        patch_size + forecast_steps,
        dtype=torch.long,
        device=device,
    ).view(1, -1)

    gpu_context = torch.as_tensor(context_seq, dtype=torch.float32, device=device)
    recent_context = gpu_context[-max_seq_len:]
    current_abs_start = context_abs_start + max(0, len(gpu_context) - max_seq_len)

    patches = recent_context.unfold(0, patch_size, model_stride).contiguous()
    num_tokens = int(patches.shape[0])
    patch_starts = (
        torch.arange(num_tokens, dtype=torch.long, device=device) * model_stride
        + int(current_abs_start)
    )
    event_tensor = full_events_gpu[patch_starts.view(-1, 1) + event_offsets]
    pred_all = model(
        patches.unsqueeze(0),
        event_tensor.unsqueeze(0),
        start_idx=patch_starts.view(1, -1),
        stride=model_stride,
    )
    take = min(direct_horizon_steps, forecast_steps)
    return pred_all[0, -1, :take].cpu().numpy().astype(np.float32)


def build_patient_result(
    sample_id,
    device_id,
    history_valid_ratio,
    true_future,
    true_future_mask,
    pre_future,
    post_future,
    target_horizon_steps,
    pre_direct_future=None,
    post_direct_future=None,
    direct_horizon_steps=None,
):
    patient_res = {
        "sample_id": str(sample_id),
        "device_id": str(device_id),
        "History_Valid_Ratio": history_valid_ratio,
    }
    windows = {
        "30min": 6,
        "1h": 12,
        "2h": 24,
        "4h": 48,
        "6h": 72,
        "12h": 144,
        "24h": 288,
    }
    for name, pts in windows.items():
        if pts > target_horizon_steps:
            continue
        pre_mae, pre_r = calc_window_metrics(
            pre_future[:pts],
            true_future[:pts],
            true_future_mask[:pts],
        )
        post_mae, post_r = calc_window_metrics(
            post_future[:pts],
            true_future[:pts],
            true_future_mask[:pts],
        )
        patient_res[f"Pre_Indiv_MAE_{name}"] = pre_mae
        patient_res[f"Pre_Indiv_R_{name}"] = pre_r
        patient_res[f"Post_Indiv_MAE_{name}"] = post_mae
        patient_res[f"Post_Indiv_R_{name}"] = post_r
        patient_res[f"Indiv_MAE_{name}"] = post_mae
        patient_res[f"Indiv_R_{name}"] = post_r

    if (
        pre_direct_future is not None
        and post_direct_future is not None
        and direct_horizon_steps is not None
        and direct_horizon_steps > 0
    ):
        direct_true = true_future[:direct_horizon_steps]
        direct_mask = true_future_mask[:direct_horizon_steps]
        pre_direct_mae, pre_direct_r = calc_window_metrics(
            pre_direct_future[:direct_horizon_steps],
            direct_true,
            direct_mask,
        )
        post_direct_mae, post_direct_r = calc_window_metrics(
            post_direct_future[:direct_horizon_steps],
            direct_true,
            direct_mask,
        )
        patient_res["Direct_Horizon_Steps"] = int(direct_horizon_steps)
        patient_res["Pre_Direct_MAE"] = pre_direct_mae
        patient_res["Pre_Direct_R"] = pre_direct_r
        patient_res["Post_Direct_MAE"] = post_direct_mae
        patient_res["Post_Direct_R"] = post_direct_r

    true_clin = calc_clinical_metrics(true_future, true_future_mask)
    pre_clin = calc_clinical_metrics(pre_future, true_future_mask)
    post_clin = calc_clinical_metrics(post_future, true_future_mask)
    for key in ["TIR", "TAR", "TBR", "SB", "SD", "Mean", "CV"]:
        patient_res[f"True_{key}"] = true_clin[key]
        patient_res[f"Pre_{key}"] = pre_clin[key]
        patient_res[f"Post_{key}"] = post_clin[key]
        patient_res[f"Pre_Delta_{key}"] = pre_clin[key] - true_clin[key]
        patient_res[f"Post_Delta_{key}"] = post_clin[key] - true_clin[key]
        patient_res[f"Pred_{key}"] = post_clin[key]
        patient_res[f"Delta_{key}"] = post_clin[key] - true_clin[key]
    return patient_res


def calc_ar_iteration_group_metrics(
    pre_future,
    post_future,
    true_future,
    true_future_mask,
    ar_step_size,
):
    """Split a full rollout into AR iteration groups of ar_step_size points each.

    Each group represents a "1h forecast" from a progressively shifted origin.
    Returns per-group MAE and R, plus aggregate statistics.
    """
    total = len(true_future)
    group_size = ar_step_size
    num_groups = total // group_size
    if num_groups == 0:
        return [], {}

    pre_groups_mae = []
    pre_groups_r = []
    post_groups_mae = []
    post_groups_r = []

    for g in range(num_groups):
        start = g * group_size
        end = start + group_size
        group_mask_chunk = true_future_mask[start:end]
        group_valid_ratio = np.mean(group_mask_chunk > 0.5)
        if group_valid_ratio < 0.5:
            pre_groups_mae.append(np.nan)
            pre_groups_r.append(np.nan)
            post_groups_mae.append(np.nan)
            post_groups_r.append(np.nan)
            continue
        pre_mae, pre_r = calc_window_metrics(
            pre_future[start:end],
            true_future[start:end],
            true_future_mask[start:end],
        )
        post_mae, post_r = calc_window_metrics(
            post_future[start:end],
            true_future[start:end],
            true_future_mask[start:end],
        )
        pre_groups_mae.append(pre_mae)
        pre_groups_r.append(pre_r)
        post_groups_mae.append(post_mae)
        post_groups_r.append(post_r)

    pre_mae_arr = np.array(pre_groups_mae)
    post_mae_arr = np.array(post_groups_mae)
    pre_r_arr = np.array(pre_groups_r)
    post_r_arr = np.array(post_groups_r)

    pre_valid_mae = pre_mae_arr[~np.isnan(pre_mae_arr)]
    post_valid_mae = post_mae_arr[~np.isnan(post_mae_arr)]
    pre_valid_r = pre_r_arr[~np.isnan(pre_r_arr)]
    post_valid_r = post_r_arr[~np.isnan(post_r_arr)]
    num_rejected = num_groups - len(pre_valid_mae)

    return (
        list(zip(pre_groups_mae, pre_groups_r, post_groups_mae, post_groups_r)),
        {
            "num_groups": num_groups,
            "group_size": group_size,
            "num_rejected": num_rejected,
            "pre_mean_mae": float(np.mean(pre_valid_mae)) if len(pre_valid_mae) else np.nan,
            "pre_std_mae": float(np.std(pre_valid_mae)) if len(pre_valid_mae) else np.nan,
            "post_mean_mae": float(np.mean(post_valid_mae)) if len(post_valid_mae) else np.nan,
            "post_std_mae": float(np.std(post_valid_mae)) if len(post_valid_mae) else np.nan,
            "pre_mean_r": float(np.mean(pre_valid_r)) if len(pre_valid_r) else np.nan,
            "post_mean_r": float(np.mean(post_valid_r)) if len(post_valid_r) else np.nan,
            "pre_first_group_mae": pre_groups_mae[0] if num_groups > 0 else np.nan,
            "post_first_group_mae": post_groups_mae[0] if num_groups > 0 else np.nan,
            "pre_last_group_mae": pre_groups_mae[-1] if num_groups > 0 else np.nan,
            "post_last_group_mae": post_groups_mae[-1] if num_groups > 0 else np.nan,
        },
    )


def build_detail_columns(patch_size, rollout_horizon_steps):
    columns = [
        "sample_id",
        "device_id",
        "history_valid_ratio",
        "input_start_idx",
        "forecast_start_idx",
        "forecast_end_idx",
        "row_type",
        "mae",
        "rmse",
        "bias",
    ]
    columns.extend([f"input_{idx + 1}" for idx in range(patch_size)])
    columns.extend([f"t+{idx + 1}" for idx in range(rollout_horizon_steps)])
    return columns


def build_detail_row(
    sample_id,
    device_id,
    history_valid_ratio,
    input_start,
    forecast_start,
    forecast_end,
    row_type,
    input_patch,
    values,
    metrics,
    rollout_horizon_steps,
):
    row = {
        "sample_id": str(sample_id),
        "device_id": str(device_id),
        "history_valid_ratio": history_valid_ratio,
        "input_start_idx": int(input_start),
        "forecast_start_idx": int(forecast_start),
        "forecast_end_idx": int(forecast_end),
        "row_type": row_type,
        "mae": None if metrics is None else metrics["mae"],
        "rmse": None if metrics is None else metrics["rmse"],
        "bias": None if metrics is None else metrics["bias"],
    }
    for idx, value in enumerate(input_patch):
        row[f"input_{idx + 1}"] = float(value)
    for idx in range(rollout_horizon_steps):
        row[f"t+{idx + 1}"] = float(values[idx])
    return row


def main():
    args = build_parser().parse_args()
    config, summary, checkpoint_path, output_dir, resolved_ckpt_kind = resolve_run_context(
        args
    )
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(args.device)
    run_tag = resolve_run_tag(config, summary, checkpoint_path)
    time_meta = load_time_metadata_compat(Path(str(config["data_csv"])))

    pre_model = build_models_compat(config, device)
    pre_base = pre_model.base
    pre_base.eval()
    del pre_model

    post_model = build_models_compat(config, device)
    load_post_model_state_compat(post_model, checkpoint_path)
    post_model.eval()

    raw_dataset = event_main.build_dataset(config)
    model_dataset = event_main.build_dataset(config)

    if summary is not None:
        selected_indices, split_map = resolve_selected_indices_compat(
            dataset=model_dataset,
            config=config,
            summary=summary,
            split_name=args.split,
        )
    else:
        split_map = event_main.split_indices(model_dataset, float(config["train_ratio"]))
        selected_indices = list(split_map[args.split])

    apply_eval_event_stats(model_dataset, config, summary, split_map)

    patch_size = int(config["patch_size"])
    forecast_steps = int(config["forecast_steps"])
    model_stride = int(config.get("model_stride", config["stride"]))
    max_seq_len = int(config["max_seq_len"])
    target_horizon_steps = int(args.rollout_horizon_steps)
    direct_horizon_steps = min(forecast_steps, target_horizon_steps)
    ar_step_size = int(args.ar_step_size or forecast_steps)
    if ar_step_size <= 0:
        raise ValueError("--ar_step_size must be positive.")
    if ar_step_size > forecast_steps:
        raise ValueError(
            f"ar_step_size={ar_step_size} cannot exceed forecast_steps={forecast_steps}."
        )

    min_history_pts = int(args.min_history_hours) * 12
    actual_min_history = max(patch_size, min_history_pts)
    required_min_len = actual_min_history + target_horizon_steps
    main_csv_name = (
        args.output_name
        if args.output_name
        else f"{run_tag}_event_compare_compact_fullcontext24_{args.split}.csv"
    )
    metrics_csv_name = (
        args.metrics_output_name
        if args.metrics_output_name
        else f"{run_tag}_event_compare_compact_fullcontext24_{args.split}_metrics.csv"
    )
    axis_csv_name = f"{run_tag}_event_compare_fullcontext_axis_{args.split}.csv"
    main_csv_path = output_dir / main_csv_name
    metrics_csv_path = output_dir / metrics_csv_name
    axis_csv_path = output_dir / axis_csv_name
    main_columns = compact.build_main_columns(config, target_horizon_steps)
    for extra_col in (COL_CONTEXT_EVENT_COUNT, COL_CONTEXT_EVENT_SUMMARY):
        if extra_col not in main_columns:
            insert_at = main_columns.index(compact.COL_ROW_TYPE)
            main_columns.insert(insert_at, extra_col)
    input_value_columns = main_columns[4 : 4 + patch_size]
    axis_columns = build_full_context_axis_columns(max_seq_len, target_horizon_steps)
    axis_context_columns = [f"ctx_{idx + 1}" for idx in range(max_seq_len)]
    axis_forecast_columns = [f"t+{idx + 1}" for idx in range(target_horizon_steps)]
    metric_state = compact.init_metric_state(target_horizon_steps)
    exported_windows = 0
    valid_forecast_points_total = 0
    num_ar_groups = target_horizon_steps // ar_step_size
    ar_group_pool = {
        "pre_pred": [[] for _ in range(num_ar_groups)],
        "post_pred": [[] for _ in range(num_ar_groups)],
        "true": [[] for _ in range(num_ar_groups)],
    }
    ar_group_rejected = 0
    patch_eval_summary = evaluate_selected_patch_losses(
        model_dataset=model_dataset,
        selected_indices=selected_indices,
        pre_base=pre_base,
        post_model=post_model,
        config=config,
        device=device,
    )

    print("\n" + "=" * 85)
    print("EventFinetuneModel 24h Full-Context AR Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint Kind: {resolved_ckpt_kind}")
    print(f"Split: {args.split} | Selected Devices: {len(selected_indices)}")
    print(
        f"Patch Size: {patch_size} | Forecast Steps: {forecast_steps} | "
        f"Model Stride: {model_stride} | AR Step Size: {ar_step_size}"
    )
    print(
        f"Use CNN: {config.get('use_cnn', True)} | "
        f"Use Transformer: {config.get('use_transformer', True)} | "
        f"Use Positional: {config.get('use_positional', True)} | "
        f"Use Temporal: {config.get('use_temporal', True)}"
    )
    print(f"Min History Hours: {args.min_history_hours}")
    print(f"Min History Valid Ratio: {args.min_history_valid_ratio:.2f}")
    window_stride_val = getattr(args, 'window_stride', None)
    print(
        f"Window Stride: {window_stride_val} "
        f"({'single tail window' if window_stride_val is None else f'{window_stride_val} pts = {window_stride_val * 5 / 60:.1f}h stride'})"
    )
    print("=" * 85)

    results = []
    raw_curves_dict = {}
    skip_counts = {
        "no_valid_points": 0,
        "insufficient_total_length": 0,
        "insufficient_target_valid_ratio": 0,
        "insufficient_history_valid_ratio": 0,
    }
    event_lookup_stats = {
        "raw_hit": 0,
        "raw_miss": 0,
        "model_hit": 0,
        "model_miss": 0,
        "window_with_input_events": 0,
        "window_with_forecast_events": 0,
        "window_with_full_context_events": 0,
    }

    event_cols = list(config["event_cols"])

    # Pre-load all patient data and generate window tasks
    patient_cache = {}
    all_window_tasks = []
    window_stride_val = getattr(args, 'window_stride', None)
    for dataset_index in selected_indices:
        sample_id = normalize_entity_key(raw_dataset.ids[dataset_index])
        device_id = normalize_entity_key(raw_dataset.device_ids[dataset_index])
        raw_glucose = np.asarray(raw_dataset.glucose[dataset_index], dtype=np.float32)

        raw_events, _ = resolve_event_array(
            raw_dataset.events_map,
            device_id=device_id,
            sample_id=sample_id,
        )
        if raw_events is None:
            event_lookup_stats["raw_miss"] += 1
            raw_events = np.zeros(
                (raw_dataset.L, len(event_cols)),
                dtype=np.float32,
            )
        else:
            event_lookup_stats["raw_hit"] += 1

        model_events, _ = resolve_event_array(
            model_dataset.events_map,
            device_id=device_id,
            sample_id=sample_id,
        )
        if model_events is None:
            event_lookup_stats["model_miss"] += 1
            model_events = np.zeros(
                (model_dataset.L, len(event_cols)),
                dtype=np.float32,
            )
        else:
            event_lookup_stats["model_hit"] += 1

        patient_cache[dataset_index] = {
            "sample_id": sample_id,
            "device_id": device_id,
            "raw_glucose": raw_glucose,
            "raw_events": raw_events,
            "model_events": model_events,
        }

        raw_mask = (~np.isnan(raw_glucose)).astype(np.float32)
        valid_pos = np.flatnonzero(raw_mask > 0.5)
        if len(valid_pos) == 0:
            skip_counts["no_valid_points"] += 1
            continue
        last_valid_idx = int(valid_pos[-1])

        if window_stride_val is not None and window_stride_val > 0:
            positions = []
            pos = required_min_len
            while pos <= last_valid_idx + 1:
                positions.append(pos)
                pos += window_stride_val
            if not positions or positions[-1] < last_valid_idx + 1:
                positions.append(last_valid_idx + 1)
        else:
            positions = [last_valid_idx + 1]

        for target_end_idx in positions:
            all_window_tasks.append((dataset_index, target_end_idx))

    with (
        main_csv_path.open("w", encoding="utf-8-sig", newline="") as handle,
        axis_csv_path.open("w", encoding="utf-8-sig", newline="") as axis_handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=main_columns)
        axis_writer = csv.DictWriter(axis_handle, fieldnames=axis_columns)
        writer.writeheader()
        axis_writer.writeheader()

        for dataset_index, target_end_idx in tqdm(all_window_tasks, desc="AR rollout windows"):
            patient = patient_cache[dataset_index]
            sample_id = patient["sample_id"]
            device_id = patient["device_id"]
            raw_events = patient["raw_events"]
            model_events = patient["model_events"]

            state, status = prepare_rollout_state(
                raw_glucose=patient["raw_glucose"],
                model_events=patient["model_events"],
                patch_size=patch_size,
                forecast_steps=forecast_steps,
                model_stride=model_stride,
                max_seq_len=max_seq_len,
                target_horizon_steps=target_horizon_steps,
                min_history_valid_ratio=float(args.min_history_valid_ratio),
                required_min_len=required_min_len,
                target_end_idx=target_end_idx,
            )
            if state is None:
                if status in skip_counts:
                    skip_counts[status] += 1
                continue

            pre_direct = direct_base_future(
                base_model=pre_base,
                context_seq=state["context_seq"],
                context_abs_start=state["context_abs_start"],
                patch_size=patch_size,
                forecast_steps=forecast_steps,
                model_stride=model_stride,
                max_seq_len=max_seq_len,
                direct_horizon_steps=direct_horizon_steps,
                device=device,
            )
            post_direct = direct_event_future(
                model=post_model,
                context_seq=state["context_seq"],
                context_abs_start=state["context_abs_start"],
                model_events=state["model_events"],
                patch_size=patch_size,
                forecast_steps=forecast_steps,
                model_stride=model_stride,
                max_seq_len=max_seq_len,
                direct_horizon_steps=direct_horizon_steps,
                device=device,
            )
            pre_future = rollout_base_future(
                base_model=pre_base,
                context_seq=state["context_seq"],
                context_abs_start=state["context_abs_start"],
                patch_size=patch_size,
                forecast_steps=forecast_steps,
                model_stride=model_stride,
                max_seq_len=max_seq_len,
                target_horizon_steps=target_horizon_steps,
                ar_step_size=ar_step_size,
                device=device,
            )
            post_future = rollout_event_future(
                model=post_model,
                context_seq=state["context_seq"],
                context_abs_start=state["context_abs_start"],
                model_events=state["model_events"],
                patch_size=patch_size,
                forecast_steps=forecast_steps,
                model_stride=model_stride,
                max_seq_len=max_seq_len,
                target_horizon_steps=target_horizon_steps,
                ar_step_size=ar_step_size,
                device=device,
            )

            patient_res = build_patient_result(
                sample_id=sample_id,
                device_id=device_id,
                history_valid_ratio=state["history_valid_ratio"],
                true_future=state["true_future"],
                true_future_mask=state["true_future_mask"],
                pre_future=pre_future,
                post_future=post_future,
                target_horizon_steps=target_horizon_steps,
                pre_direct_future=pre_direct,
                post_direct_future=post_direct,
                direct_horizon_steps=direct_horizon_steps,
            )
            results.append(patient_res)

            target_mask_bool = state["true_future_mask"].astype(bool)
            compact.accumulate_metrics(
                metric_state,
                pre_preds=pre_future,
                post_preds=post_future,
                targets=state["true_future"],
                mask=state["true_future_mask"],
            )
            valid_forecast_points_total += int(target_mask_bool.sum())

            for g in range(num_ar_groups):
                g_start = g * ar_step_size
                g_end = g_start + ar_step_size
                g_mask = state["true_future_mask"][g_start:g_end]
                g_valid = g_mask > 0.5
                if np.sum(g_valid) < ar_step_size * 0.5:
                    ar_group_rejected += 1
                    continue
                ar_group_pool["pre_pred"][g].extend(
                    pre_future[g_start:g_end][g_valid].tolist()
                )
                ar_group_pool["post_pred"][g].extend(
                    post_future[g_start:g_end][g_valid].tolist()
                )
                ar_group_pool["true"][g].extend(
                    state["true_future"][g_start:g_end][g_valid].tolist()
                )

            event_content_cache: dict[int, str] = {}
            time_text_cache: dict[int, str] = {}

            input_start_time = (
                resolve_time_info_compat(
                    time_meta,
                    sample_id,
                    device_id,
                    int(state["input_start"]),
                )["absolute_time"]
                or f"index={int(state['input_start'])}"
            )
            forecast_start_time = (
                resolve_time_info_compat(
                    time_meta,
                    sample_id,
                    device_id,
                    int(state["forecast_start"]),
                )["absolute_time"]
                or f"index={int(state['forecast_start'])}"
            )
            time_text_cache[int(state["input_start"])] = input_start_time
            time_text_cache[int(state["forecast_start"])] = forecast_start_time

            input_event_cells, input_event_count, input_summary_parts = build_event_timeline_compat(
                full_events=raw_events,
                event_cols=event_cols,
                sample_id=sample_id,
                device_id=device_id,
                time_meta=time_meta,
                event_content_cache=event_content_cache,
                time_text_cache=time_text_cache,
                start=int(state["input_start"]),
                length=patch_size,
                patch_size=patch_size,
                is_forecast=False,
            )
            forecast_event_cells, forecast_event_count, forecast_summary_parts = (
                build_event_timeline_compat(
                    full_events=raw_events,
                    event_cols=event_cols,
                    sample_id=sample_id,
                    device_id=device_id,
                    time_meta=time_meta,
                    event_content_cache=event_content_cache,
                    time_text_cache=time_text_cache,
                    start=int(state["forecast_start"]),
                    length=target_horizon_steps,
                    patch_size=patch_size,
                    is_forecast=True,
                )
            )
            context_event_count, context_event_summary = build_context_event_summary(
                full_events=raw_events,
                event_cols=event_cols,
                sample_id=sample_id,
                device_id=device_id,
                time_meta=time_meta,
                start=int(state["context_abs_start"]),
                end=int(state["forecast_end"]),
            )
            if input_event_count > 0:
                event_lookup_stats["window_with_input_events"] += 1
            if forecast_event_count > 0:
                event_lookup_stats["window_with_forecast_events"] += 1
            if context_event_count > 0:
                event_lookup_stats["window_with_full_context_events"] += 1
            all_event_summary = " | ".join(input_summary_parts + forecast_summary_parts)

            context_event_cells = []
            context_time_cells = []
            context_start = int(state["context_abs_start"])
            context_end = int(state["forecast_start"])
            for abs_index in range(context_start, context_end):
                context_event_cells.append(
                    format_event_content_compat(
                        np.asarray(raw_events[abs_index]),
                        event_cols,
                    )
                    or None
                )
                context_time_cells.append(
                    resolve_time_info_compat(
                        time_meta,
                        sample_id,
                        device_id,
                        abs_index,
                    )["absolute_time"]
                    or f"index={abs_index}"
                )

            pre_metrics = build_patch_metric_dict_compat(
                pre_future,
                state["true_future"],
                target_mask_bool,
            )
            post_metrics = build_patch_metric_dict_compat(
                post_future,
                state["true_future"],
                target_mask_bool,
            )

            shared = {
                compact.COL_SAMPLE_ID: sample_id,
                compact.COL_DEVICE_ID: device_id,
                compact.COL_WINDOW_INDEX: int(state["window_index"]),
                compact.COL_WINDOW_START: int(state["input_start"]),
                compact.COL_INPUT_START_TIME: input_start_time,
                compact.COL_FORECAST_START_TIME: forecast_start_time,
                compact.COL_INPUT_EVENT_COUNT: int(input_event_count),
                compact.COL_FORECAST_EVENT_COUNT: int(forecast_event_count),
                COL_CONTEXT_EVENT_COUNT: int(context_event_count),
                COL_CONTEXT_EVENT_SUMMARY: context_event_summary,
                compact.COL_ALL_EVENT_SUMMARY: all_event_summary,
            }

            axis_shared = {
                "sample_id": sample_id,
                "device_id": device_id,
                "window_index": int(state["window_index"]),
                "context_start_index": int(state["context_abs_start"]),
                "forecast_start_index": int(state["forecast_start"]),
                "context_length": int(len(state["context_seq"])),
                "forecast_horizon": int(target_horizon_steps),
                "history_valid_ratio": state["history_valid_ratio"],
                COL_CONTEXT_EVENT_COUNT: int(context_event_count),
                COL_CONTEXT_EVENT_SUMMARY: context_event_summary,
            }

            axis_time_row = dict(axis_shared)
            axis_time_row["row_type"] = "时间轴"
            fill_right_aligned_columns(
                axis_time_row,
                axis_context_columns,
                context_time_cells,
            )
            fill_right_aligned_columns(
                axis_time_row,
                axis_forecast_columns,
                [
                    resolve_time_info_compat(
                        time_meta,
                        sample_id,
                        device_id,
                        int(state["forecast_start"]) + idx,
                    )["absolute_time"]
                    or f"index={int(state['forecast_start']) + idx}"
                    for idx in range(target_horizon_steps)
                ],
            )
            axis_writer.writerow(axis_time_row)

            axis_input_row = dict(axis_shared)
            axis_input_row["row_type"] = "输入值"
            fill_right_aligned_columns(
                axis_input_row,
                axis_context_columns,
                state["context_seq"],
                formatter=lambda x: maybe_round_float_compat(float(x)),
            )
            axis_writer.writerow(axis_input_row)

            axis_event_row = dict(axis_shared)
            axis_event_row["row_type"] = "事件"
            fill_right_aligned_columns(
                axis_event_row,
                axis_context_columns,
                context_event_cells,
            )
            fill_right_aligned_columns(
                axis_event_row,
                axis_forecast_columns,
                forecast_event_cells,
            )
            axis_writer.writerow(axis_event_row)

            axis_true_row = dict(axis_shared)
            axis_true_row["row_type"] = "真实值"
            fill_right_aligned_columns(
                axis_true_row,
                axis_forecast_columns,
                state["true_future"],
                formatter=lambda x: maybe_round_float_compat(float(x)),
            )
            axis_writer.writerow(axis_true_row)

            axis_pre_row = dict(axis_shared)
            axis_pre_row["row_type"] = "微调前"
            fill_right_aligned_columns(
                axis_pre_row,
                axis_forecast_columns,
                pre_future,
                formatter=lambda x: maybe_round_float_compat(float(x)),
            )
            axis_writer.writerow(axis_pre_row)

            axis_post_row = dict(axis_shared)
            axis_post_row["row_type"] = "微调后"
            fill_right_aligned_columns(
                axis_post_row,
                axis_forecast_columns,
                post_future,
                formatter=lambda x: maybe_round_float_compat(float(x)),
            )
            axis_writer.writerow(axis_post_row)

            input_value_row = dict(shared)
            input_value_row[compact.COL_ROW_TYPE] = ROW_INPUT_VALUE
            input_value_row[compact.COL_MAE] = None
            input_value_row[compact.COL_RMSE] = None
            input_value_row[compact.COL_BIAS] = None
            input_value_row[compact.COL_MAE_DELTA] = None
            for column_name, value in zip(input_value_columns, state["input_patch"]):
                input_value_row[column_name] = maybe_round_float_compat(float(value))
            writer.writerow(input_value_row)

            event_row = dict(shared)
            event_row[compact.COL_ROW_TYPE] = ROW_EVENT
            event_row[compact.COL_MAE] = None
            event_row[compact.COL_RMSE] = None
            event_row[compact.COL_BIAS] = None
            event_row[compact.COL_MAE_DELTA] = None
            for column_name, value in zip(input_value_columns, input_event_cells):
                event_row[column_name] = value
            for idx, value in enumerate(forecast_event_cells):
                event_row[f"t+{idx + 1}"] = value
            compact.fill_event_columns(
                event_row,
                input_event_cells,
                forecast_event_cells,
                patch_size,
            )
            writer.writerow(event_row)

            time_row = dict(shared)
            time_row[compact.COL_ROW_TYPE] = ROW_TIME
            time_row[compact.COL_MAE] = None
            time_row[compact.COL_RMSE] = None
            time_row[compact.COL_BIAS] = None
            time_row[compact.COL_MAE_DELTA] = None
            fill_input_time_columns(
                time_row,
                input_column_names=input_value_columns,
                sample_id=sample_id,
                device_id=device_id,
                time_meta=time_meta,
                input_start=int(state["input_start"]),
            )
            fill_forecast_time_columns(
                time_row,
                sample_id=sample_id,
                device_id=device_id,
                time_meta=time_meta,
                forecast_start=int(state["forecast_start"]),
                rollout_horizon_steps=target_horizon_steps,
            )
            writer.writerow(time_row)

            true_row = dict(shared)
            true_row[compact.COL_ROW_TYPE] = compact.ROW_TRUE
            true_row[compact.COL_MAE] = None
            true_row[compact.COL_RMSE] = None
            true_row[compact.COL_BIAS] = None
            true_row[compact.COL_MAE_DELTA] = None
            compact.fill_forecast_columns(
                true_row,
                state["true_future"],
                state["true_future_mask"],
                target_horizon_steps,
            )
            compact.fill_event_columns(
                true_row,
                input_event_cells,
                forecast_event_cells,
                patch_size,
            )
            writer.writerow(true_row)

            pre_row = dict(shared)
            pre_row[compact.COL_ROW_TYPE] = compact.ROW_PRE
            pre_row[compact.COL_MAE] = maybe_round_float_compat(pre_metrics["mae"])
            pre_row[compact.COL_RMSE] = maybe_round_float_compat(pre_metrics["rmse"])
            pre_row[compact.COL_BIAS] = maybe_round_float_compat(pre_metrics["bias"])
            pre_row[compact.COL_MAE_DELTA] = None
            compact.fill_forecast_columns(
                pre_row,
                pre_future,
                state["true_future_mask"],
                target_horizon_steps,
            )
            compact.fill_event_columns(
                pre_row,
                input_event_cells,
                forecast_event_cells,
                patch_size,
            )
            writer.writerow(pre_row)

            post_row = dict(shared)
            post_row[compact.COL_ROW_TYPE] = compact.ROW_POST
            post_row[compact.COL_MAE] = maybe_round_float_compat(post_metrics["mae"])
            post_row[compact.COL_RMSE] = maybe_round_float_compat(post_metrics["rmse"])
            post_row[compact.COL_BIAS] = maybe_round_float_compat(post_metrics["bias"])
            post_row[compact.COL_MAE_DELTA] = maybe_round_float_compat(
                compact.maybe_delta(post_metrics["mae"], pre_metrics["mae"])
            )
            compact.fill_forecast_columns(
                post_row,
                post_future,
                state["true_future_mask"],
                target_horizon_steps,
            )
            compact.fill_event_columns(
                post_row,
                input_event_cells,
                forecast_event_cells,
                patch_size,
            )
            writer.writerow(post_row)
            exported_windows += 1

            curve_key = f"{sample_id}|{device_id}|{exported_windows}"
            raw_curves_dict[curve_key] = {
                "input": state["input_patch"],
                "input_start": state["input_start"],
                "forecast_start": state["forecast_start"],
                "window_index": state["window_index"],
                "true": state["true_future"],
                "direct_true": state["true_future"][:direct_horizon_steps],
                "direct_mask": state["true_future_mask"][:direct_horizon_steps],
                "direct_pre": pre_direct,
                "direct_post": post_direct,
                "pre": pre_future,
                "post": post_future,
                "mask": state["true_future_mask"],
            }



    total_devices = len(selected_indices)
    qualified_windows = len(results)
    qualified_device_ids = set(
        (r["sample_id"], r["device_id"]) for r in results
    )
    qualified_devices = len(qualified_device_ids)
    skipped_devices = total_devices - qualified_devices

    print("\n" + "=" * 85)
    print("Sample Quality Summary")
    print(f"Total selected devices: {total_devices}")
    print(f"Devices with >= 1 valid window: {qualified_devices}")
    print(f"Skipped devices: {skipped_devices}")
    print(f"Total evaluation windows: {qualified_windows} (exported={exported_windows})")
    print(f"  no_valid_points: {skip_counts['no_valid_points']}")
    print(f"  insufficient_total_length: {skip_counts['insufficient_total_length']}")
    print(
        "  insufficient_target_valid_ratio: "
        f"{skip_counts['insufficient_target_valid_ratio']}"
    )
    print(
        "  insufficient_history_valid_ratio: "
        f"{skip_counts['insufficient_history_valid_ratio']}"
    )
    print("Event Lookup Summary")
    print(
        f"  raw_events hit/miss: {event_lookup_stats['raw_hit']}/"
        f"{event_lookup_stats['raw_miss']}"
    )
    print(
        f"  model_events hit/miss: {event_lookup_stats['model_hit']}/"
        f"{event_lookup_stats['model_miss']}"
    )
    print(
        "  windows with input/forecast events: "
        f"{event_lookup_stats['window_with_input_events']}/"
        f"{event_lookup_stats['window_with_forecast_events']}"
    )
    print(
        "  windows with full-context events: "
        f"{event_lookup_stats['window_with_full_context_events']}"
    )
    if (
        event_lookup_stats["window_with_input_events"] == 0
        and event_lookup_stats["window_with_forecast_events"] == 0
    ):
        print(
            "  [WARN] No non-zero events were exported in any displayed 12h+24h window. "
            "Check whether events only appear earlier in the full context, or whether "
            "the onehot time axis/content is misaligned."
        )
    print("Selected Split Patch-Level Loss")
    print(
        f"  devices={patch_eval_summary['selected_devices']} | "
        f"Pre MAE={patch_eval_summary['pre_patch_mae']:.6f} | "
        f"Post MAE={patch_eval_summary['post_patch_mae']:.6f}"
    )
    print("=" * 85)

    if not results:
        print("No valid sequences were found for 24h AR evaluation.")
        return

    res_df = pd.DataFrame(results)
    metrics_rows = compact.build_metrics_rows(
        run_tag=run_tag,
        split_name=args.split,
        checkpoint_kind=resolved_ckpt_kind,
        rollout_horizon_steps=target_horizon_steps,
        chunk_steps=ar_step_size,
        metric_state=metric_state,
        exported_windows=exported_windows,
        valid_forecast_points=valid_forecast_points_total,
    )
    with metrics_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=compact.build_metrics_columns())
        writer.writeheader()
        writer.writerows(metrics_rows)

    all_direct_true = []
    all_direct_pre = []
    all_direct_post = []
    for curve in raw_curves_dict.values():
        valid = curve["direct_mask"] > 0.5
        all_direct_true.extend(curve["direct_true"][valid])
        all_direct_pre.extend(curve["direct_pre"][valid])
        all_direct_post.extend(curve["direct_post"][valid])
    direct_pre_metrics = calc_global_series_metrics(all_direct_true, all_direct_pre)
    direct_post_metrics = calc_global_series_metrics(all_direct_true, all_direct_post)

    windows = {
        "30min": 6,
        "1h": 12,
        "2h": 24,
        "4h": 48,
        "6h": 72,
        "12h": 144,
        "24h": 288,
    }
    global_time_metrics = []
    for name, pts in windows.items():
        if pts > target_horizon_steps:
            continue
        all_true = []
        all_pre = []
        all_post = []
        for curve in raw_curves_dict.values():
            t = curve["true"][:pts]
            p0 = curve["pre"][:pts]
            p1 = curve["post"][:pts]
            m = curve["mask"][:pts]
            valid = m > 0.5
            all_true.extend(t[valid])
            all_pre.extend(p0[valid])
            all_post.extend(p1[valid])

        all_true = np.asarray(all_true)
        all_pre = np.asarray(all_pre)
        all_post = np.asarray(all_post)
        pre_global_mae = np.mean(np.abs(all_true - all_pre)) if len(all_true) else np.nan
        post_global_mae = np.mean(np.abs(all_true - all_post)) if len(all_true) else np.nan
        pre_global_r = (
            pearsonr(all_true, all_pre)[0]
            if len(all_true) > 1
            and np.std(all_true) > 1e-5
            and np.std(all_pre) > 1e-5
            else 0.0
        )
        post_global_r = (
            pearsonr(all_true, all_post)[0]
            if len(all_true) > 1
            and np.std(all_true) > 1e-5
            and np.std(all_post) > 1e-5
            else 0.0
        )
        global_time_metrics.append(
            {
                "Horizon": name,
                "Pre_Global_MAE": pre_global_mae,
                "Pre_Global_R": pre_global_r,
                "Post_Global_MAE": post_global_mae,
                "Post_Global_R": post_global_r,
                "Pre_Avg_Indiv_MAE": res_df[f"Pre_Indiv_MAE_{name}"].mean(),
                "Pre_Avg_Indiv_R": res_df[f"Pre_Indiv_R_{name}"].mean(),
                "Post_Avg_Indiv_MAE": res_df[f"Post_Indiv_MAE_{name}"].mean(),
                "Post_Avg_Indiv_R": res_df[f"Post_Indiv_R_{name}"].mean(),
            }
        )
    direct_window_label = next(
        (name for name, pts in windows.items() if pts == direct_horizon_steps),
        f"{direct_horizon_steps}step",
    )
    rollout_direct_metric = next(
        (item for item in global_time_metrics if item["Horizon"] == direct_window_label),
        None,
    )

    metric_corr_results = []
    print("\n" + "=" * 85)
    print("Population Clinical Correlation")
    for key in ["TIR", "TAR", "TBR", "SB", "SD", "Mean", "CV"]:
        pre_valid = res_df[[f"True_{key}", f"Pre_{key}"]].dropna()
        post_valid = res_df[[f"True_{key}", f"Post_{key}"]].dropna()

        if (
            len(pre_valid) > 1
            and pre_valid[f"True_{key}"].std() > 1e-5
            and pre_valid[f"Pre_{key}"].std() > 1e-5
        ):
            pre_r, _ = pearsonr(pre_valid[f"True_{key}"], pre_valid[f"Pre_{key}"])
        else:
            pre_r = np.nan

        if (
            len(post_valid) > 1
            and post_valid[f"True_{key}"].std() > 1e-5
            and post_valid[f"Post_{key}"].std() > 1e-5
        ):
            post_r, _ = pearsonr(post_valid[f"True_{key}"], post_valid[f"Post_{key}"])
        else:
            post_r = np.nan

        print(f"{key:<4} | Pre R = {pre_r:.4f} | Post R = {post_r:.4f}")
        metric_corr_results.append(
            {
                "Metric": key,
                "Pre_Pearson_R": pre_r,
                "Post_Pearson_R": post_r,
            }
        )
    print("=" * 85)

    result_prefix = f"{run_tag}_event_ar24h_fullcontext_{args.split}_step{ar_step_size}"
    alignment_summary_rows = [
        {
            "scope": "selected_split_patch_level",
            "split": args.split,
            "devices": patch_eval_summary["selected_devices"],
            "horizon_steps": forecast_steps,
            "pre_mae": patch_eval_summary["pre_patch_mae"],
            "post_mae": patch_eval_summary["post_patch_mae"],
            "pre_r": np.nan,
            "post_r": np.nan,
            "note": "All selected patches, same evaluation style as EventFinetune_main.py",
        },
        {
            "scope": "final_window_direct_single_shot",
            "split": args.split,
            "devices": qualified_devices,
            "horizon_steps": direct_horizon_steps,
            "pre_mae": direct_pre_metrics["mae"],
            "post_mae": direct_post_metrics["mae"],
            "pre_r": direct_pre_metrics["r"],
            "post_r": direct_post_metrics["r"],
            "note": f"Direct forecast on {qualified_windows} windows from {qualified_devices} devices, without autoregressive rollout",
        },
    ]
    if rollout_direct_metric is not None:
        alignment_summary_rows.append(
            {
                "scope": f"autoregressive_rollout_{rollout_direct_metric['Horizon']}",
                "split": args.split,
                "devices": qualified_devices,
                "horizon_steps": direct_horizon_steps,
                "pre_mae": rollout_direct_metric["Pre_Global_MAE"],
                "post_mae": rollout_direct_metric["Post_Global_MAE"],
                "pre_r": rollout_direct_metric["Pre_Global_R"],
                "post_r": rollout_direct_metric["Post_Global_R"],
                "note": f"Autoregressive rollout on {qualified_windows} windows from {qualified_devices} devices",
            }
        )
    pd.DataFrame(metric_corr_results).to_csv(
        output_dir / f"{result_prefix}_pop_metric_corr.csv",
        index=False,
    )
    pd.DataFrame(global_time_metrics).to_csv(
        output_dir / f"{result_prefix}_global_vs_indiv.csv",
        index=False,
    )
    pd.DataFrame(alignment_summary_rows).to_csv(
        output_dir / f"{result_prefix}_alignment_summary.csv",
        index=False,
    )
    res_df.to_csv(
        output_dir / f"{result_prefix}_patient_metrics.csv",
        index=False,
    )
    ar_group_metrics_rows = []
    ar_group_all_mae_pre = []
    ar_group_all_mae_post = []
    for g in range(num_ar_groups):
        pre_preds = np.array(ar_group_pool["pre_pred"][g])
        post_preds = np.array(ar_group_pool["post_pred"][g])
        truths = np.array(ar_group_pool["true"][g])
        n_pts = len(truths)
        if n_pts < 2:
            pre_mae, post_mae, pre_r, post_r = np.nan, np.nan, np.nan, np.nan
        else:
            pre_mae = float(np.mean(np.abs(pre_preds - truths)))
            post_mae = float(np.mean(np.abs(post_preds - truths)))
            pre_r = float(pearsonr(pre_preds, truths)[0]) if np.std(pre_preds) > 1e-5 and np.std(truths) > 1e-5 else 0.0
            post_r = float(pearsonr(post_preds, truths)[0]) if np.std(post_preds) > 1e-5 and np.std(truths) > 1e-5 else 0.0
            ar_group_all_mae_pre.append(pre_mae)
            ar_group_all_mae_post.append(post_mae)
        ar_group_metrics_rows.append({
            "group_index": g,
            "n_points": n_pts,
            "pre_mae": pre_mae,
            "post_mae": post_mae,
            "pre_r": pre_r,
            "post_r": post_r,
        })
    if ar_group_metrics_rows:
        pd.DataFrame(ar_group_metrics_rows).to_csv(
            output_dir / f"{result_prefix}_ar_group_metrics.csv",
            index=False,
        )
    torch.save(
        raw_curves_dict,
        output_dir / f"{result_prefix}_curves.pt",
    )

    print("\nGlobal vs Avg Individual")
    for metric in global_time_metrics:
        print(
            f"{metric['Horizon']:>5} | "
            f"Pre Global R={metric['Pre_Global_R']:.4f} "
            f"(MAE={metric['Pre_Global_MAE']:.3f}) | "
            f"Post Global R={metric['Post_Global_R']:.4f} "
            f"(MAE={metric['Post_Global_MAE']:.3f})"
        )

    print("\nDirect Final-Window Forecast")
    print(
        f"{direct_horizon_steps:>5} steps | "
        f"Pre Global R={direct_pre_metrics['r']:.4f} "
        f"(MAE={direct_pre_metrics['mae']:.3f}) | "
        f"Post Global R={direct_post_metrics['r']:.4f} "
        f"(MAE={direct_post_metrics['mae']:.3f})"
    )

    if len(ar_group_all_mae_pre) > 0:
        print("\n" + "=" * 85)
        print("AR Iteration Group Metrics (pooled, point-weighted per group)")
        print(f"  Groups rejected (<50% valid): {ar_group_rejected}")
        pooled_pre = np.array(ar_group_all_mae_pre)
        pooled_post = np.array(ar_group_all_mae_post)
        print(f"  Pre  Mean MAE across groups={np.mean(pooled_pre):.4f} Std={np.std(pooled_pre):.4f}")
        print(f"  Post Mean MAE across groups={np.mean(pooled_post):.4f} Std={np.std(pooled_post):.4f}")
        print(f"  {'Iter':>5} | {'Ctx Quality':>12} | {'N pts':>8} | {'Pre MAE':>10} | {'Post MAE':>10} | {'Pre R':>8} | {'Post R':>8}")
        for g in range(min(num_ar_groups, 12)):
            row = ar_group_metrics_rows[g]
            ctx_desc = "real" if g == 0 else "real+synth" if g <= 2 else "mostly synth"
            label = f"g{g}"
            v = lambda x: x if not np.isnan(x) else 0.0
            print(
                f"  {label:>5} | {ctx_desc:>12} | {row['n_points']:>8} | "
                f"{v(row['pre_mae']):10.4f} | {v(row['post_mae']):10.4f} | "
                f"{v(row['pre_r']):8.4f} | {v(row['post_r']):8.4f}"
            )
        if num_ar_groups > 12:
            print(f"  ... ({num_ar_groups - 12} more groups)")
        print("=" * 85)

    print(f"\nSaved results to: {output_dir}")
    print(f"Compact CSV: {main_csv_path}")
    print(f"Full-Context Axis CSV: {axis_csv_path}")
    print(f"Metrics CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()


