from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent


TASK_SPECS = {
    "CGMacros": REPO_ROOT
    / "data_event"
    / "CGMacros"
    / "CGMacros_data_withevent_simple_5min_3456_libre_mmol_from00.csv",
    "HPP": REPO_ROOT
    / "data_event"
    / "HPP"
    / "participant_cgm_data_withevent_full_5min_3456_from00.csv",
    "zhao_simple": REPO_ROOT
    / "data_event"
    / "zhao"
    / "zhao_data_withevent_simple_5min_3456_from00.csv",
    "zhao_full": REPO_ROOT
    / "data_event"
    / "zhao"
    / "zhao_data_withevent_full_5min_3456_from00.csv",
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone PatchTST baseline for the wide CGM CSV format used in this repo."
        )
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=sorted(TASK_SPECS.keys()),
        default="CGMacros",
        help="Shortcut for the built-in dataset CSV paths.",
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default=None,
        help="Optional override for the glucose wide CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "data_event" / "patchtst_outputs"),
    )
    parser.add_argument("--run-name", type=str, default="patchtst")
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-mode",
        type=str,
        choices=["ordered", "shuffled"],
        default="ordered",
        help="Use ordered to mirror the current event finetune split behavior.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--glucose-scale", type=float, default=1.0)
    parser.add_argument("--lookback-len", type=int, default=144)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument(
        "--window-stride",
        type=int,
        default=3,
        help="Sliding-window stride over the raw sequence.",
    )
    parser.add_argument(
        "--min-patch-valid-ratio",
        type=float,
        default=0.5,
        help="Minimum non-NaN ratio required inside the lookback window.",
    )
    parser.add_argument(
        "--patch-len",
        type=int,
        default=12,
        help="Internal PatchTST patch length.",
    )
    parser.add_argument(
        "--patch-stride",
        type=int,
        default=6,
        help="Internal PatchTST patch stride.",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--use-revin", type=str2bool, default=True)
    parser.add_argument(
        "--prediction-split",
        type=str,
        choices=["train", "val", "all"],
        default="val",
        help="Which split to export predictions and metrics for after training.",
    )
    return parser


def resolve_data_csv(args: argparse.Namespace) -> Path:
    if args.data_csv:
        return Path(args.data_csv).expanduser().resolve()
    return TASK_SPECS[args.task].expanduser().resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fill_nan_lin(x: np.ndarray) -> np.ndarray:
    out = x.copy().astype(np.float32)
    if np.isnan(out).all():
        return np.zeros_like(out, dtype=np.float32)
    missing = np.isnan(out)
    if not missing.any():
        return out
    idx = np.arange(len(out))
    good = ~missing
    if good.sum() == 1:
        out[missing] = float(out[good][0])
        return out
    out[missing] = np.interp(idx[missing], idx[good], out[good]).astype(np.float32)
    return out


@dataclass
class ForecastCorpus:
    data_frame: pd.DataFrame
    ids: list[str]
    device_ids: list[str]
    raw_indices: list[int]
    glucose_raw: np.ndarray
    glucose_filled: np.ndarray
    valid_lengths: list[int]
    device_patch_starts: list[list[int]]
    start_times: list[pd.Timestamp | None]
    steps_per_point_minutes: int = 5

    def __len__(self) -> int:
        return len(self.device_ids)


def load_corpus(
    data_csv: Path,
    lookback_len: int,
    pred_len: int,
    window_stride: int,
    min_patch_valid_ratio: float,
    glucose_scale: float,
) -> ForecastCorpus:
    df = pd.read_csv(data_csv)

    glucose_cols = sorted(
        [col for col in df.columns if str(col).isdigit()],
        key=lambda col: int(col),
    )
    if not glucose_cols:
        raise ValueError(f"No numeric glucose columns found in {data_csv}")

    if "id" in df.columns:
        ids_all = df["id"].astype(str).tolist()
    else:
        ids_all = df.iloc[:, 0].astype(str).tolist()

    if "device_id" in df.columns:
        device_ids_all = df["device_id"].astype(str).tolist()
    else:
        device_ids_all = ids_all[:]

    if "starttime" in df.columns:
        start_times_all = pd.to_datetime(df["starttime"], errors="coerce").tolist()
    else:
        start_times_all = [pd.NaT for _ in range(len(df))]

    glucose_all = df[glucose_cols].to_numpy(dtype=np.float32) * float(glucose_scale)

    kept_rows: list[int] = []
    kept_ids: list[str] = []
    kept_device_ids: list[str] = []
    kept_raw: list[np.ndarray] = []
    kept_filled: list[np.ndarray] = []
    kept_valid_lengths: list[int] = []
    kept_starts: list[list[int]] = []
    kept_start_times: list[pd.Timestamp | None] = []

    min_ratio = float(min_patch_valid_ratio)
    required_len = int(lookback_len) + int(pred_len)

    for row_idx in range(len(df)):
        seq_raw = glucose_all[row_idx].astype(np.float32)
        valid = np.flatnonzero(~np.isnan(seq_raw))
        if valid.size == 0:
            continue

        valid_len = int(valid[-1]) + 1
        patch_starts: list[int] = []
        if valid_len >= required_len:
            max_start = valid_len - required_len
            for start in range(0, max_start + 1, int(window_stride)):
                raw_patch = seq_raw[start : start + int(lookback_len)]
                valid_ratio = float(np.sum(~np.isnan(raw_patch))) / float(lookback_len)
                if valid_ratio >= min_ratio:
                    patch_starts.append(int(start))
        if not patch_starts:
            continue

        kept_rows.append(int(row_idx))
        kept_ids.append(str(ids_all[row_idx]))
        kept_device_ids.append(str(device_ids_all[row_idx]))
        kept_raw.append(seq_raw)
        kept_filled.append(fill_nan_lin(seq_raw))
        kept_valid_lengths.append(valid_len)
        start_time = start_times_all[row_idx]
        kept_start_times.append(None if pd.isna(start_time) else start_time)
        kept_starts.append(patch_starts)

    if not kept_raw:
        raise ValueError(
            "No valid devices remain after patch filtering. "
            "Adjust lookback/pred length or min_patch_valid_ratio."
        )

    kept_df = df.iloc[kept_rows].copy().reset_index(drop=True)
    return ForecastCorpus(
        data_frame=kept_df,
        ids=kept_ids,
        device_ids=kept_device_ids,
        raw_indices=kept_rows,
        glucose_raw=np.stack(kept_raw, axis=0).astype(np.float32),
        glucose_filled=np.stack(kept_filled, axis=0).astype(np.float32),
        valid_lengths=kept_valid_lengths,
        device_patch_starts=kept_starts,
        start_times=kept_start_times,
    )


def build_split_map(
    n_devices: int,
    train_ratio: float,
    split_mode: str,
    seed: int,
) -> dict[str, list[int]]:
    if n_devices < 2:
        raise ValueError(
            "At least 2 valid devices are required to build a train/val split."
        )

    ordered = list(range(n_devices))
    if split_mode == "shuffled":
        rng = random.Random(seed)
        rng.shuffle(ordered)

    train_count = int(train_ratio * n_devices)
    train_count = max(1, min(train_count, n_devices - 1))
    train_indices = ordered[:train_count]
    val_indices = ordered[train_count:]

    return {
        "train": list(train_indices),
        "val": list(val_indices),
        "all": list(range(n_devices)),
    }


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        corpus: ForecastCorpus,
        device_indices: list[int],
        lookback_len: int,
        pred_len: int,
    ) -> None:
        self.corpus = corpus
        self.device_indices = list(device_indices)
        self.lookback_len = int(lookback_len)
        self.pred_len = int(pred_len)
        self.samples: list[tuple[int, int, int]] = []

        for device_idx in self.device_indices:
            starts = corpus.device_patch_starts[device_idx]
            for patch_index, patch_start in enumerate(starts):
                self.samples.append(
                    (int(device_idx), int(patch_index), int(patch_start))
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        device_idx, patch_index, patch_start = self.samples[idx]
        lookback_end = patch_start + self.lookback_len
        forecast_end = lookback_end + self.pred_len

        x = self.corpus.glucose_filled[device_idx, patch_start:lookback_end].astype(
            np.float32
        )
        y_raw = self.corpus.glucose_raw[device_idx, lookback_end:forecast_end].astype(
            np.float32
        )
        mask = (~np.isnan(y_raw)).astype(np.float32)
        y = y_raw.copy()
        y[np.isnan(y)] = 0.0

        forecast_start_time = ""
        base_time = self.corpus.start_times[device_idx]
        if base_time is not None:
            forecast_start_time = str(
                base_time
                + timedelta(
                    minutes=self.corpus.steps_per_point_minutes * int(lookback_end)
                )
            )

        return {
            "inputs": torch.from_numpy(x),
            "targets": torch.from_numpy(y.astype(np.float32)),
            "mask": torch.from_numpy(mask.astype(np.float32)),
            "sample_id": self.corpus.ids[device_idx],
            "device_id": self.corpus.device_ids[device_idx],
            "device_index": int(device_idx),
            "patch_index": int(patch_index),
            "patch_start": int(patch_start),
            "forecast_start_index": int(lookback_end),
            "forecast_start_time": forecast_start_time,
        }


class PatchTSTBaseline(nn.Module):
    def __init__(
        self,
        context_len: int,
        pred_len: int,
        patch_len: int,
        patch_stride: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        head_dropout: float,
        use_revin: bool,
    ) -> None:
        super().__init__()
        if context_len < patch_len:
            raise ValueError("context_len must be >= patch_len")
        if patch_len <= 0 or patch_stride <= 0:
            raise ValueError("patch_len and patch_stride must be positive")

        self.context_len = int(context_len)
        self.pred_len = int(pred_len)
        self.patch_len = int(patch_len)
        self.patch_stride = int(patch_stride)
        self.d_model = int(d_model)
        self.use_revin = bool(use_revin)
        self.eps = 1e-5

        self.num_patches = 1 + (self.context_len - self.patch_len) // self.patch_stride
        if self.num_patches <= 0:
            raise ValueError("Invalid patch config produced zero patches.")

        self.patch_embed = nn.Linear(self.patch_len, self.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        self.input_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(e_layers))
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(head_dropout),
            nn.Linear(self.num_patches * self.d_model, self.pred_len),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, L], got {tuple(x.shape)}")

        z = x.unsqueeze(1)
        if self.use_revin:
            loc = z.mean(dim=-1, keepdim=True)
            scale = z.std(dim=-1, unbiased=False, keepdim=True).clamp_min(self.eps)
            z = (z - loc) / scale
        else:
            loc = None
            scale = None

        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        patches = patches.contiguous()
        batch_size, channels, num_patches, patch_width = patches.shape
        patches = patches.view(batch_size * channels, num_patches, patch_width)

        tokens = self.patch_embed(patches)
        tokens = tokens + self.pos_embed[:, :num_patches, :]
        tokens = self.input_dropout(tokens)
        tokens = self.encoder(tokens)

        preds = self.head(tokens).view(batch_size, channels, self.pred_len)
        preds = preds.squeeze(1)

        if self.use_revin:
            preds = preds * scale[:, 0, 0].unsqueeze(-1) + loc[:, 0, 0].unsqueeze(-1)
        return preds


def masked_mae_components(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.float()
    abs_sum = torch.sum(torch.abs(pred - target) * mask)
    count = torch.sum(mask)
    return abs_sum, count


def masked_mse_components(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.float()
    sq_sum = torch.sum((pred - target) ** 2 * mask)
    count = torch.sum(mask)
    return sq_sum, count


def compute_basic_metrics(
    preds: np.ndarray, targets: np.ndarray, masks: np.ndarray
) -> dict[str, float | int | None]:
    valid = masks.astype(bool)
    valid_points = int(valid.sum())
    if valid_points == 0:
        return {
            "mae": None,
            "mse": None,
            "rmse": None,
            "bias": None,
            "valid_points": 0,
        }

    diff = preds[valid] - targets[valid]
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": bias,
        "valid_points": valid_points,
    }


def compute_horizon_rows(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_len = preds.shape[1] if preds.ndim == 2 else 0
    for step in range(pred_len):
        valid = masks[:, step].astype(bool)
        n_valid = int(valid.sum())
        if n_valid == 0:
            rows.append(
                {
                    "horizon_step": step + 1,
                    "valid_points": 0,
                    "mae": None,
                    "rmse": None,
                    "bias": None,
                }
            )
            continue
        diff = preds[valid, step] - targets[valid, step]
        mse = float(np.mean(diff**2))
        rows.append(
            {
                "horizon_step": step + 1,
                "valid_points": n_valid,
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(mse)),
                "bias": float(np.mean(diff)),
            }
        )
    return rows


def clarke_error_grid_analysis(
    ref_values: np.ndarray,
    pred_values: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(ref_values, dtype=np.float32) * 18.0
    pred = np.asarray(pred_values, dtype=np.float32) * 18.0

    n = len(ref)
    if n == 0:
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}

    zone_a = (np.abs(pred - ref) <= 0.2 * ref) | ((pred <= 70) & (ref <= 70))
    zone_e = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    zone_c = (((ref >= 70) & (ref <= 290)) & (pred >= ref + 110)) | (
        ((ref >= 130) & (ref <= 180)) & (pred <= (7.0 / 5.0) * ref - 182)
    )
    zone_d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | (
        (ref <= 70) & (pred >= 70) & (pred <= 180)
    )

    zone_e = zone_e & ~zone_a
    zone_c = zone_c & ~zone_a & ~zone_e
    zone_d = zone_d & ~zone_a & ~zone_e & ~zone_c
    zone_b = ~(zone_a | zone_e | zone_c | zone_d)

    return {
        "A": float(np.sum(zone_a) / n * 100.0),
        "B": float(np.sum(zone_b) / n * 100.0),
        "C": float(np.sum(zone_c) / n * 100.0),
        "D": float(np.sum(zone_d) / n * 100.0),
        "E": float(np.sum(zone_e) / n * 100.0),
    }


def compute_forecast_metrics_local(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    hyper_thr: float = 10.0,
    hypo_thr: float = 3.9,
) -> dict[str, Any]:
    valid = masks > 0.5
    pred_len = preds.shape[1]

    if valid.sum() == 0:
        return {
            "mae_per_step": [],
            "grouped_mae": {},
            "ceg": {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
            "ceg_10min": {},
            "ceg_30min": {},
            "hyper": {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0},
            "hypo": {"acc": 0.0, "prec": 0.0, "rec": 0.0, "f1": 0.0},
        }

    ceg = clarke_error_grid_analysis(targets[valid], preds[valid])

    ceg_10min: dict[str, float] = {}
    if pred_len >= 2:
        valid_10 = masks[:, :2] > 0.5
        ceg_10min = clarke_error_grid_analysis(
            targets[:, :2][valid_10], preds[:, :2][valid_10]
        )

    ceg_30min: dict[str, float] = {}
    if pred_len >= 6:
        valid_30 = masks[:, :6] > 0.5
        ceg_30min = clarke_error_grid_analysis(
            targets[:, :6][valid_30], preds[:, :6][valid_30]
        )

    abs_err = np.abs(preds - targets) * masks
    denom = masks.sum(axis=0).clip(min=1.0)
    mae_per_step = (abs_err.sum(axis=0) / denom).astype(np.float64)

    groups = {
        "0-10min": (0, 2),
        "10-30min": (2, 6),
        "30-60min": (6, 12),
    }
    grouped_mae: dict[str, float] = {}
    for name, (start, end) in groups.items():
        if end <= pred_len and end > start:
            grouped_mae[name] = float(mae_per_step[start:end].mean())

    pred_hyper = (preds > hyper_thr).astype(np.float32)
    true_hyper = (targets > hyper_thr).astype(np.float32)
    pred_hypo = (preds < hypo_thr).astype(np.float32)
    true_hypo = (targets < hypo_thr).astype(np.float32)

    def prf(pred_bin: np.ndarray, true_bin: np.ndarray) -> dict[str, float]:
        mask = masks.astype(np.float32)
        tp = float((pred_bin * true_bin * mask).sum())
        tn = float(((1.0 - pred_bin) * (1.0 - true_bin) * mask).sum())
        fp = float((pred_bin * (1.0 - true_bin) * mask).sum())
        fn = float(((1.0 - pred_bin) * true_bin * mask).sum())
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

    return {
        "mae_per_step": [float(value) for value in mae_per_step],
        "grouped_mae": grouped_mae,
        "ceg": ceg,
        "ceg_10min": ceg_10min,
        "ceg_30min": ceg_30min,
        "hyper": prf(pred_hyper, true_hyper),
        "hypo": prf(pred_hypo, true_hypo),
    }


def maybe_compute_extra_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    if preds.size == 0 or float(masks.sum()) <= 0.0:
        return {}

    metrics = compute_forecast_metrics_local(preds, targets, masks)

    grouped_mae = {
        str(key): float(value) for key, value in metrics.get("grouped_mae", {}).items()
    }
    ceg = {str(key): float(value) for key, value in metrics.get("ceg", {}).items()}
    ceg_10 = metrics.get("ceg_10min") or {}
    ceg_30 = metrics.get("ceg_30min") or {}
    hyper = {str(key): float(value) for key, value in metrics.get("hyper", {}).items()}
    hypo = {str(key): float(value) for key, value in metrics.get("hypo", {}).items()}
    mae_per_step = [float(value) for value in metrics.get("mae_per_step", [])]

    return {
        "mae_per_step": mae_per_step,
        "grouped_mae": grouped_mae,
        "ceg": ceg,
        "ceg_10min": {str(k): float(v) for k, v in ceg_10.items()},
        "ceg_30min": {str(k): float(v) for k, v in ceg_30.items()},
        "hyper_event_metrics": hyper,
        "hypo_event_metrics": hypo,
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    collect_rows: bool = False,
) -> dict[str, Any]:
    model.eval()
    sq_sum_total = 0.0
    mae_sum_total = 0.0
    point_count_total = 0.0
    pred_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    mask_batches: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(inputs)

            sq_sum, count_sq = masked_mse_components(preds, targets, mask)
            mae_sum, count_mae = masked_mae_components(preds, targets, mask)
            sq_sum_total += float(sq_sum.item())
            mae_sum_total += float(mae_sum.item())
            point_count_total += float(count_sq.item())
            if float(count_sq.item()) != float(count_mae.item()):
                raise RuntimeError("Mismatched mask counts during evaluation.")

            preds_np = preds.detach().cpu().numpy().astype(np.float32)
            targets_np = targets.detach().cpu().numpy().astype(np.float32)
            mask_np = mask.detach().cpu().numpy().astype(np.float32)

            pred_batches.append(preds_np)
            target_batches.append(targets_np)
            mask_batches.append(mask_np)

            if collect_rows:
                batch_size = preds_np.shape[0]
                horizon = preds_np.shape[1]
                for sample_idx in range(batch_size):
                    forecast_start_index = int(
                        batch["forecast_start_index"][sample_idx]
                    )
                    forecast_start_time_value = batch["forecast_start_time"][sample_idx]
                    parsed_forecast_start_time = pd.to_datetime(
                        forecast_start_time_value, errors="coerce"
                    )
                    row = {
                        "sample_id": batch["sample_id"][sample_idx],
                        "device_id": batch["device_id"][sample_idx],
                        "device_index": int(batch["device_index"][sample_idx]),
                        "patch_index": int(batch["patch_index"][sample_idx]),
                        "patch_start": int(batch["patch_start"][sample_idx]),
                        "forecast_start_index": forecast_start_index,
                        "forecast_start_time": forecast_start_time_value,
                    }
                    for step in range(horizon):
                        valid = bool(mask_np[sample_idx, step] > 0.5)
                        row[f"mask_t+{step + 1}"] = int(valid)
                        if valid:
                            true_value = float(targets_np[sample_idx, step])
                            pred_value = float(preds_np[sample_idx, step])
                            abs_error = abs(pred_value - true_value)
                            row[f"true_t+{step + 1}"] = true_value
                            row[f"pred_t+{step + 1}"] = pred_value
                            row[f"abs_err_t+{step + 1}"] = abs_error
                            target_time = ""
                            if not pd.isna(parsed_forecast_start_time):
                                target_time = str(
                                    parsed_forecast_start_time
                                    + timedelta(minutes=5 * step)
                                )
                            long_rows.append(
                                {
                                    "sample_id": batch["sample_id"][sample_idx],
                                    "device_id": batch["device_id"][sample_idx],
                                    "device_index": int(
                                        batch["device_index"][sample_idx]
                                    ),
                                    "patch_index": int(
                                        batch["patch_index"][sample_idx]
                                    ),
                                    "patch_start": int(
                                        batch["patch_start"][sample_idx]
                                    ),
                                    "forecast_start_index": forecast_start_index,
                                    "forecast_start_time": forecast_start_time_value,
                                    "target_time_index": forecast_start_index + step,
                                    "target_time": target_time,
                                    "horizon_step": step + 1,
                                    "true_value": true_value,
                                    "pred_value": pred_value,
                                    "abs_error": abs_error,
                                }
                            )
                        else:
                            row[f"true_t+{step + 1}"] = ""
                            row[f"pred_t+{step + 1}"] = ""
                            row[f"abs_err_t+{step + 1}"] = ""
                    rows.append(row)

    if pred_batches:
        preds_all = np.concatenate(pred_batches, axis=0)
        targets_all = np.concatenate(target_batches, axis=0)
        masks_all = np.concatenate(mask_batches, axis=0)
    else:
        preds_all = np.zeros((0, 0), dtype=np.float32)
        targets_all = np.zeros((0, 0), dtype=np.float32)
        masks_all = np.zeros((0, 0), dtype=np.float32)

    metrics = compute_basic_metrics(preds_all, targets_all, masks_all)
    metrics.update(maybe_compute_extra_metrics(preds_all, targets_all, masks_all))

    return {
        "loss_mse": sq_sum_total / max(point_count_total, 1.0),
        "loss_mae": mae_sum_total / max(point_count_total, 1.0),
        "preds": preds_all,
        "targets": targets_all,
        "masks": masks_all,
        "metrics": metrics,
        "prediction_rows": rows,
        "prediction_long_rows": long_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def export_split_artifacts(
    corpus: ForecastCorpus,
    split_map: dict[str, list[int]],
    out_dir: Path,
    run_tag: str,
) -> dict[str, Any]:
    index_to_split: dict[int, str] = {}
    for split_name in ("train", "val"):
        for device_index in split_map[split_name]:
            index_to_split[int(device_index)] = split_name

    assignment_rows: list[dict[str, Any]] = []
    for device_index in split_map["all"]:
        assignment_rows.append(
            {
                "dataset_index": int(device_index),
                "split": index_to_split.get(int(device_index), "all"),
                "id": corpus.ids[device_index],
                "device_id": corpus.device_ids[device_index],
                "valid_length": int(corpus.valid_lengths[device_index]),
                "num_windows": int(len(corpus.device_patch_starts[device_index])),
            }
        )

    assignment_path = out_dir / f"{run_tag}_split_assignment.csv"
    write_csv(
        assignment_path,
        assignment_rows,
        ["dataset_index", "split", "id", "device_id", "valid_length", "num_windows"],
    )

    split_exports: dict[str, Any] = {}
    for split_name, device_indices in split_map.items():
        split_df = corpus.data_frame.iloc[device_indices].copy().reset_index(drop=True)
        split_path = out_dir / f"{run_tag}_{split_name}_data.csv"
        split_df.to_csv(split_path, index=False, encoding="utf-8")
        split_exports[split_name] = {
            "num_devices": len(device_indices),
            "num_windows": int(
                sum(len(corpus.device_patch_starts[idx]) for idx in device_indices)
            ),
            "data_csv": str(split_path),
        }

    summary = {
        "assignment_path": str(assignment_path),
        "splits": split_exports,
    }
    summary_path = out_dir / f"{run_tag}_split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "assignment_path": str(assignment_path),
        "summary_path": str(summary_path),
        "splits": split_exports,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_val_mae: float,
    args_dict: dict[str, Any],
) -> None:
    payload = {
        "epoch": int(epoch),
        "best_val_mae": float(best_val_mae),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "args": args_dict,
    }
    torch.save(payload, path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not 0.0 < float(args.train_ratio) < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 < float(args.min_patch_valid_ratio) <= 1.0:
        raise ValueError("min_patch_valid_ratio must be in (0, 1].")
    for name in [
        "epochs",
        "batch_size",
        "lookback_len",
        "pred_len",
        "window_stride",
        "patch_len",
        "patch_stride",
        "d_model",
        "n_heads",
        "e_layers",
        "d_ff",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")

    set_seed(int(args.seed))

    data_csv = resolve_data_csv(args)
    if not data_csv.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_csv}")

    dataset_name = args.dataset_name or args.task
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    corpus = load_corpus(
        data_csv=data_csv,
        lookback_len=int(args.lookback_len),
        pred_len=int(args.pred_len),
        window_stride=int(args.window_stride),
        min_patch_valid_ratio=float(args.min_patch_valid_ratio),
        glucose_scale=float(args.glucose_scale),
    )

    split_map = build_split_map(
        n_devices=len(corpus),
        train_ratio=float(args.train_ratio),
        split_mode=str(args.split_mode),
        seed=int(args.seed),
    )

    train_ds = SlidingWindowDataset(
        corpus=corpus,
        device_indices=split_map["train"],
        lookback_len=int(args.lookback_len),
        pred_len=int(args.pred_len),
    )
    val_ds = SlidingWindowDataset(
        corpus=corpus,
        device_indices=split_map["val"],
        lookback_len=int(args.lookback_len),
        pred_len=int(args.pred_len),
    )
    export_ds = SlidingWindowDataset(
        corpus=corpus,
        device_indices=split_map[str(args.prediction_split)],
        lookback_len=int(args.lookback_len),
        pred_len=int(args.pred_len),
    )

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"Empty split detected: train_windows={len(train_ds)}, val_windows={len(val_ds)}"
        )

    run_tag = (
        f"{args.run_name}_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir = Path(args.out_dir).expanduser().resolve() / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = out_dir / f"{run_tag}_resolved_config.json"
    args_dict = vars(args).copy()
    args_dict["data_csv"] = str(data_csv)
    args_dict["device"] = str(device)
    resolved_config_path.write_text(
        json.dumps(args_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    split_artifacts = export_split_artifacts(
        corpus=corpus,
        split_map=split_map,
        out_dir=out_dir,
        run_tag=run_tag,
    )

    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    export_loader = DataLoader(
        export_ds, shuffle=False, drop_last=False, **loader_kwargs
    )

    model = PatchTSTBaseline(
        context_len=int(args.lookback_len),
        pred_len=int(args.pred_len),
        patch_len=int(args.patch_len),
        patch_stride=int(args.patch_stride),
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        e_layers=int(args.e_layers),
        d_ff=int(args.d_ff),
        dropout=float(args.dropout),
        head_dropout=float(args.head_dropout),
        use_revin=bool(args.use_revin),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improve = 0
    history_rows: list[dict[str, Any]] = []

    best_ckpt_path = out_dir / f"{run_tag}_best.pt"
    last_ckpt_path = out_dir / f"{run_tag}_last.pt"

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_sq_sum = 0.0
        train_mae_sum = 0.0
        train_point_count = 0.0

        for batch in train_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(inputs)
            sq_sum, count = masked_mse_components(preds, targets, mask)
            mae_sum, mae_count = masked_mae_components(preds, targets, mask)
            if float(count.item()) != float(mae_count.item()):
                raise RuntimeError("Mismatched mask counts during training.")

            loss = sq_sum / count.clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_sq_sum += float(sq_sum.item())
            train_mae_sum += float(mae_sum.item())
            train_point_count += float(count.item())

        train_loss_mse = train_sq_sum / max(train_point_count, 1.0)
        train_loss_mae = train_mae_sum / max(train_point_count, 1.0)

        val_result = evaluate_model(model=model, dataloader=val_loader, device=device)
        val_metrics = val_result["metrics"]
        val_mae = val_metrics["mae"]
        if val_mae is None:
            raise RuntimeError(
                "Validation MAE is None. No valid forecast points found."
            )

        history_rows.append(
            {
                "epoch": int(epoch),
                "train_loss_mse": float(train_loss_mse),
                "train_loss_mae": float(train_loss_mae),
                "val_loss_mse": float(val_result["loss_mse"]),
                "val_loss_mae": float(val_result["loss_mae"]),
                "val_mae": float(val_metrics["mae"]),
                "val_rmse": float(val_metrics["rmse"]),
                "val_bias": float(val_metrics["bias"]),
            }
        )

        improved = float(val_mae) < float(best_val_mae)
        if improved:
            best_val_mae = float(val_mae)
            best_epoch = int(epoch)
            epochs_without_improve = 0
            save_checkpoint(
                path=best_ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mae=best_val_mae,
                args_dict=args_dict,
            )
        else:
            epochs_without_improve += 1

        print(
            f"[Epoch {epoch:03d}] "
            f"train_mae={train_loss_mae:.4f} "
            f"val_mae={float(val_metrics['mae']):.4f} "
            f"val_rmse={float(val_metrics['rmse']):.4f}"
        )

        if epochs_without_improve >= int(args.patience):
            print(
                f"[EARLY_STOP] epoch={epoch} best_epoch={best_epoch} best_val_mae={best_val_mae:.4f}"
            )
            break

    save_checkpoint(
        path=last_ckpt_path,
        model=model,
        optimizer=optimizer,
        epoch=len(history_rows),
        best_val_mae=best_val_mae,
        args_dict=args_dict,
    )

    try:
        best_payload = torch.load(
            best_ckpt_path, map_location=device, weights_only=False
        )
    except TypeError:
        best_payload = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_payload["model_state"])

    export_result = evaluate_model(
        model=model,
        dataloader=export_loader,
        device=device,
        collect_rows=True,
    )

    prediction_rows = export_result["prediction_rows"]
    prediction_long_rows = export_result["prediction_long_rows"]
    pred_len = int(args.pred_len)
    prediction_headers = [
        "sample_id",
        "device_id",
        "device_index",
        "patch_index",
        "patch_start",
        "forecast_start_index",
        "forecast_start_time",
    ]
    for step in range(pred_len):
        prediction_headers.append(f"true_t+{step + 1}")
    for step in range(pred_len):
        prediction_headers.append(f"pred_t+{step + 1}")
    for step in range(pred_len):
        prediction_headers.append(f"abs_err_t+{step + 1}")
    for step in range(pred_len):
        prediction_headers.append(f"mask_t+{step + 1}")

    predictions_path = out_dir / f"{run_tag}_predictions_{args.prediction_split}.csv"
    write_csv(predictions_path, prediction_rows, prediction_headers)

    prediction_long_headers = [
        "sample_id",
        "device_id",
        "device_index",
        "patch_index",
        "patch_start",
        "forecast_start_index",
        "forecast_start_time",
        "target_time_index",
        "target_time",
        "horizon_step",
        "true_value",
        "pred_value",
        "abs_error",
    ]
    predictions_long_path = (
        out_dir / f"{run_tag}_predictions_{args.prediction_split}_long.csv"
    )
    write_csv(
        predictions_long_path,
        prediction_long_rows,
        prediction_long_headers,
    )

    horizon_rows = compute_horizon_rows(
        preds=export_result["preds"],
        targets=export_result["targets"],
        masks=export_result["masks"],
    )
    horizon_path = out_dir / f"{run_tag}_horizon_metrics_{args.prediction_split}.csv"
    write_csv(
        horizon_path,
        horizon_rows,
        ["horizon_step", "valid_points", "mae", "rmse", "bias"],
    )

    history_path = out_dir / f"{run_tag}_history.csv"
    write_csv(
        history_path,
        history_rows,
        [
            "epoch",
            "train_loss_mse",
            "train_loss_mae",
            "val_loss_mse",
            "val_loss_mae",
            "val_mae",
            "val_rmse",
            "val_bias",
        ],
    )

    summary = {
        "run_tag": run_tag,
        "dataset_name": dataset_name,
        "data_csv": str(data_csv),
        "task": args.task,
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_mae": float(best_val_mae),
        "num_devices_total": int(len(corpus)),
        "num_devices_train": int(len(split_map["train"])),
        "num_devices_val": int(len(split_map["val"])),
        "num_windows_train": int(len(train_ds)),
        "num_windows_val": int(len(val_ds)),
        "num_windows_export": int(len(export_ds)),
        "prediction_split": str(args.prediction_split),
        "export_metrics": export_result["metrics"],
        "artifacts": {
            "resolved_config": str(resolved_config_path),
            "split_assignment": split_artifacts["assignment_path"],
            "split_summary": split_artifacts["summary_path"],
            "best_checkpoint": str(best_ckpt_path),
            "last_checkpoint": str(last_ckpt_path),
            "history_csv": str(history_path),
            "predictions_csv": str(predictions_path),
            "predictions_long_csv": str(predictions_long_path),
            "horizon_metrics_csv": str(horizon_path),
        },
    }

    summary_path = out_dir / f"{run_tag}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    export_metrics = export_result["metrics"]
    print(f"[DONE] summary: {summary_path}")
    print(
        "[RESULT] "
        f"split={args.prediction_split} "
        f"mae={export_metrics['mae']:.4f} "
        f"rmse={export_metrics['rmse']:.4f} "
        f"bias={export_metrics['bias']:.4f} "
        f"valid_points={export_metrics['valid_points']}"
    )


if __name__ == "__main__":
    main()
