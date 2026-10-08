import argparse
import contextlib
import hashlib
import json
import math
import queue
import re
import sys
import time
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from LabelPredictionFinetune import lab_cv_runner as runner


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=1024,
        encoding_type="absolute",
        mask_prob=0.15,
        n_heads=16,
        n_layers=16,
        max_seq_length=25000,
        dropout=0.1,
        dim_feedforward=2048,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.max_seq_length = max_seq_length
        self.mask_prob = mask_prob
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.encoding_type = encoding_type

        self.embedding = nn.Embedding(vocab_size, n_embd)

        if encoding_type == "absolute":
            # Keep this as a plain tensor attribute for checkpoint compatibility
            # with the original training script.
            self.pos_embedding = self.create_pos_embedding(max_seq_length, n_embd)
        elif encoding_type == "time":
            self.day_emb = nn.Embedding(10, n_embd)
            self.hour_emb = nn.Embedding(30, n_embd)
            self.min_emb = nn.Embedding(65, n_embd)
            self.month_emb = nn.Embedding(15, n_embd)
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.linear = nn.Linear(n_embd, vocab_size)

    def create_pos_embedding(self, max_seq_length, n_embd):
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2) * -(math.log(10000.0) / n_embd)
        )
        pos_embedding = torch.zeros(max_seq_length, n_embd)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        return pos_embedding

    def forward(self, tokens, time_feats=None, mask=None, ret_embds=False):
        token_embeddings = self.embedding(tokens)

        if self.encoding_type == "absolute":
            position_embeddings = (
                self.pos_embedding[: tokens.size(1), :].unsqueeze(0).to(tokens.device)
            )
            embeddings = token_embeddings + position_embeddings
        else:
            if time_feats is None:
                raise ValueError("time_feats is required for time encoding.")
            day = time_feats[:, :, 0]
            hour = time_feats[:, :, 1]
            minute = time_feats[:, :, 2]
            month = time_feats[:, :, 3]
            time_embeddings = (
                self.day_emb(day)
                + self.hour_emb(hour)
                + self.min_emb(minute)
                + self.month_emb(month)
            )
            embeddings = token_embeddings + time_embeddings

        seq_length = tokens.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_length, seq_length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )

        transformer_output = self.transformer(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=mask,
        )
        logits = self.linear(transformer_output)

        if ret_embds:
            return logits, transformer_output
        return logits


CONFIG_PATH_KEYS = {
    "ckpt_path",
    "split_root",
    "dataset_split_root",
    "out_root",
    "embedding_cache_root",
}

CONFIG_SECTIONS = {
    "pretrain": {
        "ckpt_path",
        "model_name",
        "encoding_type",
        "n_heads",
        "max_seq_length",
    },
    "split": {
        "n_splits",
        "seed",
        "min_samples",
        "folds",
        "split_root",
        "dataset_split_root",
    },
    "runtime": {
        "device",
        "devices",
        "batch_size",
        "num_workers",
        "out_root",
        "prepare_only",
        "prepare_all_targets",
        "save_fold_predictions",
        "force_recompute_embeddings",
        "save_embeddings",
        "wait_for_idle_gpus",
        "idle_min_memory_gb",
        "idle_max_utilization",
        "idle_poll_seconds",
        "skip_run_aggregates",
    },
    "training": {
        "device",
        "devices",
        "batch_size",
        "num_workers",
        "out_root",
        "prepare_only",
        "prepare_all_targets",
        "save_fold_predictions",
        "force_recompute_embeddings",
        "save_embeddings",
        "wait_for_idle_gpus",
        "idle_min_memory_gb",
        "idle_max_utilization",
        "idle_poll_seconds",
        "skip_run_aggregates",
    },
    "embedding": {
        "pooling",
        "freq_minutes",
        "glucose_unit",
        "glucose_min_mgdl",
        "glucose_max_mgdl",
        "embedding_cache_root",
    },
    "ridge": {
        "alpha_values",
    },
}

DEFAULT_ALPHA_VALUES = [
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
]

LAB_TARGET_SPLIT_ROOT = REPO_ROOT / "data_lab" / "lab_target_cv"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract GluFormer representations for data_lab datasets and run "
            "ridge-based downstream prediction on existing CV splits."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=sorted(runner.DATASET_SPECS.keys()),
    )
    parser.add_argument(
        "--dataset_variant",
        type=str,
        default=None,
        help="Optional subset tag used to isolate filtered splits and outputs.",
    )
    parser.add_argument(
        "--sample_filters",
        type=str,
        default=None,
        help=(
            "Optional JSON filter spec applied to labels before split/training. "
            "Example: "
            "'[{\"column\":\"type\",\"exclude_values\":[\"T1D\"]}]'"
        ),
    )
    parser.add_argument("--targets", type=str, nargs="*", default=None)
    parser.add_argument("--list_targets", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--prepare_all_targets", action="store_true")
    parser.add_argument("--ckpt_path", type=str, required=False, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of CUDA devices for parallel task scheduling.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--split_root",
        type=str,
        default=str(REPO_ROOT / "data_lab" / "lab_target_cv"),
    )
    parser.add_argument(
        "--dataset_split_root",
        type=str,
        default=str(REPO_ROOT / "data_lab" / "lab_cv"),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "gluformer_lab_ridge"),
    )
    parser.add_argument("--embedding_cache_root", type=str, default=None)
    parser.add_argument(
        "--encoding_type",
        type=str,
        default="auto",
        choices=["auto", "absolute", "time"],
    )
    parser.add_argument("--n_heads", type=int, default=16)
    parser.add_argument("--max_seq_length", type=int, default=25000)
    parser.add_argument(
        "--pooling",
        type=str,
        default="last",
        choices=["last", "mean"],
    )
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument(
        "--glucose_unit",
        type=str,
        default="auto",
        choices=["auto", "mmol_l", "mg_dl"],
    )
    parser.add_argument("--glucose_min_mgdl", type=int, default=40)
    parser.add_argument("--glucose_max_mgdl", type=int, default=499)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_samples", type=int, default=30)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--alpha_values",
        type=float,
        nargs="*",
        default=None,
    )
    parser.add_argument("--save_embeddings", action="store_true")
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    parser.add_argument("--save_fold_predictions", action="store_true")
    parser.add_argument("--wait_for_idle_gpus", action="store_true")
    parser.add_argument("--idle_min_memory_gb", type=float, default=8.0)
    parser.add_argument("--idle_max_utilization", type=float, default=30.0)
    parser.add_argument("--idle_poll_seconds", type=float, default=60.0)
    parser.add_argument("--skip_run_aggregates", action="store_true")
    return parser


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_absolute_path(path_str: str, base_dir: Path) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((base_dir / path).resolve())


def flatten_config(raw_config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    for key, value in raw_config.items():
        if key in CONFIG_SECTIONS and isinstance(value, dict):
            continue
        flattened[key] = value

    for section_name, allowed_keys in CONFIG_SECTIONS.items():
        section = raw_config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key in allowed_keys:
                flattened[key] = value

    for key in CONFIG_PATH_KEYS:
        value = flattened.get(key)
        if isinstance(value, str) and value:
            flattened[key] = resolve_absolute_path(value, config_dir)

    return flattened


def collect_cli_overrides(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option_string in action.option_strings:
            option_to_dest[option_string] = action.dest

    overridden = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            overridden.add(dest)
    return overridden


def parse_args(argv: list[str] | None = None):
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        config_path = Path(args.config).expanduser().resolve()
        raw_config = load_json(config_path)
        config_values = flatten_config(raw_config, config_path.parent)
        cli_overrides = collect_cli_overrides(parser, argv)

        for key, value in config_values.items():
            if not hasattr(args, key):
                continue
            if key in cli_overrides:
                continue
            setattr(args, key, value)
        args.config = str(config_path)

    if args.alpha_values is None or len(args.alpha_values) == 0:
        args.alpha_values = DEFAULT_ALPHA_VALUES.copy()
    args.sample_filters = runner.normalize_sample_filters(
        getattr(args, "sample_filters", None)
    )
    args.dataset_variant = runner.normalize_dataset_variant(
        getattr(args, "dataset_variant", None)
    )

    if args.list_targets:
        return args

    if not args.dataset:
        raise ValueError("dataset is required. Pass --dataset or provide it in --config.")
    if not args.ckpt_path:
        raise ValueError("ckpt_path is required. Pass --ckpt_path or provide it in --config.")
    return args


def infer_model_name(args) -> str:
    if args.model_name:
        return str(args.model_name)
    return Path(args.ckpt_path).expanduser().stem


def compute_sample_hash(sample_ids: list[str]) -> str:
    joined = "\n".join(str(sample_id) for sample_id in sample_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def resolve_target_split_root() -> Path:
    split_root = LAB_TARGET_SPLIT_ROOT.resolve()
    if not split_root.exists():
        raise FileNotFoundError(
            "Target split root was not found. Expected prepared splits under: "
            f"{split_root}"
        )
    return split_root


def resolve_device(device_str: str) -> torch.device:
    requested = str(device_str).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] CUDA is not available, falling back to CPU instead of {device_str}.")
        return torch.device("cpu")
    return torch.device(device_str)


def strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return state_dict
    if not all(str(key).startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {str(key)[7:]: value for key, value in state_dict.items()}


def extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "network"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(torch.is_tensor(value) for value in payload.values()):
            return payload
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported checkpoint payload for GluFormer.")


def infer_layer_count(state_dict: dict[str, Any]) -> int:
    layer_indices = []
    pattern = re.compile(r"^transformer\.layers\.(\d+)\.")
    for key in state_dict.keys():
        match = pattern.match(str(key))
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise ValueError("Unable to infer GluFormer layer count from checkpoint.")
    return max(layer_indices) + 1


def build_gluformer_model(
    ckpt_path: Path,
    requested_encoding_type: str,
    n_heads: int,
    max_seq_length: int,
):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = strip_module_prefix(extract_state_dict(payload))

    if "embedding.weight" not in state_dict:
        raise KeyError("Checkpoint is missing embedding.weight.")

    inferred_encoding = (
        "time" if "day_emb.weight" in state_dict else "absolute"
    )
    encoding_type = (
        inferred_encoding
        if requested_encoding_type == "auto"
        else requested_encoding_type
    )

    vocab_size, n_embd = state_dict["embedding.weight"].shape
    n_layers = infer_layer_count(state_dict)
    dim_feedforward = int(state_dict["transformer.layers.0.linear1.weight"].shape[0])

    model = TransformerModel(
        vocab_size=int(vocab_size),
        n_embd=int(n_embd),
        encoding_type=encoding_type,
        n_heads=int(n_heads),
        n_layers=int(n_layers),
        max_seq_length=int(max_seq_length),
        dropout=0.1,
        dim_feedforward=dim_feedforward,
    )

    load_result = model.load_state_dict(state_dict, strict=True)
    meta = {
        "checkpoint": str(ckpt_path),
        "encoding_type": encoding_type,
        "inferred_encoding_type": inferred_encoding,
        "vocab_size": int(vocab_size),
        "n_embd": int(n_embd),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "dim_feedforward": int(dim_feedforward),
        "max_seq_length": int(max_seq_length),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    return model, meta


def resolve_glucose_scale(values: np.ndarray, glucose_unit: str) -> tuple[str, float]:
    if glucose_unit == "mmol_l":
        return "mmol_l", 18.0
    if glucose_unit == "mg_dl":
        return "mg_dl", 1.0

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return "mmol_l", 18.0

    upper_quantile = float(np.nanpercentile(finite_values, 99.0))
    if upper_quantile < 40.0:
        return "mmol_l", 18.0
    return "mg_dl", 1.0


def parse_timestamp_value(raw_value: Any) -> pd.Timestamp | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return None

    text = str(raw_value).strip()
    if not text or text.lower() == "nan":
        return None

    if text.isdigit() and len(text) == 8:
        timestamp = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        timestamp = pd.to_datetime(text, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def resolve_start_timestamp(row: pd.Series) -> pd.Timestamp:
    for key in ("starttime", "startdate", "startday"):
        if key not in row.index:
            continue
        timestamp = parse_timestamp_value(row[key])
        if timestamp is not None:
            return timestamp
    return pd.Timestamp("2000-01-01 00:00:00")


def prepare_cgm_bundle(cgm_df: pd.DataFrame, glucose_unit: str):
    numeric_cols = runner.resolve_numeric_cgm_columns(cgm_df)
    numeric_frame = cgm_df.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce")
    glucose_values = numeric_frame.to_numpy(dtype=np.float32)
    sample_ids = cgm_df.index.astype(str).tolist()
    sample_hash = compute_sample_hash(sample_ids)

    meta_cols = [column for column in cgm_df.columns if column not in numeric_cols]
    if meta_cols:
        meta_frame = cgm_df.loc[:, meta_cols]
        start_timestamps = [
            resolve_start_timestamp(meta_frame.iloc[index]).isoformat()
            for index in range(len(meta_frame))
        ]
    else:
        start_timestamps = ["2000-01-01T00:00:00"] * len(sample_ids)

    resolved_unit, glucose_scale = resolve_glucose_scale(glucose_values, glucose_unit)
    return {
        "sample_ids": sample_ids,
        "sample_hash": sample_hash,
        "glucose_values": glucose_values,
        "start_timestamps": start_timestamps,
        "seq_len": int(glucose_values.shape[1]),
        "resolved_glucose_unit": resolved_unit,
        "glucose_scale": float(glucose_scale),
        "numeric_cols": numeric_cols,
    }


def tokenize_glucose(
    values: np.ndarray,
    glucose_scale: float,
    glucose_min_mgdl: int,
    glucose_max_mgdl: int,
) -> np.ndarray:
    tokens = np.zeros(values.shape[0], dtype=np.int64)
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return tokens

    glucose_mgdl = np.rint(values[valid_mask] * glucose_scale).astype(np.int64)
    glucose_mgdl = np.clip(glucose_mgdl, glucose_min_mgdl, glucose_max_mgdl)
    tokens[valid_mask] = glucose_mgdl - glucose_min_mgdl + 1
    return tokens


@lru_cache(maxsize=4096)
def build_time_feature_matrix(
    start_timestamp_iso: str,
    seq_len: int,
    freq_minutes: int,
) -> np.ndarray:
    start_timestamp = pd.Timestamp(start_timestamp_iso)
    date_index = pd.date_range(
        start=start_timestamp,
        periods=int(seq_len),
        freq=f"{int(freq_minutes)}min",
    )
    return np.stack(
        [
            date_index.dayofweek.to_numpy(dtype=np.int64),
            date_index.hour.to_numpy(dtype=np.int64),
            date_index.minute.to_numpy(dtype=np.int64),
            date_index.month.to_numpy(dtype=np.int64),
        ],
        axis=1,
    )


class GluFormerFeatureDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        glucose_values: np.ndarray,
        start_timestamps: list[str],
        encoding_type: str,
        glucose_scale: float,
        glucose_min_mgdl: int,
        glucose_max_mgdl: int,
        freq_minutes: int,
    ):
        self.sample_ids = sample_ids
        self.glucose_values = glucose_values
        self.start_timestamps = start_timestamps
        self.encoding_type = encoding_type
        self.glucose_scale = float(glucose_scale)
        self.glucose_min_mgdl = int(glucose_min_mgdl)
        self.glucose_max_mgdl = int(glucose_max_mgdl)
        self.freq_minutes = int(freq_minutes)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        values = self.glucose_values[idx]
        tokens = tokenize_glucose(
            values=values,
            glucose_scale=self.glucose_scale,
            glucose_min_mgdl=self.glucose_min_mgdl,
            glucose_max_mgdl=self.glucose_max_mgdl,
        )

        if self.encoding_type == "time":
            time_feats = build_time_feature_matrix(
                self.start_timestamps[idx],
                seq_len=tokens.shape[0],
                freq_minutes=self.freq_minutes,
            )
        else:
            time_feats = np.zeros((tokens.shape[0], 4), dtype=np.int64)

        return {
            "sample_id": self.sample_ids[idx],
            "tokens": torch.from_numpy(tokens),
            "time_feats": torch.from_numpy(time_feats),
            "valid_len": int(np.count_nonzero(tokens)),
        }


def forward_hidden_states(
    model: TransformerModel,
    tokens: torch.Tensor,
    time_feats: torch.Tensor | None,
    pad_mask: torch.Tensor,
) -> torch.Tensor:
    token_embeddings = model.embedding(tokens)

    if model.encoding_type == "absolute":
        position_embeddings = (
            model.pos_embedding[: tokens.size(1), :].unsqueeze(0).to(tokens.device)
        )
        embeddings = token_embeddings + position_embeddings
    else:
        if time_feats is None:
            raise ValueError("time_feats is required for GluFormer time encoding.")
        day = time_feats[:, :, 0]
        hour = time_feats[:, :, 1]
        minute = time_feats[:, :, 2]
        month = time_feats[:, :, 3]
        time_embeddings = (
            model.day_emb(day)
            + model.hour_emb(hour)
            + model.min_emb(minute)
            + model.month_emb(month)
        )
        embeddings = token_embeddings + time_embeddings

    seq_length = tokens.size(1)
    causal_mask = torch.triu(
        torch.ones(seq_length, seq_length, dtype=torch.bool, device=tokens.device),
        diagonal=1,
    )
    return model.transformer(
        embeddings,
        mask=causal_mask,
        src_key_padding_mask=pad_mask,
    )


def pool_hidden_states(
    hidden_states: torch.Tensor,
    pad_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    valid_mask = ~pad_mask

    if pooling == "mean":
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
        masked_hidden = hidden_states * valid_mask.unsqueeze(-1)
        return masked_hidden.sum(dim=1) / denom

    valid_lengths = valid_mask.sum(dim=1)
    last_indices = (valid_lengths - 1).clamp(min=0)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    pooled = hidden_states[batch_indices, last_indices]

    empty_rows = valid_lengths == 0
    if torch.any(empty_rows):
        pooled = pooled.clone()
        pooled[empty_rows] = 0
    return pooled


def build_embedding_cache_paths(
    args,
    model_name: str,
    dataset_name: str,
    encoding_type: str,
    pooling: str,
) -> tuple[Path, Path]:
    if args.embedding_cache_root:
        cache_root = Path(args.embedding_cache_root).expanduser().resolve()
    else:
        cache_root = (
            Path(args.out_root).expanduser().resolve()
            / model_name
            / dataset_name
            / "_embedding_cache"
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    base_name = f"{dataset_name}_{encoding_type}_{pooling}"
    return cache_root / f"{base_name}.npz", cache_root / f"{base_name}.json"


def extract_embeddings(
    model: TransformerModel,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pooling: str,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    all_sample_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_valid_lengths: list[int] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            sample_ids = [str(sample_id) for sample_id in batch["sample_id"]]
            tokens = batch["tokens"].to(device, non_blocking=True)
            time_feats = batch["time_feats"].to(device, non_blocking=True)
            pad_mask = tokens.eq(0)

            autocast_ctx = (
                autocast(enabled=True)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                hidden_states = forward_hidden_states(
                    model=model,
                    tokens=tokens,
                    time_feats=time_feats if model.encoding_type == "time" else None,
                    pad_mask=pad_mask,
                )
                pooled = pool_hidden_states(
                    hidden_states=hidden_states,
                    pad_mask=pad_mask,
                    pooling=pooling,
                )

            all_sample_ids.extend(sample_ids)
            all_valid_lengths.extend(
                int(value) for value in batch["valid_len"].detach().cpu().tolist()
            )
            all_embeddings.append(pooled.float().cpu().numpy())

    return (
        all_sample_ids,
        np.concatenate(all_embeddings, axis=0),
        np.asarray(all_valid_lengths, dtype=np.int32),
    )


def load_cached_embeddings(cache_path: Path):
    with np.load(cache_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        embeddings = payload["embeddings"].astype(np.float32)
        valid_lengths = payload["valid_lengths"].astype(np.int32)
    return sample_ids, embeddings, valid_lengths


def extract_or_load_embeddings(
    args,
    model_name: str,
    dataset_name: str,
    model: TransformerModel,
    model_meta: dict[str, Any],
    cgm_bundle: dict[str, Any],
    device: torch.device,
):
    cache_path, meta_path = build_embedding_cache_paths(
        args=args,
        model_name=model_name,
        dataset_name=dataset_name,
        encoding_type=model_meta["encoding_type"],
        pooling=args.pooling,
    )

    expected_hash = cgm_bundle["sample_hash"]
    if (
        cache_path.exists()
        and meta_path.exists()
        and not args.force_recompute_embeddings
    ):
        cache_meta = load_json(meta_path)
        if (
            cache_meta.get("sample_hash") == expected_hash
            and cache_meta.get("pooling") == args.pooling
            and cache_meta.get("encoding_type") == model_meta["encoding_type"]
            and cache_meta.get("model_name") == model_name
        ):
            sample_ids, embeddings, valid_lengths = load_cached_embeddings(cache_path)
            print(f"[EMBED_CACHE_HIT] {dataset_name} -> {cache_path}")
            return {
                "sample_ids": sample_ids,
                "embeddings": embeddings,
                "valid_lengths": valid_lengths,
                "cache_path": str(cache_path),
                "cache_meta_path": str(meta_path),
                "cache_status": "hit",
            }

    feature_dataset = GluFormerFeatureDataset(
        sample_ids=cgm_bundle["sample_ids"],
        glucose_values=cgm_bundle["glucose_values"],
        start_timestamps=cgm_bundle["start_timestamps"],
        encoding_type=model_meta["encoding_type"],
        glucose_scale=cgm_bundle["glucose_scale"],
        glucose_min_mgdl=args.glucose_min_mgdl,
        glucose_max_mgdl=args.glucose_max_mgdl,
        freq_minutes=args.freq_minutes,
    )

    sample_ids, embeddings, valid_lengths = extract_embeddings(
        model=model,
        dataset=feature_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pooling=args.pooling,
    )

    np.savez_compressed(
        cache_path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
        valid_lengths=valid_lengths.astype(np.int32),
    )
    runner.save_json(
        meta_path,
        {
            "model_name": model_name,
            "checkpoint": args.ckpt_path,
            "dataset": dataset_name,
            "encoding_type": model_meta["encoding_type"],
            "pooling": args.pooling,
            "num_samples": int(len(sample_ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "sample_hash": expected_hash,
            "resolved_glucose_unit": cgm_bundle["resolved_glucose_unit"],
            "glucose_scale": cgm_bundle["glucose_scale"],
            "glucose_min_mgdl": int(args.glucose_min_mgdl),
            "glucose_max_mgdl": int(args.glucose_max_mgdl),
            "freq_minutes": int(args.freq_minutes),
            "model_meta": model_meta,
        },
    )
    cache_status = "saved"

    print(
        f"[EMBED_EXTRACT_DONE] {dataset_name} samples={len(sample_ids)} "
        f"dim={embeddings.shape[1]} pooling={args.pooling}"
    )
    return {
        "sample_ids": sample_ids,
        "embeddings": embeddings,
        "valid_lengths": valid_lengths,
        "cache_path": str(cache_path),
        "cache_meta_path": str(meta_path),
        "cache_status": cache_status,
    }


def load_task_split_ids(
    split_manifest_path: Path,
    selected_folds: set[int] | None,
) -> list[dict[str, Any]]:
    split_payload = load_json(split_manifest_path)
    fold_items = []
    for fold_info in split_payload["folds"]:
        fold_idx = int(fold_info["fold"])
        if selected_folds is not None and fold_idx not in selected_folds:
            continue
        train_ids = runner.load_split_ids(fold_info["train_csv"])
        val_ids = runner.load_split_ids(fold_info["val_csv"])
        fold_items.append(
            {
                "fold": fold_idx,
                "train_ids": train_ids,
                "val_ids": val_ids,
                "train_size": len(train_ids),
                "val_size": len(val_ids),
            }
        )
    return fold_items


def resolve_existing_split_bundle(
    dataset_name: str,
    task_name: str,
    task_df: pd.DataFrame,
    split_root: Path,
    n_splits: int,
    seed: int,
    selected_folds: set[int] | None,
):
    task_manifest_path = (
        split_root
        / dataset_name
        / task_name
        / f"kfold_{n_splits}_seed_{seed}"
        / "split_manifest.json"
    )
    if task_manifest_path.exists():
        fold_items = load_task_split_ids(task_manifest_path, selected_folds)
        return {
            "source": "target_split",
            "manifest_path": str(task_manifest_path),
            "folds": fold_items,
        }

    raise FileNotFoundError(
        "No prepared target split was found. Expected: "
        f"{task_manifest_path}"
    )


def build_classification_probabilities(decision_scores: np.ndarray) -> np.ndarray:
    decision_scores = np.asarray(decision_scores)
    if decision_scores.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-decision_scores))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)

    shifted = decision_scores - np.max(decision_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def save_fold_artifacts(
    fold_dir: Path,
    mode: str,
    fold_metrics: dict[str, float],
    sample_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
):
    fold_dir.mkdir(parents=True, exist_ok=True)
    prediction_table = runner.build_prediction_table(
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
        task_type=mode,
        y_probs=y_probs,
        y_true_raw=y_true if mode == "reg" else None,
        y_pred_raw=y_pred if mode == "reg" else None,
    )
    prediction_table.to_csv(fold_dir / "predictions.csv", index=False, encoding="utf-8")
    runner.save_json(
        fold_dir / "metrics.json",
        {
            "metrics": fold_metrics,
            "num_samples": int(len(sample_ids)),
        },
    )


def fit_ridge_regressor(alpha_values: list[float]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.asarray(alpha_values, dtype=np.float64))),
        ]
    )


def fit_ridge_classifier(alpha_values: list[float]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifierCV(alphas=np.asarray(alpha_values, dtype=np.float64))),
        ]
    )


def run_task_bundle(
    args,
    run_root: Path,
    embedding_bundle: dict[str, Any],
    task_name: str,
    target: str,
    mode: str,
    task_df: pd.DataFrame,
    label_map: dict[str, int] | None,
    split_bundle: dict[str, Any],
):
    sample_ids = embedding_bundle["sample_ids"]
    embeddings = embedding_bundle["embeddings"]
    id_to_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}

    labels = task_df[target].copy()
    labels.index = labels.index.astype(str)

    task_dir = run_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    runner.save_json(
        task_dir / "split_source.json",
        {
            "source": split_bundle["source"],
            "manifest_path": split_bundle["manifest_path"],
        },
    )

    all_metrics: list[dict[str, float]] = []
    fold_metrics_rows: list[dict[str, Any]] = []
    num_classes = runner.resolve_num_classes(label_map)

    for fold_info in split_bundle["folds"]:
        fold_idx = int(fold_info["fold"])
        fold_dir = task_dir / f"fold_{fold_idx}"

        train_ids = [
            sample_id
            for sample_id in fold_info["train_ids"]
            if sample_id in labels.index and sample_id in id_to_index
        ]
        val_ids = [
            sample_id
            for sample_id in fold_info["val_ids"]
            if sample_id in labels.index and sample_id in id_to_index
        ]

        if len(train_ids) < 2 or len(val_ids) == 0:
            print(
                f"[SKIP_FOLD] {task_name} fold={fold_idx} "
                f"train={len(train_ids)} val={len(val_ids)}"
            )
            continue

        x_train = embeddings[[id_to_index[sample_id] for sample_id in train_ids]]
        x_val = embeddings[[id_to_index[sample_id] for sample_id in val_ids]]

        if mode == "reg":
            y_train = labels.loc[train_ids].to_numpy(dtype=np.float64)
            y_val = labels.loc[val_ids].to_numpy(dtype=np.float64)
            estimator = fit_ridge_regressor(args.alpha_values)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(x_val).astype(np.float64)
            y_probs = None
        else:
            y_train = labels.loc[train_ids].to_numpy(dtype=np.int64)
            y_val = labels.loc[val_ids].to_numpy(dtype=np.int64)
            if len(np.unique(y_train)) < 2:
                print(
                    f"[SKIP_FOLD] {task_name} fold={fold_idx} "
                    "classification train split only has one class."
                )
                continue
            estimator = fit_ridge_classifier(args.alpha_values)
            estimator.fit(x_train, y_train)
            y_pred = estimator.predict(x_val).astype(np.int64)
            decision_scores = estimator.decision_function(x_val)
            y_probs = build_classification_probabilities(decision_scores)

        fitted_ridge = estimator.named_steps["ridge"]
        alpha_value = float(fitted_ridge.alpha_)
        fold_metrics = runner.compute_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_probs=y_probs,
            task_type=mode,
            num_classes=num_classes,
        )
        fold_metrics["alpha"] = alpha_value

        if args.save_fold_predictions:
            save_fold_artifacts(
                fold_dir=fold_dir,
                mode=mode,
                fold_metrics=fold_metrics,
                sample_ids=val_ids,
                y_true=y_val,
                y_pred=y_pred,
                y_probs=y_probs,
            )
        else:
            fold_dir.mkdir(parents=True, exist_ok=True)
            runner.save_json(
                fold_dir / "metrics.json",
                {
                    "metrics": fold_metrics,
                    "num_samples": int(len(val_ids)),
                },
            )

        all_metrics.append(fold_metrics)
        fold_metrics_row = {
            "dataset": args.dataset,
            "task_name": task_name,
            "target": target,
            "task_type": mode,
            "fold": fold_idx,
            "device": str(args.device),
            "split_source": split_bundle["source"],
            "train_size": int(len(train_ids)),
            "val_size": int(len(val_ids)),
            "alpha": alpha_value,
        }
        for key, value in fold_metrics.items():
            if key == "alpha":
                continue
            fold_metrics_row[key] = float(value)
        fold_metrics_rows.append(fold_metrics_row)

        print(
            f"[RIDGE_FOLD] {task_name} fold={fold_idx} alpha={alpha_value:.6g} "
            + " ".join(
                f"{key}={value:.4f}"
                for key, value in fold_metrics.items()
                if key != "alpha" and not np.isnan(value)
            )
        )

    summary = runner.summarize_metrics(all_metrics)
    runner.save_json(
        task_dir / "cv_summary.json",
        {
            "dataset": args.dataset,
            "task_name": task_name,
            "target": target,
            "task_type": mode,
            "split_source": split_bundle["source"],
            "num_folds": int(len(all_metrics)),
            "metrics": summary,
        },
    )
    if fold_metrics_rows:
        pd.DataFrame(fold_metrics_rows).to_csv(
            task_dir / "fold_metrics.csv",
            index=False,
            encoding="utf-8",
        )

    task_summary_row = {
        "dataset": args.dataset,
        "task_name": task_name,
        "target": target,
        "task_type": mode,
        "device": str(args.device),
        "split_source": split_bundle["source"],
        "num_samples": int(len(labels)),
        "num_folds": int(len(all_metrics)),
    }
    for key, value in summary.items():
        task_summary_row[key] = float(value)

    return {
        "task_summary_row": task_summary_row,
        "fold_metrics_rows": fold_metrics_rows,
    }


def gpu_worker(
    task_queue,
    result_queue,
    device_str: str,
    run_root_str: str,
    embedding_cache_path_str: str,
    args_payload: dict[str, Any],
):
    try:
        runner.set_gpu_flags()
        args_device = torch.device(device_str)
        if args_device.type == "cuda":
            torch.cuda.set_device(args_device)

        worker_args = argparse.Namespace(**args_payload)
        worker_args.device = device_str

        sample_ids, embeddings, valid_lengths = load_cached_embeddings(
            Path(embedding_cache_path_str)
        )
        embedding_bundle = {
            "sample_ids": sample_ids,
            "embeddings": embeddings,
            "valid_lengths": valid_lengths,
        }

        result_queue.put(
            {
                "type": "worker_started",
                "device": device_str,
            }
        )

        while True:
            try:
                task_bundle = task_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break

            result = run_task_bundle(
                args=worker_args,
                run_root=Path(run_root_str),
                embedding_bundle=embedding_bundle,
                task_name=task_bundle["task_name"],
                target=task_bundle["target"],
                mode=task_bundle["mode"],
                task_df=task_bundle["task_df"],
                label_map=task_bundle["label_map"],
                split_bundle=task_bundle["split_bundle"],
            )
            result_queue.put(
                {
                    "type": "task_result",
                    "device": device_str,
                    **result,
                }
            )
    except Exception:
        result_queue.put(
            {
                "type": "worker_error",
                "device": device_str,
                "traceback": traceback.format_exc(),
            }
        )


def run_parallel_task_bundles(
    task_bundles: list[dict[str, Any]],
    devices: list[str],
    args,
    run_root: Path,
    embedding_cache_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for task_bundle in task_bundles:
        task_queue.put(task_bundle)

    pending_devices = list(devices)
    workers: dict[str, mp.Process] = {}
    all_task_summaries: list[dict[str, Any]] = []
    all_fold_metrics: list[dict[str, Any]] = []
    expected_task_results = len(task_bundles)
    received_task_results = 0
    worker_errors: list[str] = []
    args_payload = vars(args).copy()

    while received_task_results < expected_task_results or workers:
        for device_str in pending_devices.copy():
            if received_task_results >= expected_task_results:
                break
            if not args.wait_for_idle_gpus:
                idle = True
                status = runner.get_gpu_status(device_str)
            else:
                idle, status = runner.is_device_idle(
                    device=device_str,
                    min_free_memory_gb=args.idle_min_memory_gb,
                    max_utilization=args.idle_max_utilization,
                )
            if not idle:
                free_memory = status.get("free_memory_gb")
                utilization = status.get("utilization")
                print(
                    f"[WAIT_GPU] {device_str} free={free_memory}GB util={utilization}% "
                    f"(need free>={args.idle_min_memory_gb}GB util<={args.idle_max_utilization}%)"
                )
                continue

            worker = ctx.Process(
                target=gpu_worker,
                args=(
                    task_queue,
                    result_queue,
                    device_str,
                    str(run_root),
                    str(embedding_cache_path),
                    args_payload,
                ),
            )
            worker.start()
            workers[device_str] = worker
            pending_devices.remove(device_str)
            print(f"[LAUNCH_GPU] device={device_str} pid={worker.pid}")

        try:
            message = result_queue.get(timeout=max(float(args.idle_poll_seconds), 1.0))
        except queue.Empty:
            if pending_devices:
                continue
            message = None

        if message is not None:
            if message["type"] == "task_result":
                received_task_results += 1
                runner.collect_aggregate_rows(
                    message,
                    all_task_summaries,
                    all_fold_metrics,
                )
            elif message["type"] == "worker_error":
                worker_errors.append(message["traceback"])
                break

        finished_devices = []
        for device_str, worker in workers.items():
            if not worker.is_alive():
                worker.join()
                finished_devices.append(device_str)
        for device_str in finished_devices:
            workers.pop(device_str, None)

        if pending_devices and args.wait_for_idle_gpus:
            time.sleep(max(float(args.idle_poll_seconds), 1.0))

    for worker in workers.values():
        worker.join()

    if worker_errors:
        raise RuntimeError(worker_errors[0])
    return all_task_summaries, all_fold_metrics


def resolve_target_specs(args, labels_df: pd.DataFrame) -> list[dict[str, Any]]:
    if args.targets:
        return runner.parse_target_specs(args.targets, labels_df)
    if args.prepare_all_targets:
        return runner.parse_target_specs(labels_df.columns.tolist(), labels_df)
    if args.prepare_only:
        return []
    raise ValueError(
        "Prediction requires --targets, or pass --prepare_only to cache embeddings only."
    )


def select_embedding_device(args, devices: list[str]) -> str:
    if not devices:
        return str(args.device)
    if not args.wait_for_idle_gpus:
        return devices[0]

    while True:
        for device_str in devices:
            if runner.parse_cuda_device_index(device_str) is None:
                return device_str
            idle, status = runner.is_device_idle(
                device=device_str,
                min_free_memory_gb=args.idle_min_memory_gb,
                max_utilization=args.idle_max_utilization,
            )
            if idle:
                print(f"[EMBED_LAUNCH_GPU] device={device_str}")
                return device_str
            print(
                f"[EMBED_WAIT_GPU] {device_str} free={status.get('free_memory_gb')}GB "
                f"util={status.get('utilization')}% "
                f"(need free>={args.idle_min_memory_gb}GB util<={args.idle_max_utilization}%)"
            )
        time.sleep(max(float(args.idle_poll_seconds), 1.0))


def run_from_args(args):
    runner.set_gpu_flags()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_key = runner.resolve_dataset_key(
        args.dataset,
        dataset_variant=getattr(args, "dataset_variant", None),
        sample_filters=getattr(args, "sample_filters", None),
    )
    cgm_df, labels_df = runner.load_dataset_frames(
        args.dataset,
        sample_filters=getattr(args, "sample_filters", None),
    )

    if args.list_targets:
        available_targets = labels_df.columns.tolist()
        for column in labels_df.columns.tolist():
            print(column)
        return {
            "dataset": args.dataset,
            "available_targets": available_targets,
        }

    target_specs = resolve_target_specs(args, labels_df)

    model_name = infer_model_name(args)
    run_root = Path(args.out_root).expanduser().resolve() / model_name / dataset_key
    run_root.mkdir(parents=True, exist_ok=True)
    split_root = resolve_target_split_root()
    resolved_devices = runner.normalize_device_list(args.devices, args.device)
    embedding_device_str = select_embedding_device(args, resolved_devices)
    args.device = embedding_device_str
    device = resolve_device(embedding_device_str)

    model, model_meta = build_gluformer_model(
        ckpt_path=Path(args.ckpt_path).expanduser().resolve(),
        requested_encoding_type=args.encoding_type,
        n_heads=args.n_heads,
        max_seq_length=args.max_seq_length,
    )
    model = model.to(device)

    cgm_bundle = prepare_cgm_bundle(cgm_df, args.glucose_unit)
    embedding_bundle = extract_or_load_embeddings(
        args=args,
        model_name=model_name,
        dataset_name=dataset_key,
        model=model,
        model_meta=model_meta,
        cgm_bundle=cgm_bundle,
        device=device,
    )

    resolved_config = {
        "dataset": args.dataset,
        "dataset_key": dataset_key,
        "dataset_variant": getattr(args, "dataset_variant", None),
        "sample_filters": getattr(args, "sample_filters", None),
        "targets": target_specs,
        "checkpoint": args.ckpt_path,
        "model_name": model_name,
        "device": str(device),
        "devices": resolved_devices,
        "pooling": args.pooling,
        "alpha_values": list(float(value) for value in args.alpha_values),
        "split_root": str(split_root),
        "split_policy": "target_only",
        "runtime": {
            "wait_for_idle_gpus": bool(args.wait_for_idle_gpus),
            "idle_min_memory_gb": float(args.idle_min_memory_gb),
            "idle_max_utilization": float(args.idle_max_utilization),
            "idle_poll_seconds": float(args.idle_poll_seconds),
        },
        "embedding_cache": {
            "cache_path": embedding_bundle["cache_path"],
            "cache_meta_path": embedding_bundle["cache_meta_path"],
            "cache_status": embedding_bundle["cache_status"],
            "num_samples": int(len(embedding_bundle["sample_ids"])),
            "embedding_dim": int(embedding_bundle["embeddings"].shape[1]),
            "resolved_glucose_unit": cgm_bundle["resolved_glucose_unit"],
            "glucose_scale": cgm_bundle["glucose_scale"],
        },
        "model_meta": model_meta,
    }
    runner.save_json(run_root / "_resolved_config.json", resolved_config)

    if args.prepare_only:
        print(
            f"[PREPARE_DONE] dataset={args.dataset} "
            f"cache={embedding_bundle['cache_path']}"
        )
        return {
            "dataset": dataset_key,
            "model_name": model_name,
            "run_root": str(run_root),
            "num_tasks": 0,
            "num_fold_rows": 0,
            "task_summaries": [],
            "fold_metrics_rows": [],
            "resolved_config_path": str(run_root / "_resolved_config.json"),
            "embedding_cache_path": embedding_bundle["cache_path"],
            "embedding_cache_meta_path": embedding_bundle["cache_meta_path"],
        }

    all_task_summaries: list[dict[str, Any]] = []
    all_fold_metrics: list[dict[str, Any]] = []
    selected_folds = set(args.folds) if args.folds else None
    task_bundles: list[dict[str, Any]] = []

    for spec in target_specs:
        target = spec["target"]
        mode = spec["mode"]
        task_df, label_map = runner.build_task_frame(
            labels_df=labels_df,
            spec=spec,
            min_samples=args.min_samples,
        )
        if task_df is None:
            print(
                f"[SKIP] dataset={args.dataset} target={target}: "
                f"valid sample count < {args.min_samples} or only one class."
            )
            continue

        task_name = str(spec.get("task_name") or runner.build_task_name(target, mode))
        split_bundle = resolve_existing_split_bundle(
            dataset_name=dataset_key,
            task_name=task_name,
            task_df=task_df,
            split_root=split_root,
            n_splits=args.n_splits,
            seed=args.seed,
            selected_folds=selected_folds,
        )

        split_task_root = (
            split_root
            / dataset_key
            / task_name
            / f"kfold_{args.n_splits}_seed_{args.seed}"
        )
        runner.save_json(
            split_task_root / "target_info.json",
            {
                "dataset": args.dataset,
                "dataset_key": dataset_key,
                "dataset_variant": getattr(args, "dataset_variant", None),
                "sample_filters": getattr(args, "sample_filters", None),
                "target": target,
                "task_name": task_name,
                "task_type": mode,
                "num_samples": int(len(task_df)),
                "num_classes": runner.resolve_num_classes(label_map),
                "target_spec": spec,
            },
        )
        if label_map is not None:
            runner.save_json(
                split_task_root / "label_mapping.json",
                {
                    "target": target,
                    "mapping": {str(key): int(value) for key, value in label_map.items()},
                },
            )

        print(
            f"[TASK_START] dataset={dataset_key} task={task_name} "
            f"split_source={split_bundle['source']}"
        )
        task_bundles.append(
            {
                "dataset": dataset_key,
                "task_name": task_name,
                "target": target,
                "mode": mode,
                "task_df": task_df,
                "label_map": label_map,
                "split_bundle": split_bundle,
                "spec": spec,
            }
        )

    if len(resolved_devices) > 1:
        all_task_summaries, all_fold_metrics = run_parallel_task_bundles(
            task_bundles=task_bundles,
            devices=resolved_devices,
            args=args,
            run_root=run_root,
            embedding_cache_path=embedding_bundle["cache_path"],
        )
    else:
        device_str = resolved_devices[0]
        if (
            args.wait_for_idle_gpus
            and runner.parse_cuda_device_index(device_str) is not None
        ):
            while True:
                idle, status = runner.is_device_idle(
                    device=device_str,
                    min_free_memory_gb=args.idle_min_memory_gb,
                    max_utilization=args.idle_max_utilization,
                )
                if idle:
                    print(f"[LAUNCH_GPU] device={device_str}")
                    break
                print(
                    f"[WAIT_GPU] {device_str} free={status.get('free_memory_gb')}GB "
                    f"util={status.get('utilization')}% "
                    f"(need free>={args.idle_min_memory_gb}GB util<={args.idle_max_utilization}%)"
                )
                time.sleep(max(float(args.idle_poll_seconds), 1.0))

        for task_bundle in task_bundles:
            result = run_task_bundle(
                args=args,
                run_root=run_root,
                embedding_bundle=embedding_bundle,
                task_name=task_bundle["task_name"],
                target=task_bundle["target"],
                mode=task_bundle["mode"],
                task_df=task_bundle["task_df"],
                label_map=task_bundle["label_map"],
                split_bundle=task_bundle["split_bundle"],
            )
            runner.collect_aggregate_rows(result, all_task_summaries, all_fold_metrics)

    if not args.skip_run_aggregates:
        runner.write_run_aggregates(
            run_root=run_root,
            all_task_summaries=all_task_summaries,
            all_fold_metrics=all_fold_metrics,
        )
    print(
        f"[RUN_DONE] dataset={dataset_key} run_root={run_root} "
        f"tasks={len(all_task_summaries)} folds={len(all_fold_metrics)}"
    )
    return {
        "dataset": dataset_key,
        "model_name": model_name,
        "run_root": str(run_root),
        "num_tasks": len(all_task_summaries),
        "num_fold_rows": len(all_fold_metrics),
        "task_summaries": all_task_summaries,
        "fold_metrics_rows": all_fold_metrics,
        "resolved_config_path": str(run_root / "_resolved_config.json"),
        "embedding_cache_path": embedding_bundle["cache_path"],
        "embedding_cache_meta_path": embedding_bundle["cache_meta_path"],
    }


def main():
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
