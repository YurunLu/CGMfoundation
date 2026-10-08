import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
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
from models.downstream_base_model import CGMBaseModel

CONFIG_PATH_KEYS = {
    "ckpt_path",
    "split_root",
    "out_root",
    "embedding_cache_root",
}

CONFIG_SECTIONS = {
    "pretrain": {
        "ckpt_path",
        "model_name",
    },
    "split": {
        "prepare_only",
        "prepare_all_targets",
        "n_splits",
        "seed",
        "min_samples",
        "folds",
        "split_root",
        "force_regenerate_splits",
    },
    "runtime": {
        "device",
        "devices",
        "batch_size",
        "num_workers",
        "out_root",
        "embedding_cache_root",
        "pooling",
        "force_recompute_embeddings",
        "save_fold_predictions",
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
        "embedding_cache_root",
        "pooling",
        "force_recompute_embeddings",
        "save_fold_predictions",
        "wait_for_idle_gpus",
        "idle_min_memory_gb",
        "idle_max_utilization",
        "idle_poll_seconds",
        "skip_run_aggregates",
    },
    "model": {
        "patch_size",
        "stride",
        "forecast_size",
        "embed_dim",
        "depth",
        "num_heads",
        "hidden_dim",
        "dropout",
        "max_seq_len",
        "use_cnn",
        "use_transformer",
        "use_temporal",
        "use_positional",
    },
    "embedding": {
        "pooling",
        "embedding_cache_root",
    },
    "feature": {
        "required_label_columns",
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


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed CGMfoundation representations for data_lab datasets and "
            "run ridge-based downstream prediction on prepared CV splits."
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
            '\'[{"column":"type","exclude_values":["T1D"]}]\''
        ),
    )
    parser.add_argument("--targets", type=str, nargs="*", default=None)
    parser.add_argument("--list_targets", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--prepare_all_targets", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
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
        help="Optional list of CUDA devices. The first idle device is used for embedding extraction.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--split_root",
        type=str,
        default=str(REPO_ROOT / "splits" / "lab_target_cv"),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "cgmfoundation_lab_ridge"),
    )
    parser.add_argument("--embedding_cache_root", type=str, default=None)
    parser.add_argument(
        "--required_label_columns",
        type=str,
        nargs="*",
        default=None,
        help="Optional label columns that must be non-null for a sample to remain in the task cohort.",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["last", "mean"],
    )
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--forecast_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--use_cnn", type=str, default=None)
    parser.add_argument("--use_transformer", type=str, default=None)
    parser.add_argument("--use_temporal", type=str, default=None)
    parser.add_argument("--use_positional", type=str, default=None)
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
    parser.add_argument("--force_regenerate_splits", action="store_true")
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
    if args.required_label_columns is None:
        args.required_label_columns = []
    args.sample_filters = runner.normalize_sample_filters(
        getattr(args, "sample_filters", None)
    )
    args.dataset_variant = runner.normalize_dataset_variant(
        getattr(args, "dataset_variant", None)
    )

    if args.list_targets:
        return args

    if not args.dataset:
        raise ValueError(
            "dataset is required. Pass --dataset or provide it in --config."
        )
    if not args.ckpt_path and not args.prepare_only:
        raise ValueError(
            "ckpt_path is required for embedding extraction/training. "
            "Pass --ckpt_path or provide it in --config."
        )
    return args


def infer_model_name(args) -> str:
    if args.model_name:
        return str(args.model_name)
    if args.ckpt_path:
        return Path(args.ckpt_path).expanduser().stem
    return "prepare_only"


def resolve_device(device_str: str) -> torch.device:
    requested = str(device_str).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"[WARN] CUDA is not available, falling back to CPU instead of {device_str}."
        )
        return torch.device("cpu")
    return torch.device(device_str)


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


class CGMFoundationFeatureDataset(Dataset):
    def __init__(
        self,
        pre_bundle: dict[str, Any],
        patch_size: int,
        stride: int,
    ):
        self.sample_ids = [str(sample_id) for sample_id in pre_bundle["sample_ids"]]
        self.data = pre_bundle["data"]
        self.valid_lengths = [int(value) for value in pre_bundle["valid_lengths"]]
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.seq_len = int(pre_bundle["seq_len"])

        max_start = self.seq_len - self.patch_size
        if max_start < 0:
            raise ValueError(
                f"patch_size={self.patch_size} is larger than seq_len={self.seq_len}."
            )
        self.starts = list(range(0, max_start + 1, self.stride))
        self.patch_end_positions = torch.tensor(
            [start + self.patch_size for start in self.starts],
            dtype=torch.int32,
        )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        seq = self.data[idx]
        patches = seq.unfold(0, self.patch_size, self.stride).contiguous()
        valid_length = int(self.valid_lengths[idx])
        valid_mask = self.patch_end_positions <= valid_length
        return {
            "sample_id": self.sample_ids[idx],
            "patches": patches,
            "forecast_mask": valid_mask,
            "valid_patch_count": int(valid_mask.sum().item()),
        }


def feature_collate_fn(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "patches": torch.stack([item["patches"] for item in batch], dim=0),
        "forecast_mask": torch.stack([item["forecast_mask"] for item in batch], dim=0),
        "valid_patch_count": torch.tensor(
            [item["valid_patch_count"] for item in batch],
            dtype=torch.int32,
        ),
    }


def pool_hidden_states(
    hidden_states: torch.Tensor,
    valid_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
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

    base_name = f"{dataset_name}_{pooling}"
    return cache_root / f"{base_name}.npz", cache_root / f"{base_name}.json"


def load_cached_embeddings(cache_path: Path):
    with np.load(cache_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        embeddings = payload["embeddings"].astype(np.float32)
        valid_patch_counts = payload["valid_patch_counts"].astype(np.int32)
    return sample_ids, embeddings, valid_patch_counts


def extract_embeddings(
    model: CGMBaseModel,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pooling: str,
    model_stride: int,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=feature_collate_fn,
    )

    all_sample_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_valid_patch_counts: list[int] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            sample_ids = [str(sample_id) for sample_id in batch["sample_ids"]]
            patches = batch["patches"].to(device, non_blocking=True)
            valid_mask = batch["forecast_mask"].to(device, non_blocking=True).bool()

            autocast_ctx = (
                autocast(enabled=True)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                hidden_states = model(
                    patches,
                    start_idx=0,
                    stride=model_stride,
                    return_embedding=True,
                )
                pooled = pool_hidden_states(
                    hidden_states=hidden_states,
                    valid_mask=valid_mask,
                    pooling=pooling,
                )

            all_sample_ids.extend(sample_ids)
            all_valid_patch_counts.extend(
                int(value) for value in batch["valid_patch_count"].cpu().tolist()
            )
            all_embeddings.append(pooled.float().cpu().numpy())

    return (
        all_sample_ids,
        np.concatenate(all_embeddings, axis=0),
        np.asarray(all_valid_patch_counts, dtype=np.int32),
    )


def extract_or_load_embeddings(
    args,
    model_name: str,
    dataset_name: str,
    model: CGMBaseModel,
    model_config: dict[str, Any],
    pre_bundle: dict[str, Any],
    device: torch.device,
):
    cache_path, meta_path = build_embedding_cache_paths(
        args=args,
        model_name=model_name,
        dataset_name=dataset_name,
        pooling=args.pooling,
    )

    expected_hash = runner.compute_sample_hash(pre_bundle["sample_ids"])
    expected_model_config = {
        key: model_config[key]
        for key in (
            "patch_size",
            "forecast_size",
            "embed_dim",
            "depth",
            "num_heads",
            "hidden_dim",
            "dropout",
            "max_seq_len",
            "use_cnn",
            "use_transformer",
            "use_temporal",
            "use_positional",
            "model_stride",
        )
    }

    if (
        cache_path.exists()
        and meta_path.exists()
        and not args.force_recompute_embeddings
    ):
        cache_meta = load_json(meta_path)
        if (
            cache_meta.get("sample_hash") == expected_hash
            and cache_meta.get("pooling") == args.pooling
            and cache_meta.get("checkpoint")
            == str(Path(args.ckpt_path).expanduser().resolve())
            and cache_meta.get("model_config") == expected_model_config
        ):
            sample_ids, embeddings, valid_patch_counts = load_cached_embeddings(
                cache_path
            )
            print(f"[EMBED_CACHE_HIT] {dataset_name} -> {cache_path}")
            return {
                "sample_ids": sample_ids,
                "embeddings": embeddings,
                "valid_patch_counts": valid_patch_counts,
                "cache_path": str(cache_path),
                "cache_meta_path": str(meta_path),
                "cache_status": "hit",
            }

    feature_dataset = CGMFoundationFeatureDataset(
        pre_bundle=pre_bundle,
        patch_size=model_config["patch_size"],
        stride=model_config["model_stride"],
    )
    sample_ids, embeddings, valid_patch_counts = extract_embeddings(
        model=model,
        dataset=feature_dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pooling=args.pooling,
        model_stride=model_config["model_stride"],
    )

    np.savez_compressed(
        cache_path,
        sample_ids=np.asarray(sample_ids, dtype=str),
        embeddings=embeddings.astype(np.float32),
        valid_patch_counts=valid_patch_counts.astype(np.int32),
    )
    runner.save_json(
        meta_path,
        {
            "model_name": model_name,
            "checkpoint": str(Path(args.ckpt_path).expanduser().resolve()),
            "dataset": dataset_name,
            "pooling": args.pooling,
            "num_samples": int(len(sample_ids)),
            "embedding_dim": int(embeddings.shape[1]),
            "sample_hash": expected_hash,
            "model_config": expected_model_config,
        },
    )
    print(
        f"[EMBED_EXTRACT_DONE] {dataset_name} samples={len(sample_ids)} "
        f"dim={embeddings.shape[1]} pooling={args.pooling}"
    )
    return {
        "sample_ids": sample_ids,
        "embeddings": embeddings,
        "valid_patch_counts": valid_patch_counts,
        "cache_path": str(cache_path),
        "cache_meta_path": str(meta_path),
        "cache_status": "saved",
    }


def build_classification_probabilities(decision_scores: np.ndarray) -> np.ndarray:
    decision_scores = np.asarray(decision_scores)
    if decision_scores.ndim == 1:
        probs_pos = 1.0 / (1.0 + np.exp(-decision_scores))
        return np.stack([1.0 - probs_pos, probs_pos], axis=1)

    shifted = decision_scores - np.max(decision_scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


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
            (
                "ridge",
                RidgeClassifierCV(alphas=np.asarray(alpha_values, dtype=np.float64)),
            ),
        ]
    )


def resolve_target_specs(args, labels_df: pd.DataFrame) -> list[dict[str, Any]]:
    if args.targets:
        return runner.parse_target_specs(args.targets, labels_df)
    if args.prepare_all_targets:
        return runner.parse_target_specs(labels_df.columns.tolist(), labels_df)
    if args.prepare_only:
        return []
    raise ValueError(
        "Prediction requires --targets, or pass --prepare_only to only create splits/cache embeddings."
    )


def apply_required_label_columns(
    task_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    required_label_columns: list[str] | None,
) -> pd.DataFrame:
    required = [
        str(column).strip()
        for column in (required_label_columns or [])
        if str(column).strip()
    ]
    if not required:
        return task_df

    filtered = task_df.copy()
    for column in required:
        if column not in labels_df.columns:
            raise KeyError(f"Required label column not found: {column}")
        before = len(filtered)
        values = labels_df.loc[filtered.index, column]
        filtered = filtered.loc[values.notna()].copy()
        after = len(filtered)
        print(f"[REQUIRE_LABEL] column={column} kept={after}/{before}")
    return filtered


def ensure_target_split_bundle(
    args,
    dataset_key: str,
    task_name: str,
    target: str,
    mode: str,
    task_df: pd.DataFrame,
    selected_folds: set[int] | None,
):
    split_root = Path(args.split_root).expanduser().resolve()
    payload = runner.ensure_target_split(
        dataset_name=dataset_key,
        task_name=task_name,
        target_col=target,
        task_type=mode,
        task_df=task_df,
        split_root=split_root,
        n_splits=args.n_splits,
        seed=args.seed,
        force_regenerate=args.force_regenerate_splits,
    )
    status = str(payload.get("_status") or "created").upper()
    print(
        f"[TARGET_SPLIT:{status}] {dataset_key}/{task_name} -> {len(task_df)} samples"
    )

    fold_items = []
    for fold_info in payload["folds"]:
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

    manifest_path = (
        split_root
        / dataset_key
        / task_name
        / f"kfold_{args.n_splits}_seed_{args.seed}"
        / "split_manifest.json"
    )
    return {
        "source": "target_split",
        "manifest_path": str(manifest_path),
        "folds": fold_items,
    }


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


def run_task_bundle(
    args,
    run_root: Path,
    dataset_key: str,
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
        "dataset_key": dataset_key,
        "task_name": task_name,
        "target": target,
        "task_type": mode,
        "num_samples": int(len(task_df)),
        "num_folds": int(len(all_metrics)),
        "split_source": split_bundle["source"],
    }
    for key, value in summary.items():
        task_summary_row[key] = float(value)

    return {
        "task_summary_row": task_summary_row,
        "fold_metrics_rows": fold_metrics_rows,
    }


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
        for column in available_targets:
            print(column)
        return {
            "dataset": args.dataset,
            "available_targets": available_targets,
        }

    target_specs = resolve_target_specs(args, labels_df)
    model_name = infer_model_name(args)
    run_root = Path(args.out_root).expanduser().resolve() / model_name / dataset_key
    run_root.mkdir(parents=True, exist_ok=True)

    selected_folds = set(args.folds) if args.folds else None
    prepared = 0
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

        task_df = apply_required_label_columns(
            task_df=task_df,
            labels_df=labels_df,
            required_label_columns=args.required_label_columns,
        )
        if len(task_df) < args.min_samples:
            print(
                f"[SKIP] dataset={args.dataset} target={target}: "
                f"valid sample count < {args.min_samples} after required_label_columns."
            )
            continue
        if mode == "cls" and task_df[target].nunique() < 2:
            print(
                f"[SKIP] dataset={args.dataset} target={target}: "
                "only one class remains after required_label_columns."
            )
            continue

        task_name = str(spec.get("task_name") or runner.build_task_name(target, mode))
        split_bundle = runner.resolve_existing_split_bundle(
            dataset_name=dataset_key,
            task_name=task_name,
            split_root=Path(args.split_root).expanduser().resolve(),
            n_splits=args.n_splits,
            seed=args.seed,
            selected_folds=selected_folds,
        )
        prepared += 1

        split_task_root = (
            Path(args.split_root).expanduser().resolve()
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
                    "mapping": {
                        str(key): int(value) for key, value in label_map.items()
                    },
                },
            )

        task_bundles.append(
            {
                "task_name": task_name,
                "target": target,
                "mode": mode,
                "task_df": task_df,
                "label_map": label_map,
                "split_bundle": split_bundle,
            }
        )

    if args.prepare_only and not args.ckpt_path:
        print(
            f"[PREPARE_DONE] prepared {prepared} target-specific split datasets under "
            f"{Path(args.split_root).expanduser().resolve() / dataset_key}"
        )
        return {
            "dataset": dataset_key,
            "run_root": str(run_root),
            "num_tasks": 0,
            "num_fold_rows": 0,
            "task_summaries": [],
            "fold_metrics_rows": [],
            "resolved_config_path": None,
        }

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    payload, _ = runner.load_checkpoint_payload(ckpt_path)
    model_config = runner.resolve_model_config(
        args, runner.extract_base_model_hparams(payload)
    )

    resolved_devices = runner.normalize_device_list(args.devices, args.device)
    embedding_device_str = select_embedding_device(args, resolved_devices)
    args.device = embedding_device_str
    device = resolve_device(embedding_device_str)

    base_model = CGMBaseModel(
        patch_size=model_config["patch_size"],
        forecast_size=model_config["forecast_size"],
        embed_dim=model_config["embed_dim"],
        depth=model_config["depth"],
        num_heads=model_config["num_heads"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"],
        max_seq_len=model_config["max_seq_len"],
        use_cnn=model_config["use_cnn"],
        use_transformer=model_config["use_transformer"],
        use_temporal=model_config["use_temporal"],
        use_positional=model_config["use_positional"],
    )
    load_report = runner.load_base_weights(base_model, ckpt_path)
    base_model = base_model.to(device)

    pre_bundle = runner.pre_process_all_cgm(
        cgm_df, max_seq_len=model_config["max_seq_len"]
    )
    embedding_bundle = extract_or_load_embeddings(
        args=args,
        model_name=model_name,
        dataset_name=dataset_key,
        model=base_model,
        model_config=model_config,
        pre_bundle=pre_bundle,
        device=device,
    )

    resolved_config = {
        "dataset": args.dataset,
        "dataset_key": dataset_key,
        "dataset_variant": getattr(args, "dataset_variant", None),
        "sample_filters": getattr(args, "sample_filters", None),
        "targets": target_specs,
        "checkpoint": str(ckpt_path),
        "model_name": model_name,
        "device": str(device),
        "devices": resolved_devices,
        "pooling": args.pooling,
        "alpha_values": list(float(value) for value in args.alpha_values),
        "split_root": str(Path(args.split_root).expanduser().resolve()),
        "runtime": {
            "wait_for_idle_gpus": bool(args.wait_for_idle_gpus),
            "idle_min_memory_gb": float(args.idle_min_memory_gb),
            "idle_max_utilization": float(args.idle_max_utilization),
            "idle_poll_seconds": float(args.idle_poll_seconds),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
        },
        "embedding_cache": {
            "cache_path": embedding_bundle["cache_path"],
            "cache_meta_path": embedding_bundle["cache_meta_path"],
            "cache_status": embedding_bundle["cache_status"],
            "num_samples": int(len(embedding_bundle["sample_ids"])),
            "embedding_dim": int(embedding_bundle["embeddings"].shape[1]),
        },
        "model_config": model_config,
        "checkpoint_load_report": {
            "missing_keys": load_report["missing_keys"],
            "unexpected_keys": load_report["unexpected_keys"],
        },
    }
    runner.save_json(run_root / "_resolved_config.json", resolved_config)

    if args.prepare_only:
        print(
            f"[PREPARE_DONE] prepared {prepared} target-specific split datasets under "
            f"{Path(args.split_root).expanduser().resolve() / dataset_key}"
        )
        print(
            f"[EMBED_CACHE_READY] dataset={dataset_key} cache={embedding_bundle['cache_path']}"
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
    for task_bundle in task_bundles:
        print(
            f"[TASK_START] dataset={dataset_key} task={task_bundle['task_name']} "
            f"split_source={task_bundle['split_bundle']['source']}"
        )
        result = run_task_bundle(
            args=args,
            run_root=run_root,
            dataset_key=dataset_key,
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
