import argparse
import contextlib
import functools
import hashlib
import json
import queue
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.cuda.amp as amp
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.downstream_base_model import CGMBaseModel
from utils.downstream_data_utils import CGMForecastDataset
from utils.downstream_train_utils import WarmupCosineScheduler

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    cgm_csv: Path
    label_csvs: tuple[Path, ...]
    cgm_index_col: str | None = None


DATASET_SPECS = {
    "colas": DatasetSpec(
        name="colas",
        cgm_csv=REPO_ROOT / "data_lab" / "Colas" / "Colas_CGM_mmol.csv",
        label_csvs=(
            REPO_ROOT / "data_lab" / "Colas" / "Colas_incident_t2d_labels.csv",
        ),
        cgm_index_col="id",
    ),
    "zhao": DatasetSpec(
        name="zhao",
        cgm_csv=REPO_ROOT / "data_lab" / "zhao" / "Zhao_data.csv",
        label_csvs=(
            REPO_ROOT / "data_lab" / "zhao" / "Summary_Zhao.csv",
            REPO_ROOT / "data_lab" / "zhao" / "Zhao_label.csv",
        ),
        cgm_index_col=None,
    ),
    "nicheng": DatasetSpec(
        name="nicheng",
        cgm_csv=REPO_ROOT / "data_lab" / "Nicheng" / "Nicheng_CGM.csv",
        label_csvs=(
            REPO_ROOT
            / "data_lab"
            / "Nicheng"
            / "Nicheng_Clinical24_data_rename_filter.csv",
            REPO_ROOT
            / "data_lab"
            / "Nicheng"
            / "Nicheng_Clinical24_label_rename_filter.csv",
        ),
        cgm_index_col=None,
    ),
    "nicheng_new": DatasetSpec(
        name="nicheng_new",
        cgm_csv=REPO_ROOT / "data_lab" / "Nicheng" / "Nicheng_CGM.csv",
        label_csvs=(
            REPO_ROOT
            / "data_lab"
            / "Nicheng"
            / "Nicheng_Clinical24_data_rename_filter.csv",
            REPO_ROOT
            / "data_lab"
            / "Nicheng"
            / "Nicheng_Clinical24_label_rename_filter.csv",
            REPO_ROOT
            / "data_lab"
            / "Nicheng"
            / "Nicheng_Clinical24_label_new.csv",
        ),
        cgm_index_col=None,
    ),
    "hpp": DatasetSpec(
        name="hpp",
        cgm_csv=REPO_ROOT
        / "data_lab"
        / "HPP"
        / "participant_cgm_data_withevent_full_5min_3456_from00.csv",
        label_csvs=(
            REPO_ROOT / "data_lab" / "HPP" / "hpp_summary.csv",
            REPO_ROOT / "data_lab" / "HPP" / "cgm_dynamic_horizons_labeled.csv",
            REPO_ROOT
            / "data_lab"
            / "HPP"
            / "hpp_cgmanchored_incident_180d_labels.csv",
            REPO_ROOT / "data_lab" / "HPP" / "hpp_need_condition_labels.csv",
        ),
        cgm_index_col="id",
    ),
}

DEFAULT_BASE_MODEL_CONFIG = {
    "patch_size": 144,
    "forecast_size": 12,
    "embed_dim": 256,
    "depth": 12,
    "num_heads": 8,
    "hidden_dim": 1024,
    "dropout": 0.2,
    "max_seq_len": 3456,
    "use_cnn": True,
    "use_transformer": True,
    "use_temporal": True,
    "use_positional": True,
    "model_stride": 3,
}

CONFIG_PATH_KEYS = {
    "ckpt_path",
    "event_config_dir",
    "event_out_root",
    "split_root",
    "out_root",
}

CONFIG_SECTIONS = {
    "pretrain": {
        "ckpt_path",
        "event_pretrain_task",
        "event_config_dir",
        "event_out_root",
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
    "training": {
        "device",
        "devices",
        "epochs",
        "batch_size",
        "num_workers",
        "lr",
        "weight_decay",
        "layer_decay",
        "freeze_base",
        "unfreeze_after",
        "patience",
        "out_root",
        "disable_compile",
        "wait_for_idle_gpus",
        "idle_min_memory_gb",
        "idle_max_utilization",
        "idle_poll_seconds",
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
}


def set_gpu_flags():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reusable downstream CV runner for data_lab datasets. "
            "It persists dataset splits and reuses them across checkpoints."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config file. CLI arguments override config values.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=sorted(DATASET_SPECS.keys()),
        help="Dataset under data_lab: zhao / nicheng / hpp.",
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
    parser.add_argument(
        "--targets",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Target list. Use items like hba1c or reg:hba1c or cls:type. "
            "Required for training."
        ),
    )
    parser.add_argument("--list_targets", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--prepare_all_targets", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--event_pretrain_task", type=str, default=None)
    parser.add_argument(
        "--event_config_dir",
        type=str,
        default=str(REPO_ROOT / "configs" / "event_finetune_tasks"),
    )
    parser.add_argument(
        "--event_out_root",
        type=str,
        default=str(REPO_ROOT / "data_event" / "event_finetune_outputs"),
    )
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
        help="Optional list of CUDA devices for parallel task scheduling, e.g. cuda:0 cuda:1.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--layer_decay", type=float, default=0.7)
    parser.add_argument("--freeze_base", action="store_true")
    parser.add_argument("--unfreeze_after", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
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
        "--split_root",
        type=str,
        default=str(REPO_ROOT / "data_lab" / "lab_target_cv"),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "lab_cv"),
    )
    parser.add_argument("--force_regenerate_splits", action="store_true")
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--skip_run_aggregates", action="store_true")
    parser.add_argument("--wait_for_idle_gpus", action="store_true")
    parser.add_argument("--idle_min_memory_gb", type=float, default=8.0)
    parser.add_argument("--idle_max_utilization", type=float, default=30.0)
    parser.add_argument("--idle_poll_seconds", type=float, default=60.0)
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

    if not args.dataset:
        raise ValueError("dataset is required. Pass --dataset or provide it in --config.")
    args.sample_filters = normalize_sample_filters(getattr(args, "sample_filters", None))
    args.dataset_variant = normalize_dataset_variant(
        getattr(args, "dataset_variant", None)
    )
    return args


def parse_bool_override(raw_value: str | bool | int | None) -> bool | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return bool(raw_value)
    value = raw_value.strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean override: {raw_value}")


def infer_model_name(args) -> str:
    if args.model_name:
        return args.model_name
    if args.event_pretrain_task:
        return f"event_{args.event_pretrain_task}"
    if args.ckpt_path:
        return Path(args.ckpt_path).expanduser().stem
    return "prepare_only"


def sanitize_name(value: str) -> str:
    text = str(value).strip()
    chars = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    cleaned = "".join(chars).strip("_")
    return cleaned or "target"


def normalize_dataset_variant(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    return text


def normalize_filter_scalar(raw_value: Any, case_sensitive: bool) -> str | None:
    if raw_value is None:
        return None
    try:
        if pd.isna(raw_value):
            return None
    except TypeError:
        pass
    text = str(raw_value).strip()
    if not text:
        return None
    return text if case_sensitive else text.casefold()


def normalize_filter_values(raw_values: Any, case_sensitive: bool) -> list[str]:
    if raw_values is None:
        return []
    if isinstance(raw_values, (list, tuple, set)):
        values = list(raw_values)
    else:
        values = [raw_values]

    normalized = []
    for value in values:
        token = normalize_filter_scalar(value, case_sensitive)
        if token is not None:
            normalized.append(token)
    return normalized


def normalize_sample_filters(raw_filters: Any) -> list[dict[str, Any]]:
    if raw_filters is None:
        return []
    if isinstance(raw_filters, str):
        text = raw_filters.strip()
        if not text:
            return []
        try:
            raw_filters = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid sample_filters JSON: {exc}") from exc

    if isinstance(raw_filters, dict):
        raw_filters = [raw_filters]
    if not isinstance(raw_filters, list):
        raise ValueError("sample_filters must be a dict, list, or JSON string.")

    normalized_filters = []
    for index, item in enumerate(raw_filters, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"sample_filters[{index}] must be an object.")

        column = str(item.get("column") or item.get("label") or "").strip()
        if not column:
            raise ValueError(f"sample_filters[{index}] is missing 'column'.")

        case_sensitive = bool(item.get("case_sensitive", True))
        include_values = normalize_filter_values(
            item.get("include_values", item.get("include")),
            case_sensitive=case_sensitive,
        )
        exclude_values = normalize_filter_values(
            item.get("exclude_values", item.get("exclude")),
            case_sensitive=case_sensitive,
        )
        if not include_values and not exclude_values:
            raise ValueError(
                f"sample_filters[{index}] must define include_values and/or exclude_values."
            )

        normalized_filters.append(
            {
                "column": column,
                "case_sensitive": case_sensitive,
                "include_values": include_values,
                "exclude_values": exclude_values,
            }
        )
    return normalized_filters


def summarize_filter_values(values: list[str]) -> str:
    tokens = [sanitize_name(value).lower() for value in values if str(value).strip()]
    if not tokens:
        return "values"
    if len(tokens) > 3:
        tokens = tokens[:3] + ["etc"]
    return "_".join(tokens)


def infer_dataset_variant_from_filters(sample_filters: list[dict[str, Any]]) -> str | None:
    if not sample_filters:
        return None

    parts = []
    for spec in sample_filters:
        column = sanitize_name(spec["column"]).lower()
        include_values = spec.get("include_values") or []
        exclude_values = spec.get("exclude_values") or []
        if include_values:
            parts.append(f"{column}_only_{summarize_filter_values(include_values)}")
        if exclude_values:
            parts.append(f"{column}_without_{summarize_filter_values(exclude_values)}")
    if not parts:
        return None
    return "__".join(parts)


def resolve_dataset_key(
    dataset_name: str,
    dataset_variant: str | None = None,
    sample_filters: list[dict[str, Any]] | None = None,
) -> str:
    variant = normalize_dataset_variant(dataset_variant)
    if variant is None:
        variant = infer_dataset_variant_from_filters(sample_filters or [])
    if variant is None:
        return dataset_name
    return f"{dataset_name}__{sanitize_name(variant).lower()}"


def apply_sample_filters(
    labels_df: pd.DataFrame,
    sample_filters: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if not sample_filters:
        return labels_df

    filtered = labels_df.copy()
    for spec in sample_filters:
        column = spec["column"]
        if column not in filtered.columns:
            raise ValueError(f"Sample filter column not found: {column}")

        before = len(filtered)
        case_sensitive = bool(spec.get("case_sensitive", True))
        normalized_col = filtered[column].map(
            lambda value: normalize_filter_scalar(value, case_sensitive=case_sensitive)
        )
        mask = pd.Series(True, index=filtered.index, dtype=bool)
        include_values = spec.get("include_values") or []
        exclude_values = spec.get("exclude_values") or []
        if include_values:
            mask &= normalized_col.isin(include_values)
        if exclude_values:
            mask &= ~normalized_col.isin(exclude_values)

        filtered = filtered.loc[mask].copy()
        after = len(filtered)
        print(
            f"[SAMPLE_FILTER] column={column} "
            f"include={include_values or '-'} exclude={exclude_values or '-'} "
            f"kept={after}/{before}"
        )
        if after == 0:
            raise RuntimeError(
                f"Sample filters removed all samples after applying column '{column}'."
            )
    return filtered


def build_task_name(target: str, mode: str) -> str:
    return f"{mode}_{sanitize_name(target)}"


def normalize_table_index(df: pd.DataFrame, preferred_col: str | None = None) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    index_col = None
    if preferred_col and preferred_col in frame.columns:
        index_col = preferred_col
    elif "id" in frame.columns:
        index_col = "id"
    elif frame.index.name is not None:
        frame.index = frame.index.astype(str)
        return frame
    else:
        index_col = frame.columns[0]

    frame[index_col] = frame[index_col].astype(str)
    frame = frame.set_index(index_col, drop=True)
    frame.index = frame.index.astype(str)
    return frame


def merge_label_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0].copy()
    for frame in frames[1:]:
        for col in frame.columns:
            if col not in merged.columns:
                merged[col] = frame[col]
    merged.index = merged.index.astype(str)
    return merged


def load_dataset_frames(
    dataset_name: str,
    sample_filters: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = DATASET_SPECS[dataset_name]

    if spec.cgm_index_col is not None:
        cgm_df = pd.read_csv(spec.cgm_csv)
        cgm_df = normalize_table_index(cgm_df, preferred_col=spec.cgm_index_col)
    else:
        cgm_df = pd.read_csv(spec.cgm_csv, index_col=0)
        cgm_df.index = cgm_df.index.astype(str)

    label_frames = []
    for path in spec.label_csvs:
        frame = pd.read_csv(path)
        frame = normalize_table_index(frame, preferred_col="id")
        label_frames.append(frame)

    labels_df = merge_label_frames(label_frames)
    labels_df = labels_df.loc[~labels_df.index.duplicated(keep="first")]
    labels_df = apply_sample_filters(labels_df, sample_filters=sample_filters)
    cgm_df = cgm_df.loc[~cgm_df.index.duplicated(keep="first")]

    common_ids = [sid for sid in cgm_df.index.astype(str).tolist() if sid in labels_df.index]
    if not common_ids:
        raise RuntimeError(f"No common sample ids found for dataset {dataset_name}.")

    return cgm_df.loc[common_ids].copy(), labels_df.loc[common_ids].copy()


def resolve_numeric_cgm_columns(df: pd.DataFrame) -> list[str]:
    cols = [str(col) for col in df.columns if str(col).isdigit()]
    if not cols:
        raise ValueError("No numeric CGM time-step columns were found.")
    return sorted(cols, key=lambda value: int(value))


def compute_sample_hash(sample_ids: list[str]) -> str:
    joined = "\n".join(sample_ids).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def ensure_dataset_split(
    dataset_name: str,
    sample_ids: list[str],
    split_root: Path,
    n_splits: int,
    seed: int,
    force_regenerate: bool,
) -> dict[str, Any]:
    split_dir = split_root / dataset_name
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"kfold_{n_splits}_seed_{seed}.json"
    split_csv_path = split_dir / f"kfold_{n_splits}_seed_{seed}.csv"

    sample_hash = compute_sample_hash(sample_ids)
    expected = {
        "dataset": dataset_name,
        "n_splits": n_splits,
        "seed": seed,
        "num_samples": len(sample_ids),
        "sample_hash": sample_hash,
    }

    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        matches = all(payload.get(key) == value for key, value in expected.items())
        if matches:
            payload["_status"] = "reused"
            return payload
        if not force_regenerate:
            raise RuntimeError(
                "Existing split metadata does not match the current dataset state. "
                f"Delete {split_path} or rerun with --force_regenerate_splits."
            )

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sample_array = np.arange(len(sample_ids))
    fold_assignments: dict[str, int] = {}
    folds = []
    rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(sample_array), start=1):
        train_ids = [sample_ids[idx] for idx in train_idx.tolist()]
        val_ids = [sample_ids[idx] for idx in val_idx.tolist()]
        for sample_id in val_ids:
            fold_assignments[sample_id] = fold_idx
        folds.append(
            {
                "fold": fold_idx,
                "train_size": len(train_ids),
                "val_size": len(val_ids),
                "train_ids": train_ids,
                "val_ids": val_ids,
            }
        )

    for sample_id in sample_ids:
        rows.append({"sample_id": sample_id, "fold": fold_assignments[sample_id]})

    payload = {
        **expected,
        "splitter": "KFold",
        "created_from": "lab_cv_runner.py",
        "fold_assignments": fold_assignments,
        "folds": folds,
        "csv_path": str(split_csv_path),
        "_status": "created",
    }
    split_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(split_csv_path, index=False, encoding="utf-8")
    return payload


def infer_task_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "reg"

    if pd.api.types.is_object_dtype(non_null.dtype) or pd.api.types.is_string_dtype(
        non_null.dtype
    ):
        return "cls"

    numeric = pd.to_numeric(non_null, errors="coerce")
    if numeric.isna().any():
        return "cls"

    unique_count = int(numeric.nunique())
    integer_like = np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy()))
    if integer_like and 2 <= unique_count <= 10:
        return "cls"
    return "reg"


def parse_target_specs(
    raw_targets: list[Any] | None,
    labels_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not raw_targets:
        raise ValueError(
            "Training requires --targets. Use --list_targets to inspect available columns."
        )

    specs = []
    for item in raw_targets:
        if isinstance(item, dict):
            target = str(item.get("target", "")).strip()
            if not target:
                raise ValueError("Target config dict must contain a non-empty 'target'.")
            mode = str(item.get("mode", "")).strip().lower() or infer_task_type(labels_df[target])
            if mode not in {"reg", "cls"}:
                raise ValueError(
                    f"Unsupported target mode in config for {target}: {mode}. Use reg or cls."
                )
            spec = {"target": target, "mode": mode}
            if item.get("task_name"):
                spec["task_name"] = str(item["task_name"]).strip()
            if item.get("include_values") is not None:
                spec["include_values"] = list(item["include_values"])
            if item.get("exclude_values") is not None:
                spec["exclude_values"] = list(item["exclude_values"])
            if item.get("label_mapping") is not None:
                spec["label_mapping"] = {
                    key: int(value) for key, value in dict(item["label_mapping"]).items()
                }
        else:
            item = str(item)
            if ":" in item:
                prefix, target = item.split(":", 1)
                mode = prefix.strip().lower()
                if mode not in {"reg", "cls"}:
                    raise ValueError(f"Unsupported target mode in {item}. Use reg: or cls:.")
            else:
                target = item
                mode = infer_task_type(labels_df[target])
            target = target.strip()
            spec = {"target": target, "mode": mode}

        if target not in labels_df.columns:
            raise KeyError(f"Target column not found: {target}")
        specs.append(spec)
    return specs


def build_task_frame(
    labels_df: pd.DataFrame,
    spec: dict[str, Any],
    min_samples: int,
) -> tuple[pd.DataFrame | None, dict[str, int] | None]:
    target = spec["target"]
    mode = spec["mode"]
    sub_df = labels_df[[target]].copy()
    sub_df = sub_df.dropna(subset=[target])
    sub_df.index = sub_df.index.astype(str)

    include_values = spec.get("include_values")
    if include_values is not None:
        sub_df = sub_df[sub_df[target].isin(include_values)]

    exclude_values = spec.get("exclude_values")
    if exclude_values is not None:
        sub_df = sub_df[~sub_df[target].isin(exclude_values)]

    if mode == "reg":
        sub_df[target] = pd.to_numeric(sub_df[target], errors="coerce")
        sub_df = sub_df.dropna(subset=[target])
        if len(sub_df) < min_samples:
            return None, None
        return sub_df, None

    custom_label_map = spec.get("label_mapping")
    if custom_label_map is not None:
        sub_df[target] = sub_df[target].map(custom_label_map)
        sub_df = sub_df.dropna(subset=[target])
        label_values = sorted({int(value) for value in custom_label_map.values()})
        if label_values != list(range(len(label_values))):
            raise ValueError(
                f"label_mapping for {target} must map to contiguous class ids starting at 0."
            )
        label_map = {str(key): int(value) for key, value in custom_label_map.items()}
        if len(label_values) < 2:
            return None, None
    else:
        unique_vals = sorted(sub_df[target].unique().tolist(), key=lambda value: str(value))
        if len(unique_vals) < 2:
            return None, None
        label_map = {raw_value: idx for idx, raw_value in enumerate(unique_vals)}
        sub_df[target] = sub_df[target].map(label_map)

    sub_df[target] = sub_df[target].astype(np.int64)
    if len(sub_df) < min_samples:
        return None, None
    return sub_df, label_map


def resolve_num_classes(label_map: dict[Any, int] | None) -> int | None:
    if label_map is None:
        return None
    return len({int(value) for value in label_map.values()})


def compute_target_hash(task_df: pd.DataFrame, target_col: str) -> str:
    rows = [
        f"{sample_id}\t{task_df.at[sample_id, target_col]}"
        for sample_id in task_df.index.astype(str).tolist()
    ]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def export_target_split_table(
    table: pd.DataFrame,
    target_col: str,
    path: Path,
):
    export_df = table.copy()
    export_df.index = export_df.index.astype(str)
    export_df = export_df.reset_index()
    export_df = export_df.rename(columns={export_df.columns[0]: "sample_id"})
    export_df = export_df.rename(columns={target_col: "label"})
    export_df.to_csv(path, index=False, encoding="utf-8")


def ensure_target_split(
    dataset_name: str,
    task_name: str,
    target_col: str,
    task_type: str,
    task_df: pd.DataFrame,
    split_root: Path,
    n_splits: int,
    seed: int,
    force_regenerate: bool,
) -> dict[str, Any]:
    target_root = split_root / dataset_name / task_name / f"kfold_{n_splits}_seed_{seed}"
    target_root.mkdir(parents=True, exist_ok=True)

    split_path = target_root / "split_manifest.json"
    all_samples_csv = target_root / "all_samples.csv"
    task_df = task_df.copy()
    task_df.index = task_df.index.astype(str)

    sample_ids = task_df.index.astype(str).tolist()
    sample_array = np.arange(len(sample_ids))
    splitter_name = "KFold"
    split_iter = None

    # Preserve class balance for classification whenever every class can
    # support the requested number of folds.
    if task_type == "cls":
        stratify_labels = task_df[target_col].to_numpy()
        _, class_counts = np.unique(stratify_labels, return_counts=True)
        if class_counts.size >= 2 and int(class_counts.min()) >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(sample_array, stratify_labels)
            splitter_name = "StratifiedKFold"

    if split_iter is None:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(sample_array)

    expected = {
        "dataset": dataset_name,
        "task_name": task_name,
        "target_col": target_col,
        "task_type": task_type,
        "n_splits": n_splits,
        "seed": seed,
        "num_samples": int(len(task_df)),
        "target_hash": compute_target_hash(task_df, target_col),
    }

    if split_path.exists() and not force_regenerate:
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        matches = all(payload.get(key) == value for key, value in expected.items())
        if matches:
            existing_splitter = str(payload.get("splitter") or "KFold")
            payload["_status"] = (
                "reused"
                if existing_splitter == splitter_name
                else f"reused_{existing_splitter.lower()}"
            )
            payload["_requested_splitter"] = splitter_name
            return payload
        raise RuntimeError(
            "Existing target split metadata does not match current labeled samples. "
            f"Delete {split_path} or rerun with --force_regenerate_splits."
        )

    folds = []

    export_target_split_table(task_df, target_col, all_samples_csv)

    for fold_idx, (train_idx, val_idx) in enumerate(split_iter, start=1):
        fold_dir = target_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_ids = [sample_ids[idx] for idx in train_idx.tolist()]
        val_ids = [sample_ids[idx] for idx in val_idx.tolist()]
        train_df = task_df.loc[train_ids]
        val_df = task_df.loc[val_ids]

        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        export_target_split_table(train_df, target_col, train_csv)
        export_target_split_table(val_df, target_col, val_csv)

        folds.append(
            {
                "fold": fold_idx,
                "train_size": int(len(train_df)),
                "val_size": int(len(val_df)),
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
            }
        )

    payload = {
        **expected,
        "splitter": splitter_name,
        "created_from": "lab_cv_runner.py",
        "all_samples_csv": str(all_samples_csv),
        "folds": folds,
        "_status": "created",
    }
    split_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def load_task_split_ids(
    split_manifest_path: Path,
    selected_folds: set[int] | None,
) -> list[dict[str, Any]]:
    split_payload = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    fold_items = []
    for fold_info in split_payload["folds"]:
        fold_idx = int(fold_info["fold"])
        if selected_folds is not None and fold_idx not in selected_folds:
            continue
        train_ids = load_split_ids(fold_info["train_csv"])
        val_ids = load_split_ids(fold_info["val_csv"])
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
    if not task_manifest_path.exists():
        raise FileNotFoundError(
            f"No prepared target split was found. Expected: {task_manifest_path}"
        )
    fold_items = load_task_split_ids(task_manifest_path, selected_folds)
    return {
        "source": "target_split",
        "manifest_path": str(task_manifest_path),
        "folds": fold_items,
    }


def resolve_event_pretrain_checkpoint(
    task_name: str,
    config_dir: Path,
    out_root: Path,
) -> Path:
    config_path = config_dir / f"{task_name}.json"
    candidate_dir = None

    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        out_dir = Path(config["out_dir"]).expanduser()
        if not out_dir.is_absolute():
            out_dir = (REPO_ROOT / out_dir).resolve()
        candidate_dir = out_dir
    else:
        candidate_dir = (out_root / task_name).resolve()

    if not candidate_dir.exists():
        raise FileNotFoundError(
            f"Event pretrain output directory not found for task {task_name}: {candidate_dir}"
        )

    patterns = ["*_best.pt", "*_final.pt", "best*.pt", "*.pt"]
    candidates = []
    for pattern in patterns:
        candidates = sorted(candidate_dir.glob(pattern), key=lambda item: item.stat().st_mtime)
        if candidates:
            break
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint was found under event pretrain directory: {candidate_dir}"
        )
    return candidates[-1].resolve()


def pre_process_all_cgm(cgm_df: pd.DataFrame, max_seq_len: int) -> dict[str, Any]:
    cols = resolve_numeric_cgm_columns(cgm_df)
    raw_np = cgm_df[cols].to_numpy(dtype=np.float32)
    raw_np = raw_np[:, :max_seq_len]

    # Interpolate once and reuse across all targets and folds.
    interpolated = (
        pd.DataFrame(raw_np)
        .interpolate(method="linear", axis=1, limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    valid_lengths = []
    for row in raw_np:
        valid_idx = np.flatnonzero(~np.isnan(row))
        valid_lengths.append(int(valid_idx[-1]) + 1 if valid_idx.size > 0 else 0)

    sample_ids = cgm_df.index.astype(str).tolist()
    return {
        "data": torch.from_numpy(interpolated).share_memory_(),
        "sample_ids": sample_ids,
        "valid_lengths": valid_lengths,
        "id_to_idx": {sample_id: idx for idx, sample_id in enumerate(sample_ids)},
        "seq_len": interpolated.shape[1],
    }


class SlidingPatchDataset(Dataset):
    def __init__(
        self,
        pre_bundle: dict[str, Any],
        task_df: pd.DataFrame,
        target_col: str,
        task_type: str,
        patch_size: int,
        stride: int,
        forecast_steps: int = 0,
    ):
        task_df = task_df.copy()
        task_df.index = task_df.index.astype(str)

        all_cgm = pre_bundle["data"]
        all_ids_map = pre_bundle["id_to_idx"]
        common_ids = [sample_id for sample_id in task_df.index if sample_id in all_ids_map]
        if not common_ids:
            raise RuntimeError(f"No valid samples remain for target {target_col}.")

        self.sample_ids = common_ids
        self.patch_size = patch_size
        self.stride = stride
        self.forecast_steps = forecast_steps
        self.task_type = task_type
        self._on_gpu = False

        raw_labels = task_df.loc[common_ids, target_col].to_numpy(dtype=np.float32)
        if task_type == "reg":
            self.target_mean = float(np.nanmean(raw_labels))
            self.target_std = float(np.nanstd(raw_labels)) if np.nanstd(raw_labels) > 0 else 1.0
            labels = (raw_labels - self.target_mean) / self.target_std
            self.labels = torch.tensor(labels, dtype=torch.float32)
            self.raw_labels = torch.tensor(raw_labels, dtype=torch.float32)
        else:
            self.target_mean = None
            self.target_std = None
            self.labels = torch.tensor(raw_labels.astype(np.int64), dtype=torch.long)
            self.raw_labels = self.labels.clone()

        indices = [all_ids_map[sample_id] for sample_id in common_ids]
        subset = all_cgm[indices]
        seq_len = int(pre_bundle["seq_len"])
        max_start = seq_len - (patch_size + forecast_steps)
        if max_start < 0:
            raise ValueError(
                f"patch_size={patch_size} and forecast_steps={forecast_steps} exceed seq_len={seq_len}."
            )

        starts = list(range(0, max_start + 1, stride))
        if not starts:
            raise RuntimeError(f"No valid patches can be built for target {target_col}.")

        # pre_bundle["data"] is already interpolated/filled, so NaNs can no longer
        # identify trailing invalid CGM. Keep the original valid_lengths mask here.
        self.patches = subset.unfold(1, patch_size, stride)[:, : len(starts)].contiguous()
        patch_end_positions = torch.tensor(
            [start + patch_size + forecast_steps for start in starts],
            dtype=torch.int32,
        )
        valid_lengths = torch.tensor(
            [int(pre_bundle["valid_lengths"][idx]) for idx in indices],
            dtype=torch.int32,
        )
        self.masks = patch_end_positions.unsqueeze(0) <= valid_lengths.unsqueeze(1)

        self.patches.share_memory_()
        self.masks.share_memory_()
        self.labels.share_memory_()
        self.raw_labels.share_memory_()

    def estimated_bytes(self) -> int:
        return (
            self.patches.numel() * self.patches.element_size()
            + self.masks.numel() * self.masks.element_size()
            + self.labels.numel() * self.labels.element_size()
            + self.raw_labels.numel() * self.raw_labels.element_size()
        )

    def move_to_gpu(self, device: torch.device):
        self.patches = self.patches.to(device, non_blocking=True)
        self.masks = self.masks.to(device, non_blocking=True)
        self.labels = self.labels.to(device, non_blocking=True)
        self.raw_labels = self.raw_labels.to(device, non_blocking=True)
        self._on_gpu = True

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return {
            "sample_id": self.sample_ids[idx],
            "patches": self.patches[idx],
            "forecast_mask": self.masks[idx],
            "label": self.labels[idx],
            "raw_label": self.raw_labels[idx],
        }


def collate_fn(batch):
    return {
        "sample_ids": [item["sample_id"] for item in batch],
        "patches": torch.stack([item["patches"] for item in batch], dim=0),
        "forecast_mask": torch.stack([item["forecast_mask"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
    }


class FinetuneModel(nn.Module):
    def __init__(
        self,
        base_model: CGMBaseModel,
        dropout: float,
        model_stride: int,
        num_classes: int | None = None,
        regression_dim: int | None = None,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes
        self.regression_dim = regression_dim
        self.model_stride = model_stride

        embed_dim = base_model.embed_dim
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

        self.pool_attn = nn.Linear(embed_dim, 1)
        out_dim = num_classes if num_classes is not None else regression_dim
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, patches, forecast_mask=None):
        embeddings = self.base(
            patches,
            start_idx=0,
            stride=self.model_stride,
            return_embedding=True,
        )
        scores = self.pool_attn(embeddings).squeeze(-1)
        if forecast_mask is not None:
            valid_mask = forecast_mask.bool()
            scores = scores.masked_fill(~valid_mask, float("-inf"))
            empty_rows = ~valid_mask.any(dim=1)
            if torch.any(empty_rows):
                scores = scores.clone()
                scores[empty_rows] = 0.0
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1)
        if forecast_mask is not None:
            empty_rows = ~forecast_mask.bool().any(dim=1)
            if torch.any(empty_rows):
                pooled = pooled.clone()
                pooled[empty_rows] = 0.0
        out = self.head(pooled)
        if self.num_classes is not None:
            return {"logits": out}
        return {"regression": out}


def load_checkpoint_payload(path: Path) -> tuple[dict[str, Any] | Any, dict[str, Any]]:
    return _load_checkpoint_payload_cached(str(Path(path).expanduser().resolve()))


@functools.lru_cache(maxsize=8)
def _load_checkpoint_payload_cached(path_str: str) -> tuple[dict[str, Any] | Any, dict[str, Any]]:
    path = Path(path_str)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        state_dict = payload.get(
            "model_state_dict",
            payload.get("model_state", payload.get("state_dict", payload)),
        )
    else:
        state_dict = payload
    return payload, state_dict


def extract_base_model_hparams(payload: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    for key in ("model_hparams", "config", "args"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return {}


def normalize_state_key(key: str) -> str:
    for prefix in ("module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def extract_base_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    normalized = {normalize_state_key(key): value for key, value in state_dict.items()}
    if any(key.startswith("base.") for key in normalized):
        return {
            key[len("base.") :]: value
            for key, value in normalized.items()
            if key.startswith("base.")
        }
    return normalized


def resolve_model_config(args, ckpt_hparams: dict[str, Any]) -> dict[str, Any]:
    config = dict(DEFAULT_BASE_MODEL_CONFIG)

    hparam_map = {
        "patch_size": "patch_size",
        "forecast_size": "forecast_size",
        "forecast_steps": "forecast_size",
        "embed_dim": "embed_dim",
        "depth": "depth",
        "num_heads": "num_heads",
        "hidden_dim": "hidden_dim",
        "dropout": "dropout",
        "max_seq_len": "max_seq_len",
        "use_cnn": "use_cnn",
        "use_transformer": "use_transformer",
        "use_temporal": "use_temporal",
        "use_positional": "use_positional",
        "stride": "model_stride",
        "model_stride": "model_stride",
    }
    for src_key, dst_key in hparam_map.items():
        if src_key in ckpt_hparams and ckpt_hparams[src_key] is not None:
            config[dst_key] = ckpt_hparams[src_key]

    cli_overrides = {
        "patch_size": args.patch_size,
        "forecast_size": args.forecast_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "max_seq_len": args.max_seq_len,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            config[key] = value

    if args.stride is not None:
        config["model_stride"] = args.stride

    for key in ("use_cnn", "use_transformer", "use_temporal", "use_positional"):
        override = parse_bool_override(getattr(args, key))
        if override is not None:
            config[key] = override

    config["patch_size"] = int(config["patch_size"])
    config["forecast_size"] = int(config["forecast_size"])
    config["embed_dim"] = int(config["embed_dim"])
    config["depth"] = int(config["depth"])
    config["num_heads"] = int(config["num_heads"])
    config["hidden_dim"] = int(config["hidden_dim"])
    config["max_seq_len"] = int(config["max_seq_len"])
    config["model_stride"] = int(config["model_stride"])
    config["dropout"] = float(config["dropout"])
    config["use_cnn"] = bool(config["use_cnn"])
    config["use_transformer"] = bool(config["use_transformer"])
    config["use_temporal"] = bool(config["use_temporal"])
    config["use_positional"] = bool(config["use_positional"])
    return config


def load_base_weights(base_model: CGMBaseModel, ckpt_path: Path) -> dict[str, Any]:
    payload, raw_state_dict = load_checkpoint_payload(ckpt_path)
    base_state_dict = extract_base_state_dict(raw_state_dict)
    missing_keys, unexpected_keys = base_model.load_state_dict(
        base_state_dict, strict=False
    )
    return {
        "missing_keys": sorted(missing_keys),
        "unexpected_keys": sorted(unexpected_keys),
        "payload": payload,
    }


def get_layerwise_optimizer(
    model: FinetuneModel,
    base_lr: float,
    layer_decay: float,
    weight_decay: float,
) -> optim.AdamW:
    param_groups = []
    head_params = list(model.pool_attn.parameters()) + list(model.head.parameters())
    param_groups.append({"params": head_params, "lr": base_lr})

    depth = getattr(model.base.transformer, "num_layers", 12)
    for name, param in model.base.named_parameters():
        if not param.requires_grad:
            continue
        layer_idx = depth - 1
        for i in range(depth):
            if f".{i}." in name:
                layer_idx = i
                break
        lr_i = base_lr * (layer_decay ** max(depth - 1 - layer_idx, 0))
        param_groups.append({"params": [param], "lr": lr_i})

    return optim.AdamW(param_groups, weight_decay=weight_decay)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None,
    task_type: str,
    num_classes: int | None,
) -> dict[str, float]:
    metrics = {}
    if task_type == "reg":
        metrics["MAE"] = float(mean_absolute_error(y_true, y_pred))
        metrics["RMSE"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        nonzero_mask = np.abs(y_true) > 1e-6
        metrics["MAPE"] = (
            float(
                np.mean(
                    np.abs(
                        (y_true[nonzero_mask] - y_pred[nonzero_mask])
                        / y_true[nonzero_mask]
                    )
                )
                * 100
            )
            if nonzero_mask.sum() > 0
            else float("nan")
        )
        metrics["R2"] = float(r2_score(y_true, y_pred))

        if len(y_true) > 2:
            pearson_r, pearson_p = pearsonr(y_true, y_pred)
            spearman_r, _ = spearmanr(y_true, y_pred)
            metrics["Pearson_r"] = float(pearson_r)
            metrics["Pearson_p"] = float(pearson_p)
            metrics["Spearman_r"] = float(spearman_r)
        else:
            metrics["Pearson_r"] = float("nan")
            metrics["Pearson_p"] = float("nan")
            metrics["Spearman_r"] = float("nan")
        return metrics

    metrics["Acc"] = float(accuracy_score(y_true, y_pred))
    metrics["F1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["MCC"] = float(matthews_corrcoef(y_true, y_pred))

    metrics["AUC"] = float("nan")
    if y_probs is None or num_classes is None:
        return metrics

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        return metrics

    try:
        if num_classes == 2:
            if y_probs.ndim != 2 or y_probs.shape[1] < 2:
                return metrics
            metrics["AUC"] = float(roc_auc_score(y_true, y_probs[:, 1]))
            return metrics

        if y_probs.ndim != 2 or y_probs.shape[1] != num_classes:
            return metrics
        if unique_labels.size != num_classes:
            return metrics

        metrics["AUC"] = float(
            roc_auc_score(
                y_true,
                y_probs,
                labels=np.arange(num_classes),
                multi_class="ovr",
            )
        )
    except Exception:
        metrics["AUC"] = float("nan")
    return metrics


def sanitize_json_value(value: Any):
    if isinstance(value, dict):
        return {str(key): sanitize_json_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def build_prediction_table(
    sample_ids: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    y_probs: np.ndarray | None,
    y_true_raw: np.ndarray | None = None,
    y_pred_raw: np.ndarray | None = None,
) -> pd.DataFrame:
    if task_type == "reg":
        table = pd.DataFrame(
            {
                "sample_id": [str(sample_id) for sample_id in sample_ids],
                "y_true_raw": y_true_raw if y_true_raw is not None else y_true,
                "y_pred_raw": y_pred_raw if y_pred_raw is not None else y_pred,
                "y_true_norm": y_true,
                "y_pred_norm": y_pred,
            }
        )
        table["error_raw"] = table["y_pred_raw"] - table["y_true_raw"]
        table["abs_error_raw"] = table["error_raw"].abs()
        table["squared_error_raw"] = table["error_raw"] ** 2
        table["error_norm"] = table["y_pred_norm"] - table["y_true_norm"]
        table["abs_error_norm"] = table["error_norm"].abs()
        table["squared_error_norm"] = table["error_norm"] ** 2
        return table

    table = pd.DataFrame(
        {
            "sample_id": [str(sample_id) for sample_id in sample_ids],
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    table["y_true"] = table["y_true"].astype(np.int64)
    table["y_pred"] = table["y_pred"].astype(np.int64)
    table["correct"] = (table["y_true"] == table["y_pred"]).astype(np.int64)
    if y_probs is not None:
        for class_idx in range(y_probs.shape[1]):
            table[f"prob_class_{class_idx}"] = y_probs[:, class_idx]
        table["pred_confidence"] = np.max(y_probs, axis=1)
    return table


def save_prediction_artifacts(
    out_dir: Path,
    task_type: str,
    metrics: dict[str, float],
    epoch: int,
    val_loss: float,
    eval_outputs: dict[str, Any],
):
    prediction_table = build_prediction_table(
        sample_ids=eval_outputs["sample_ids"],
        y_true=eval_outputs["y_true"],
        y_pred=eval_outputs["y_pred"],
        task_type=task_type,
        y_probs=eval_outputs["y_probs"],
        y_true_raw=eval_outputs.get("y_true_raw"),
        y_pred_raw=eval_outputs.get("y_pred_raw"),
    )
    prediction_table.to_csv(
        out_dir / "best_predictions.csv",
        index=False,
        encoding="utf-8",
    )

    summary_payload = {
        "best_epoch": epoch,
        "best_val_loss": float(val_loss),
        "num_val_samples": int(len(prediction_table)),
        "metrics": metrics,
    }
    if task_type == "reg" and eval_outputs.get("y_true_raw") is not None:
        summary_payload["metrics_raw_scale"] = compute_metrics(
            eval_outputs["y_true_raw"],
            eval_outputs["y_pred_raw"],
            y_probs=None,
            task_type=task_type,
            num_classes=None,
        )
        summary_payload["metrics_note"] = (
            "best_metrics.json follows original zhao_lab.py behavior: "
            "regression metrics are computed on normalized labels."
        )

    save_json(
        out_dir / "best_eval_summary.json",
        summary_payload,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    task_type: str,
    num_classes: int | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype,
    target_mean: float | None = None,
    target_std: float | None = None,
):
    model.eval()
    all_sample_ids = []
    all_true = []
    all_pred = []
    all_probs = []
    all_true_raw = []
    all_pred_raw = []
    losses = []

    with torch.no_grad():
        autocast_ctx = (
            amp.autocast(enabled=amp_enabled, dtype=amp_dtype)
            if device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            for batch in loader:
                all_sample_ids.extend(str(sample_id) for sample_id in batch["sample_ids"])
                patches = batch["patches"].to(device, non_blocking=True)
                mask = batch["forecast_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                out = model(patches, forecast_mask=mask)

                if task_type == "reg":
                    pred_norm = out["regression"].view(-1)
                    true_norm = labels.view(-1).to(pred_norm.dtype)
                    losses.append(loss_fn(pred_norm, true_norm).item())
                    pred_norm_np = pred_norm.float().cpu().numpy()
                    true_norm_np = labels.view(-1).cpu().numpy()
                    all_pred.append(pred_norm_np)
                    all_true.append(true_norm_np)
                    if target_mean is not None and target_std is not None:
                        all_pred_raw.append(pred_norm_np * target_std + target_mean)
                        all_true_raw.append(true_norm_np * target_std + target_mean)
                else:
                    logits = out["logits"]
                    losses.append(loss_fn(logits, labels).item())
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs.float().cpu().numpy())
                    all_pred.append(torch.argmax(probs, dim=1).cpu().numpy())
                    all_true.append(labels.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_probs = np.concatenate(all_probs) if all_probs else None
    metrics = compute_metrics(y_true, y_pred, y_probs, task_type, num_classes)
    eval_outputs = {
        "sample_ids": all_sample_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
        "y_true_raw": np.concatenate(all_true_raw) if all_true_raw else None,
        "y_pred_raw": np.concatenate(all_pred_raw) if all_pred_raw else None,
    }
    return float(np.mean(losses)), metrics, eval_outputs


def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    layer_decay: float,
    device: torch.device,
    task_type: str,
    num_classes: int | None,
    freeze_base: bool,
    unfreeze_after: int,
    out_dir: Path,
    patience: int,
    disable_compile: bool,
    class_weights: torch.Tensor | None = None,
    target_mean: float | None = None,
    target_std: float | None = None,
):
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model.to(device)

    if device.type == "cuda" and not disable_compile:
        try:
            model = torch.compile(model)
        except Exception:
            pass

    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    )
    scaler = amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    if task_type == "reg":
        loss_fn = nn.MSELoss()
    else:
        weight = class_weights.to(device) if class_weights is not None else None
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=min(100, max(len(train_loader) * 5, 1)),
        max_steps=max(unfreeze_after, 1) * max(len(train_loader), 1),
    )

    best_val_loss = float("inf")
    best_selection_score = -float("inf")
    patience_counter = 0
    history = []
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        if freeze_base and epoch == unfreeze_after + 1:
            for param in model.parameters():
                param.requires_grad = True
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            optimizer = get_layerwise_optimizer(
                raw_model,
                base_lr=lr * 0.1,
                layer_decay=layer_decay,
                weight_decay=weight_decay,
            )
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_steps=0,
                max_steps=max(epochs - unfreeze_after, 1) * max(len(train_loader), 1),
            )

        model.train()
        train_losses = []
        for batch in train_loader:
            patches = batch["patches"].to(device, non_blocking=True)
            mask = batch["forecast_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (
                amp.autocast(enabled=amp_enabled, dtype=amp_dtype)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                out = model(patches, forecast_mask=mask)
                if task_type == "reg":
                    pred_norm = out["regression"].view(-1)
                    loss = loss_fn(pred_norm, labels.view(-1).to(pred_norm.dtype))
                else:
                    loss = loss_fn(out["logits"], labels)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            train_losses.append(float(loss.item()))

        val_loss, val_metrics, val_outputs = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            task_type=task_type,
            num_classes=num_classes,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            target_mean=target_mean,
            target_std=target_std,
        )
        history_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(val_loss),
            "is_best": False,
            **{key: float(value) for key, value in val_metrics.items()},
        }

        selection_metric = "val_loss"
        selection_score = -float(val_loss)
        if task_type == "cls":
            auc_value = float(val_metrics.get("AUC", float("nan")))
            if np.isfinite(auc_value):
                selection_metric = "AUC"
                selection_score = auc_value
        history_row["selection_metric"] = selection_metric
        history_row["selection_score"] = float(selection_score)

        if selection_score > best_selection_score:
            best_val_loss = val_loss
            best_selection_score = selection_score
            patience_counter = 0
            best_metrics = {
                **val_metrics,
                "selection_metric": selection_metric,
                "selection_score": float(selection_score),
            }
            history_row["is_best"] = True
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(raw_model.state_dict(), out_dir / "best_model.pt")
            save_json(out_dir / "best_metrics.json", best_metrics)
            save_prediction_artifacts(
                out_dir=out_dir,
                task_type=task_type,
                metrics=best_metrics,
                epoch=epoch,
                val_loss=val_loss,
                eval_outputs=val_outputs,
            )
        else:
            patience_counter += 1

        history.append(history_row)
        if patience_counter >= patience:
            break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8")
    return best_metrics


def is_numeric_metric(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.isfinite(float(value))
    return False


def summarize_metrics(all_metrics: list[dict[str, Any]]) -> dict[str, float]:
    summary = {}
    if not all_metrics:
        return summary

    metric_keys = sorted({key for metrics in all_metrics for key in metrics.keys()})
    for key in metric_keys:
        values = [
            float(metrics[key])
            for metrics in all_metrics
            if key in metrics and is_numeric_metric(metrics[key])
        ]
        if not values:
            continue
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary


def resolve_fold_subset(
    split_payload: dict[str, Any],
    task_ids: set[str],
    selected_folds: set[int] | None,
) -> list[dict[str, Any]]:
    fold_items = []
    for fold_info in split_payload["folds"]:
        fold_idx = int(fold_info["fold"])
        if selected_folds is not None and fold_idx not in selected_folds:
            continue
        train_ids = [sample_id for sample_id in fold_info["train_ids"] if sample_id in task_ids]
        val_ids = [sample_id for sample_id in fold_info["val_ids"] if sample_id in task_ids]
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


def build_id_index_map(sample_ids: list[str]) -> dict[str, int]:
    return {sample_id: idx for idx, sample_id in enumerate(sample_ids)}


def load_split_ids(csv_path: str | Path) -> list[str]:
    split_df = pd.read_csv(csv_path)
    if "sample_id" in split_df.columns:
        series = split_df["sample_id"]
    else:
        series = split_df.iloc[:, 0]
    return series.astype(str).tolist()


def save_json(path: Path, payload: dict[str, Any]):
    path.write_text(
        json.dumps(sanitize_json_value(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_device_list(raw_devices: Any, fallback_device: str) -> list[str]:
    if raw_devices is None:
        return [str(fallback_device)]
    if isinstance(raw_devices, str):
        parts = [part.strip() for part in raw_devices.split(",") if part.strip()]
        return parts or [str(fallback_device)]
    return [str(item).strip() for item in raw_devices if str(item).strip()]


def parse_cuda_device_index(device: str) -> int | None:
    device = str(device).strip().lower()
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


def query_gpu_status_via_nvidia_smi() -> dict[int, dict[str, float | int | None]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return {}

    statuses: dict[int, dict[str, float | int | None]] = {}
    for raw_line in completed.stdout.strip().splitlines():
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpu_idx = int(parts[0])
            free_mb = float(parts[1])
            util = float(parts[2])
        except ValueError:
            continue
        statuses[gpu_idx] = {
            "free_memory_gb": free_mb / 1024.0,
            "utilization": util,
        }
    return statuses


def get_gpu_status(device: str) -> dict[str, Any]:
    gpu_idx = parse_cuda_device_index(device)
    if gpu_idx is None:
        return {
            "device": device,
            "gpu_index": None,
            "free_memory_gb": None,
            "utilization": None,
        }

    smi_statuses = query_gpu_status_via_nvidia_smi()
    if gpu_idx in smi_statuses:
        return {
            "device": device,
            "gpu_index": gpu_idx,
            **smi_statuses[gpu_idx],
        }

    if not torch.cuda.is_available() or gpu_idx >= torch.cuda.device_count():
        return {
            "device": device,
            "gpu_index": gpu_idx,
            "free_memory_gb": None,
            "utilization": None,
        }

    try:
        with torch.cuda.device(gpu_idx):
            free_bytes, _ = torch.cuda.mem_get_info()
        return {
            "device": device,
            "gpu_index": gpu_idx,
            "free_memory_gb": float(free_bytes) / (1024.0**3),
            "utilization": None,
        }
    except Exception:
        return {
            "device": device,
            "gpu_index": gpu_idx,
            "free_memory_gb": None,
            "utilization": None,
        }


def is_device_idle(
    device: str,
    min_free_memory_gb: float,
    max_utilization: float,
) -> tuple[bool, dict[str, Any]]:
    status = get_gpu_status(device)
    free_memory_gb = status.get("free_memory_gb")
    utilization = status.get("utilization")

    free_ok = free_memory_gb is None or free_memory_gb >= min_free_memory_gb
    util_ok = utilization is None or utilization <= max_utilization
    return bool(free_ok and util_ok), status


def collect_aggregate_rows(
    result: dict[str, Any],
    all_task_summaries: list[dict[str, Any]],
    all_fold_metrics: list[dict[str, Any]],
):
    task_summary_row = result.get("task_summary_row")
    if task_summary_row is not None:
        all_task_summaries.append(task_summary_row)
    for row in result.get("fold_metrics_rows", []):
        all_fold_metrics.append(row)


def write_run_aggregates(
    run_root: Path,
    all_task_summaries: list[dict[str, Any]],
    all_fold_metrics: list[dict[str, Any]],
):
    if all_fold_metrics:
        pd.DataFrame(all_fold_metrics).to_csv(
            run_root / "all_folds_metrics.csv",
            index=False,
            encoding="utf-8",
        )
        save_json(
            run_root / "all_folds_metrics.json",
            {"rows": all_fold_metrics},
        )
    if all_task_summaries:
        pd.DataFrame(all_task_summaries).to_csv(
            run_root / "all_tasks_cv_summary.csv",
            index=False,
            encoding="utf-8",
        )
        save_json(
            run_root / "all_tasks_cv_summary.json",
            {"rows": all_task_summaries},
        )


def run_task_bundle(
    task_bundle: dict[str, Any],
    run_root: Path,
    pre_bundle: dict[str, Any],
    model_config: dict[str, Any],
    ckpt_path: Path,
    runtime_config: dict[str, Any],
    device_str: str,
) -> dict[str, Any]:
    target = task_bundle["target"]
    mode = task_bundle["mode"]
    task_name = task_bundle["task_name"]
    task_df = task_bundle["task_df"]
    label_map = task_bundle["label_map"]
    split_payload = task_bundle["split_payload"]
    split_manifest_path = Path(task_bundle["split_manifest_path"])
    selected_folds = runtime_config["selected_folds"]
    args_device = torch.device(device_str)
    if args_device.type == "cuda":
        torch.cuda.set_device(args_device)

    dataset = SlidingPatchDataset(
        pre_bundle=pre_bundle,
        task_df=task_df,
        target_col=target,
        task_type=mode,
        patch_size=model_config["patch_size"],
        stride=model_config["model_stride"],
        forecast_steps=0,
    )
    use_gpu_dataset = False
    if args_device.type == "cuda":
        try:
            gpu_index = parse_cuda_device_index(device_str)
            if gpu_index is not None:
                free_bytes, _ = torch.cuda.mem_get_info(gpu_index)
            else:
                free_bytes, _ = torch.cuda.mem_get_info()
            dataset_bytes = dataset.estimated_bytes()
            if dataset_bytes <= free_bytes * 0.3:
                dataset.move_to_gpu(args_device)
                use_gpu_dataset = True
                print(
                    f"[MOVE_DATASET_TO_GPU] {task_name} device={device_str} "
                    f"dataset={dataset_bytes / (1024.0 ** 2):.1f}MB"
                )
            else:
                print(
                    f"[KEEP_DATASET_ON_CPU] {task_name} device={device_str} "
                    f"dataset={dataset_bytes / (1024.0 ** 2):.1f}MB "
                    f"free={free_bytes / (1024.0 ** 2):.1f}MB"
                )
        except Exception as exc:
            print(f"[MOVE_DATASET_TO_GPU_FAIL] {task_name} device={device_str} error={exc}")
    sample_id_to_local_idx = build_id_index_map(dataset.sample_ids)

    task_dir = run_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        task_dir / "task_summary.json",
        {
            "dataset": runtime_config["dataset"],
            "target": target,
            "task_type": mode,
            "num_samples": len(dataset.sample_ids),
            "num_classes": resolve_num_classes(label_map),
            "split_manifest": str(split_manifest_path),
            "target_spec": task_bundle["spec"],
            "device": device_str,
            "available_folds": [
                int(item["fold"])
                for item in split_payload["folds"]
                if (selected_folds is None or int(item["fold"]) in selected_folds)
            ],
        },
    )

    all_metrics = []
    fold_metrics_rows = []
    for fold_info in split_payload["folds"]:
        fold_idx = int(fold_info["fold"])
        if selected_folds is not None and fold_idx not in selected_folds:
            continue

        train_ids = load_split_ids(fold_info["train_csv"])
        val_ids = load_split_ids(fold_info["val_csv"])

        train_indices = [
            sample_id_to_local_idx[sample_id]
            for sample_id in train_ids
            if sample_id in sample_id_to_local_idx
        ]
        val_indices = [
            sample_id_to_local_idx[sample_id]
            for sample_id in val_ids
            if sample_id in sample_id_to_local_idx
        ]
        if not train_indices or not val_indices:
            print(f"[SKIP] {task_name} fold={fold_idx}: empty train or val after loading split files.")
            continue

        num_classes = resolve_num_classes(label_map)
        class_weights = None
        if mode == "cls":
            train_labels_for_fold = dataset.labels[train_indices].detach().cpu()
            if torch.unique(train_labels_for_fold).numel() < 2:
                print(
                    f"[SKIP] {task_name} fold={fold_idx}: "
                    "classification train split only has one class."
                )
                continue
            if num_classes is not None:
                class_counts = torch.bincount(
                    train_labels_for_fold.to(torch.long),
                    minlength=int(num_classes),
                ).to(torch.float32)
                if torch.all(class_counts > 0):
                    class_weights = class_counts.sum() / (
                        float(num_classes) * class_counts
                    )

        fold_dir = task_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            fold_dir / "fold_membership.json",
            {
                "fold": fold_idx,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "train_csv": fold_info["train_csv"],
                "val_csv": fold_info["val_csv"],
                "device": device_str,
            },
        )

        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)

        loader_num_workers = 0 if use_gpu_dataset else runtime_config["num_workers"]
        loader_kwargs = dict(
            collate_fn=collate_fn,
            num_workers=loader_num_workers,
            pin_memory=args_device.type == "cuda" and not use_gpu_dataset,
            persistent_workers=loader_num_workers > 0,
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=runtime_config["batch_size"],
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=runtime_config["batch_size"],
            shuffle=False,
            **loader_kwargs,
        )

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
        load_report = load_base_weights(base_model, ckpt_path)
        save_json(
            fold_dir / "checkpoint_load_report.json",
            {
                "checkpoint": str(ckpt_path),
                "missing_keys": load_report["missing_keys"],
                "unexpected_keys": load_report["unexpected_keys"],
            },
        )

        model = FinetuneModel(
            base_model=base_model,
            dropout=model_config["dropout"],
            model_stride=model_config["model_stride"],
            num_classes=num_classes,
            regression_dim=1 if mode == "reg" else None,
            freeze_base=runtime_config["freeze_base"],
        )

        target_mean = None
        target_std = None
        if mode == "reg":
            target_mean = float(dataset.target_mean)
            target_std = float(dataset.target_std)
            save_json(
                fold_dir / "target_stats.json",
                {
                    "target": target,
                    "normalization_scope": "all_task_samples_before_cv",
                    "target_mean": target_mean,
                    "target_std": target_std,
                },
            )

        metrics = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=runtime_config["epochs"],
            lr=runtime_config["lr"],
            weight_decay=runtime_config["weight_decay"],
            layer_decay=runtime_config["layer_decay"],
            device=args_device,
            task_type=mode,
            num_classes=num_classes,
            freeze_base=runtime_config["freeze_base"],
            unfreeze_after=runtime_config["unfreeze_after"],
            out_dir=fold_dir,
            patience=runtime_config["patience"],
            disable_compile=runtime_config["disable_compile"],
            class_weights=class_weights,
            target_mean=target_mean,
            target_std=target_std,
        )
        all_metrics.append(metrics)
        fold_metrics_row = {
            "dataset": runtime_config["dataset"],
            "task_name": task_name,
            "target": target,
            "task_type": mode,
            "fold": fold_idx,
            "device": device_str,
            "train_size": len(train_indices),
            "val_size": len(val_indices),
        }
        for key, value in metrics.items():
            fold_metrics_row[key] = float(value) if is_numeric_metric(value) else value
        fold_metrics_rows.append(fold_metrics_row)
        metric_parts = []
        for key, value in metrics.items():
            if is_numeric_metric(value):
                metric_parts.append(f"{key}={float(value):.4f}")
            else:
                metric_parts.append(f"{key}={value}")
        print(
            f"[DONE] {task_name} fold={fold_idx} device={device_str} "
            + " ".join(metric_parts)
        )

        del train_loader, val_loader, train_subset, val_subset, model, base_model
        if args_device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize(args_device)

    summary = summarize_metrics(all_metrics)
    save_json(task_dir / "cv_summary.json", summary)
    task_summary_row = {
        "dataset": runtime_config["dataset"],
        "task_name": task_name,
        "target": target,
        "task_type": mode,
        "device": device_str,
        "num_samples": int(len(task_df)),
        "num_folds": int(len(all_metrics)),
    }
    task_summary_row.update({key: float(value) for key, value in summary.items()})
    print(f"[SUMMARY] {task_name} device={device_str} -> {json.dumps(summary, ensure_ascii=False)}")
    return {
        "task_name": task_name,
        "task_summary_row": task_summary_row,
        "fold_metrics_rows": fold_metrics_rows,
    }


def gpu_worker(
    task_queue,
    result_queue,
    device_str: str,
    run_root_str: str,
    pre_bundle: dict[str, Any],
    model_config: dict[str, Any],
    ckpt_path_str: str,
    runtime_config: dict[str, Any],
):
    try:
        set_gpu_flags()
        args_device = torch.device(device_str)
        if args_device.type == "cuda":
            torch.cuda.set_device(args_device)
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
                task_bundle=task_bundle,
                run_root=Path(run_root_str),
                pre_bundle=pre_bundle,
                model_config=model_config,
                ckpt_path=Path(ckpt_path_str),
                runtime_config=runtime_config,
                device_str=device_str,
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
    pre_bundle: dict[str, Any],
    model_config: dict[str, Any],
    ckpt_path: Path,
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

    runtime_config = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "layer_decay": args.layer_decay,
        "freeze_base": args.freeze_base,
        "unfreeze_after": args.unfreeze_after,
        "patience": args.patience,
        "disable_compile": args.disable_compile,
        "selected_folds": set(args.folds) if args.folds else None,
    }

    pending_devices = list(devices)
    workers: dict[str, mp.Process] = {}
    all_task_summaries: list[dict[str, Any]] = []
    all_fold_metrics: list[dict[str, Any]] = []
    expected_task_results = len(task_bundles)
    received_task_results = 0
    worker_errors: list[str] = []

    while received_task_results < expected_task_results or workers:
        for device_str in pending_devices.copy():
            if received_task_results >= expected_task_results:
                break
            if not args.wait_for_idle_gpus:
                idle = True
                status = get_gpu_status(device_str)
            else:
                idle, status = is_device_idle(
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
                    pre_bundle,
                    model_config,
                    str(ckpt_path),
                    runtime_config,
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
                collect_aggregate_rows(message, all_task_summaries, all_fold_metrics)
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


def main():
    args = parse_args()
    set_gpu_flags()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_key = resolve_dataset_key(
        args.dataset,
        dataset_variant=args.dataset_variant,
        sample_filters=args.sample_filters,
    )
    cgm_df, labels_df = load_dataset_frames(
        args.dataset,
        sample_filters=args.sample_filters,
    )

    if args.list_targets:
        for column in labels_df.columns.tolist():
            print(column)
        return

    split_root = Path(args.split_root).expanduser().resolve()
    selected_folds = set(args.folds) if args.folds else None

    if args.targets:
        target_specs = parse_target_specs(args.targets, labels_df)
    elif args.prepare_only or args.prepare_all_targets:
        target_specs = parse_target_specs(labels_df.columns.tolist(), labels_df)
    else:
        raise ValueError("Training requires --targets.")

    ckpt_path = None
    model_config = None
    pre_bundle = None
    run_root = None
    all_task_summaries = []
    all_fold_metrics = []
    task_bundles = []

    if not args.prepare_only:
        if args.ckpt_path:
            ckpt_path = Path(args.ckpt_path).expanduser().resolve()
        elif args.event_pretrain_task:
            ckpt_path = resolve_event_pretrain_checkpoint(
                task_name=args.event_pretrain_task,
                config_dir=Path(args.event_config_dir).expanduser().resolve(),
                out_root=Path(args.event_out_root).expanduser().resolve(),
            )
        else:
            raise ValueError(
                "Training requires --ckpt_path or --event_pretrain_task."
            )

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        payload, _ = load_checkpoint_payload(ckpt_path)
        ckpt_hparams = extract_base_model_hparams(payload)
        model_config = resolve_model_config(args, ckpt_hparams)
        pre_bundle = pre_process_all_cgm(cgm_df, max_seq_len=model_config["max_seq_len"])

        out_root = Path(args.out_root).expanduser().resolve()
        model_name = infer_model_name(args)
        resolved_devices = normalize_device_list(args.devices, args.device)
        run_root = out_root / model_name / dataset_key
        run_root.mkdir(parents=True, exist_ok=True)

        run_config = {
            "dataset": args.dataset,
            "dataset_key": dataset_key,
            "dataset_variant": normalize_dataset_variant(args.dataset_variant),
            "sample_filters": args.sample_filters,
            "targets": target_specs,
            "config_path": args.config,
            "checkpoint": str(ckpt_path),
            "event_pretrain_task": args.event_pretrain_task,
            "model_name": model_name,
            "split_root": str(split_root),
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "layer_decay": args.layer_decay,
                "freeze_base": args.freeze_base,
                "unfreeze_after": args.unfreeze_after,
                "patience": args.patience,
                "min_samples": args.min_samples,
                "folds": args.folds,
                "device": args.device,
                "devices": resolved_devices,
                "wait_for_idle_gpus": args.wait_for_idle_gpus,
                "idle_min_memory_gb": args.idle_min_memory_gb,
                "idle_max_utilization": args.idle_max_utilization,
                "idle_poll_seconds": args.idle_poll_seconds,
            },
            "model_config": model_config,
        }
        save_json(run_root / "_resolved_config.json", run_config)

    prepared = 0
    for spec in target_specs:
        target = spec["target"]
        mode = spec["mode"]
        task_df, label_map = build_task_frame(
            labels_df=labels_df,
            spec=spec,
            min_samples=args.min_samples,
        )
        if task_df is None:
            print(f"[SKIP] {target}: valid sample count < {args.min_samples} or only one class.")
            continue

        task_name = str(spec.get("task_name") or build_task_name(target, mode))
        split_payload = ensure_target_split(
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
        split_manifest_path = (
            split_root
            / dataset_key
            / task_name
            / f"kfold_{args.n_splits}_seed_{args.seed}"
            / "split_manifest.json"
        )
        print(
            f"[TARGET_SPLIT:{split_payload.get('_status', 'ok').upper()}] "
            f"{dataset_key}/{task_name} -> {split_payload['num_samples']} samples, "
            f"file={split_manifest_path}"
        )
        prepared += 1

        split_task_root = split_manifest_path.parent
        save_json(
            split_task_root / "target_info.json",
            {
                "dataset": args.dataset,
                "dataset_key": dataset_key,
                "dataset_variant": normalize_dataset_variant(args.dataset_variant),
                "sample_filters": args.sample_filters,
                "target": target,
                "task_name": task_name,
                "task_type": mode,
                "num_samples": int(len(task_df)),
                "num_classes": resolve_num_classes(label_map),
                "all_samples_csv": split_payload["all_samples_csv"],
                "target_spec": spec,
            },
        )
        if label_map is not None:
            save_json(
                split_task_root / "label_mapping.json",
                {
                    "target": target,
                    "mapping": {str(key): int(value) for key, value in label_map.items()},
                },
            )

        if args.prepare_only:
            continue

        task_bundles.append(
            {
                "dataset": dataset_key,
                "target": target,
                "mode": mode,
                "task_name": task_name,
                "task_df": task_df,
                "label_map": label_map,
                "split_payload": split_payload,
                "split_manifest_path": str(split_manifest_path),
                "spec": spec,
            }
        )

    if not args.prepare_only and run_root is not None:
        resolved_devices = normalize_device_list(args.devices, args.device)
        runtime_config = {
            "dataset": dataset_key,
            "base_dataset": args.dataset,
            "dataset_variant": normalize_dataset_variant(args.dataset_variant),
            "sample_filters": args.sample_filters,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "layer_decay": args.layer_decay,
            "freeze_base": args.freeze_base,
            "unfreeze_after": args.unfreeze_after,
            "patience": args.patience,
            "disable_compile": args.disable_compile,
            "selected_folds": selected_folds,
        }

        if len(resolved_devices) > 1:
            all_task_summaries, all_fold_metrics = run_parallel_task_bundles(
                task_bundles=task_bundles,
                devices=resolved_devices,
                args=args,
                run_root=run_root,
                pre_bundle=pre_bundle,
                model_config=model_config,
                ckpt_path=ckpt_path,
            )
        else:
            device_str = resolved_devices[0]
            if args.wait_for_idle_gpus and parse_cuda_device_index(device_str) is not None:
                while True:
                    idle, status = is_device_idle(
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
                    task_bundle=task_bundle,
                    run_root=run_root,
                    pre_bundle=pre_bundle,
                    model_config=model_config,
                    ckpt_path=ckpt_path,
                    runtime_config=runtime_config,
                    device_str=device_str,
                )
                collect_aggregate_rows(result, all_task_summaries, all_fold_metrics)

        if not args.skip_run_aggregates:
            write_run_aggregates(
                run_root=run_root,
                all_task_summaries=all_task_summaries,
                all_fold_metrics=all_fold_metrics,
            )

    if args.prepare_only:
        print(f"[PREPARE_DONE] prepared {prepared} target-specific split datasets under {split_root / args.dataset}")


if __name__ == "__main__":
    main()
