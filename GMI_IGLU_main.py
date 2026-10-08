import argparse
import importlib
import json
import math
import queue
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from LabelPredictionFinetune import lab_cv_runner as runner

CONFIG_PATH_KEYS = {
    "out_root",
    "feature_cache_root",
}

CONFIG_SECTIONS = {
    "feature": {
        "feature_set",
        "model_name",
        "freq_minutes",
        "glucose_unit",
        "feature_cache_root",
        "force_recompute_features",
    },
    "split": {
        "n_splits",
        "seed",
        "min_samples",
        "folds",
    },
    "runtime": {
        "device",
        "devices",
        "out_root",
        "prepare_only",
        "prepare_all_targets",
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
        "out_root",
        "prepare_only",
        "prepare_all_targets",
        "save_fold_predictions",
        "wait_for_idle_gpus",
        "idle_min_memory_gb",
        "idle_max_utilization",
        "idle_poll_seconds",
        "skip_run_aggregates",
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
GMI_FEATURE_NAMES = ["gmi"]
IGLU_CORE_FEATURE_NAMES = [
    "mean_glu",
    "median_glu",
    "sd_glu",
    "cv_glu",
    "iqr_glu",
    "range_glu",
    "j_index",
    "lbgi",
    "hbgi",
    "tir_70_180",
    "tbr_lt_70",
    "tbr_lt_54",
    "tar_gt_180",
    "tar_gt_250",
    "active_percent",
]
IGLU_EXTRA_FEATURE_NAMES = [
    "adrr",
    "auc",
    "cogi",
    "conga",
    "ea1c",
    "grade",
    "grade_eugly",
    "grade_hyper",
    "grade_hypo",
    "gri",
    "gvp",
    "hyper_index",
    "hypo_index",
    "igc",
    "m_value",
    "mad_glu",
    "mag",
    "mage",
    "modd",
    "pgs",
    "sd_roc",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Build GMI / IGLU-style CGM summary features and run ridge-based "
            "downstream prediction on prepared target CV splits."
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
    parser.add_argument(
        "--feature_set",
        type=str,
        default="iglu",
        choices=["gmi", "iglu", "iglu_all", "gmi_iglu"],
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
        help="Optional list of CUDA devices used as worker slots.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "gmi_iglu_lab_ridge"),
    )
    parser.add_argument("--feature_cache_root", type=str, default=None)
    parser.add_argument("--freq_minutes", type=int, default=5)
    parser.add_argument(
        "--glucose_unit",
        type=str,
        default="auto",
        choices=["auto", "mmol_l", "mg_dl"],
    )
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
    parser.add_argument("--force_recompute_features", action="store_true")
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
        raise ValueError(
            "dataset is required. Pass --dataset or provide it in --config."
        )
    return args


def infer_model_name(args) -> str:
    if args.model_name:
        return str(args.model_name)
    return str(args.feature_set)


def compute_sample_hash(sample_ids: list[str]) -> str:
    joined = "\n".join(str(sample_id) for sample_id in sample_ids).encode("utf-8")
    return (
        runner.hashlib.sha256(joined).hexdigest()
        if hasattr(runner, "hashlib")
        else __import__("hashlib").sha256(joined).hexdigest()
    )


def resolve_target_split_root() -> Path:
    split_root = LAB_TARGET_SPLIT_ROOT.resolve()
    if not split_root.exists():
        raise FileNotFoundError(
            "Target split root was not found. Expected prepared splits under: "
            f"{split_root}"
        )
    return split_root


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
    sample_hash = (
        __import__("hashlib").sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    )

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
    }


def try_import_iglu_python():
    try:
        import iglu_python as iglu  # type: ignore

        return "iglu_python", iglu
    except Exception:
        return "builtin", None


def build_glucose_series(
    values: np.ndarray,
    start_timestamp_iso: str,
    freq_minutes: int,
    glucose_scale: float,
) -> tuple[pd.Series, int]:
    values = np.asarray(values, dtype=np.float64)
    total_len = int(values.shape[0])
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return pd.Series(dtype=np.float64), total_len

    glucose_mgdl = values[valid_mask] * float(glucose_scale)
    date_index = pd.date_range(
        start=pd.Timestamp(start_timestamp_iso),
        periods=total_len,
        freq=f"{int(freq_minutes)}min",
    )
    return pd.Series(
        glucose_mgdl.astype(np.float64), index=date_index[valid_mask]
    ), total_len


def safe_std(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def compute_gmi_from_values(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(3.31 + 0.02392 * float(np.mean(values)))


def compute_j_index(mean_glu: float, sd_glu: float) -> float:
    return float(0.001 * (mean_glu + sd_glu) ** 2)


def compute_bg_risk_indices(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    clipped = np.clip(values.astype(np.float64), 1e-3, None)
    transform = 1.509 * ((np.log(clipped) ** 1.084) - 5.381)
    risk = 10.0 * (transform**2)
    lbgi = float(np.mean(np.where(transform < 0, risk, 0.0)))
    hbgi = float(np.mean(np.where(transform > 0, risk, 0.0)))
    return lbgi, hbgi


def compute_percent(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean() * 100.0)


def maybe_call_iglu_metric(
    iglu_module,
    fn_name: str,
    series: pd.Series,
    fallback: float,
    kwargs: dict[str, Any] | None = None,
) -> float:
    if iglu_module is None:
        return float(fallback)
    fn = resolve_iglu_metric_callable(iglu_module, fn_name)
    if fn is None:
        return float(fallback)
    call_kwargs = kwargs or {}
    try:
        try:
            value = fn(series, **call_kwargs)
        except TypeError:
            value = fn(series)
        if isinstance(value, pd.DataFrame):
            if value.shape[1] >= 2:
                return float(value.iloc[0, 1])
            if value.shape[1] == 1:
                return float(value.iloc[0, 0])
        if isinstance(value, pd.Series):
            if len(value) > 0:
                return float(value.iloc[0])
        if isinstance(value, dict):
            if value:
                return float(next(iter(value.values())))
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return float(value[0])
        return float(value)
    except Exception:
        return float(fallback)


def resolve_iglu_metric_callable(iglu_module, fn_name: str):
    if iglu_module is None:
        return None

    direct_fn = getattr(iglu_module, fn_name, None)
    if callable(direct_fn):
        return direct_fn

    module_name = getattr(iglu_module, "__name__", "")
    if not module_name:
        return None

    try:
        submodule = importlib.import_module(f"{module_name}.{fn_name}")
    except Exception:
        return None

    submodule_fn = getattr(submodule, fn_name, None)
    if callable(submodule_fn):
        return submodule_fn

    for attr_name in dir(submodule):
        if attr_name.startswith("_"):
            continue
        candidate = getattr(submodule, attr_name)
        if callable(candidate):
            return candidate
    return None


def feature_names_for_set(feature_set: str) -> list[str]:
    if feature_set == "gmi":
        return GMI_FEATURE_NAMES.copy()
    if feature_set == "iglu":
        return IGLU_CORE_FEATURE_NAMES.copy()
    if feature_set == "iglu_all":
        return IGLU_CORE_FEATURE_NAMES + IGLU_EXTRA_FEATURE_NAMES
    if feature_set == "gmi_iglu":
        return GMI_FEATURE_NAMES + IGLU_CORE_FEATURE_NAMES
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def compute_feature_row(
    feature_set: str,
    values: np.ndarray,
    start_timestamp_iso: str,
    freq_minutes: int,
    glucose_scale: float,
    iglu_backend,
) -> dict[str, float]:
    series, total_len = build_glucose_series(
        values=values,
        start_timestamp_iso=start_timestamp_iso,
        freq_minutes=freq_minutes,
        glucose_scale=glucose_scale,
    )
    glucose_values = series.to_numpy(dtype=np.float64)
    valid_count = int(glucose_values.size)

    if feature_set == "gmi":
        return {"gmi": compute_gmi_from_values(glucose_values)}

    mean_glu = float(np.mean(glucose_values)) if valid_count else 0.0
    median_glu = float(np.median(glucose_values)) if valid_count else 0.0
    sd_glu = safe_std(glucose_values)
    cv_glu = float(sd_glu / mean_glu * 100.0) if mean_glu > 1e-8 else 0.0
    q1 = float(np.percentile(glucose_values, 25.0)) if valid_count else 0.0
    q3 = float(np.percentile(glucose_values, 75.0)) if valid_count else 0.0
    iqr_glu = q3 - q1
    range_glu = (
        float(np.max(glucose_values) - np.min(glucose_values)) if valid_count else 0.0
    )
    lbgi, hbgi = compute_bg_risk_indices(glucose_values)
    active_percent = float(valid_count / total_len * 100.0) if total_len > 0 else 0.0

    metrics = {
        "mean_glu": maybe_call_iglu_metric(iglu_backend, "mean_glu", series, mean_glu),
        "median_glu": maybe_call_iglu_metric(
            iglu_backend,
            "median_glu",
            series,
            median_glu,
        ),
        "sd_glu": maybe_call_iglu_metric(iglu_backend, "sd_glu", series, sd_glu),
        "cv_glu": maybe_call_iglu_metric(iglu_backend, "cv_glu", series, cv_glu),
        "iqr_glu": maybe_call_iglu_metric(iglu_backend, "iqr_glu", series, iqr_glu),
        "range_glu": maybe_call_iglu_metric(
            iglu_backend,
            "range_glu",
            series,
            range_glu,
        ),
        "j_index": maybe_call_iglu_metric(
            iglu_backend,
            "j_index",
            series,
            compute_j_index(mean_glu, sd_glu),
        ),
        "lbgi": maybe_call_iglu_metric(iglu_backend, "lbgi", series, lbgi),
        "hbgi": maybe_call_iglu_metric(iglu_backend, "hbgi", series, hbgi),
        "tir_70_180": compute_percent(
            (glucose_values >= 70.0) & (glucose_values <= 180.0)
        ),
        "tbr_lt_70": compute_percent(glucose_values < 70.0),
        "tbr_lt_54": compute_percent(glucose_values < 54.0),
        "tar_gt_180": compute_percent(glucose_values > 180.0),
        "tar_gt_250": compute_percent(glucose_values > 250.0),
        "active_percent": active_percent,
    }

    if feature_set == "iglu_all":
        extra_metrics = {
            "adrr": maybe_call_iglu_metric(iglu_backend, "adrr", series, float("nan")),
            "auc": maybe_call_iglu_metric(iglu_backend, "auc", series, float("nan")),
            "cogi": maybe_call_iglu_metric(iglu_backend, "cogi", series, float("nan")),
            "conga": maybe_call_iglu_metric(
                iglu_backend, "conga", series, float("nan")
            ),
            "ea1c": maybe_call_iglu_metric(iglu_backend, "ea1c", series, float("nan")),
            "grade": maybe_call_iglu_metric(
                iglu_backend, "grade", series, float("nan")
            ),
            "grade_eugly": maybe_call_iglu_metric(
                iglu_backend, "grade_eugly", series, float("nan")
            ),
            "grade_hyper": maybe_call_iglu_metric(
                iglu_backend, "grade_hyper", series, float("nan")
            ),
            "grade_hypo": maybe_call_iglu_metric(
                iglu_backend, "grade_hypo", series, float("nan")
            ),
            "gri": maybe_call_iglu_metric(iglu_backend, "gri", series, float("nan")),
            "gvp": maybe_call_iglu_metric(iglu_backend, "gvp", series, float("nan")),
            "hyper_index": maybe_call_iglu_metric(
                iglu_backend, "hyper_index", series, float("nan")
            ),
            "hypo_index": maybe_call_iglu_metric(
                iglu_backend, "hypo_index", series, float("nan")
            ),
            "igc": maybe_call_iglu_metric(iglu_backend, "igc", series, float("nan")),
            "m_value": maybe_call_iglu_metric(
                iglu_backend, "m_value", series, float("nan")
            ),
            "mad_glu": maybe_call_iglu_metric(
                iglu_backend, "mad_glu", series, float("nan")
            ),
            "mag": maybe_call_iglu_metric(iglu_backend, "mag", series, float("nan")),
            "mage": maybe_call_iglu_metric(iglu_backend, "mage", series, float("nan")),
            "modd": maybe_call_iglu_metric(iglu_backend, "modd", series, float("nan")),
            "pgs": maybe_call_iglu_metric(iglu_backend, "pgs", series, float("nan")),
            "sd_roc": maybe_call_iglu_metric(
                iglu_backend, "sd_roc", series, float("nan")
            ),
        }
        return {**metrics, **extra_metrics}

    if feature_set == "gmi_iglu":
        return {"gmi": compute_gmi_from_values(glucose_values), **metrics}
    return metrics


def build_feature_cache_paths(
    args,
    model_name: str,
    dataset_name: str,
) -> tuple[Path, Path]:
    if args.feature_cache_root:
        cache_root = Path(args.feature_cache_root).expanduser().resolve()
    else:
        cache_root = (
            Path(args.out_root).expanduser().resolve()
            / model_name
            / dataset_name
            / "_feature_cache"
        )
    cache_root.mkdir(parents=True, exist_ok=True)

    base_name = f"{dataset_name}_{args.feature_set}"
    return cache_root / f"{base_name}.npz", cache_root / f"{base_name}.json"


def load_cached_features(cache_path: Path):
    with np.load(cache_path, allow_pickle=False) as payload:
        sample_ids = payload["sample_ids"].astype(str).tolist()
        features = payload["features"].astype(np.float32)
        feature_names = payload["feature_names"].astype(str).tolist()
    return sample_ids, features, feature_names


def extract_or_load_features(
    args,
    model_name: str,
    dataset_name: str,
    cgm_bundle: dict[str, Any],
):
    feature_backend_name, iglu_backend = try_import_iglu_python()
    cache_path, meta_path = build_feature_cache_paths(
        args=args,
        model_name=model_name,
        dataset_name=dataset_name,
    )
    feature_names = feature_names_for_set(args.feature_set)
    expected_hash = cgm_bundle["sample_hash"]

    if cache_path.exists() and meta_path.exists() and not args.force_recompute_features:
        cache_meta = load_json(meta_path)
        if (
            cache_meta.get("sample_hash") == expected_hash
            and cache_meta.get("feature_set") == args.feature_set
            and cache_meta.get("model_name") == model_name
            and cache_meta.get("feature_backend") == feature_backend_name
            and cache_meta.get("feature_names") == feature_names
        ):
            sample_ids, features, cached_feature_names = load_cached_features(
                cache_path
            )
            if cached_feature_names == feature_names:
                print(f"[FEATURE_CACHE_HIT] {dataset_name} -> {cache_path}")
                return {
                    "sample_ids": sample_ids,
                    "features": features,
                    "feature_names": feature_names,
                    "cache_path": str(cache_path),
                    "cache_meta_path": str(meta_path),
                    "cache_status": "hit",
                    "feature_backend": feature_backend_name,
                }

    rows = []
    total_samples = len(cgm_bundle["sample_ids"])
    for index, sample_id in enumerate(cgm_bundle["sample_ids"]):
        row = compute_feature_row(
            feature_set=args.feature_set,
            values=cgm_bundle["glucose_values"][index],
            start_timestamp_iso=cgm_bundle["start_timestamps"][index],
            freq_minutes=args.freq_minutes,
            glucose_scale=cgm_bundle["glucose_scale"],
            iglu_backend=iglu_backend,
        )
        rows.append([float(row[name]) for name in feature_names])
        if (index + 1) % 500 == 0 or (index + 1) == total_samples:
            print(
                f"[FEATURE_PROGRESS] dataset={dataset_name} "
                f"samples={index + 1}/{total_samples}"
            )

    features = np.asarray(rows, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez_compressed(
        cache_path,
        sample_ids=np.asarray(cgm_bundle["sample_ids"], dtype=str),
        features=features,
        feature_names=np.asarray(feature_names, dtype=str),
    )
    runner.save_json(
        meta_path,
        {
            "model_name": model_name,
            "dataset": dataset_name,
            "feature_set": args.feature_set,
            "feature_backend": feature_backend_name,
            "feature_names": feature_names,
            "num_samples": int(len(cgm_bundle["sample_ids"])),
            "feature_dim": int(features.shape[1]),
            "sample_hash": expected_hash,
            "resolved_glucose_unit": cgm_bundle["resolved_glucose_unit"],
            "glucose_scale": cgm_bundle["glucose_scale"],
            "freq_minutes": int(args.freq_minutes),
        },
    )
    print(
        f"[FEATURE_EXTRACT_DONE] {dataset_name} samples={len(cgm_bundle['sample_ids'])} "
        f"dim={features.shape[1]} feature_set={args.feature_set} backend={feature_backend_name}"
    )
    return {
        "sample_ids": cgm_bundle["sample_ids"],
        "features": features,
        "feature_names": feature_names,
        "cache_path": str(cache_path),
        "cache_meta_path": str(meta_path),
        "cache_status": "saved",
        "feature_backend": feature_backend_name,
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
            (
                "ridge",
                RidgeClassifierCV(alphas=np.asarray(alpha_values, dtype=np.float64)),
            ),
        ]
    )


def run_task_bundle(
    args,
    run_root: Path,
    feature_bundle: dict[str, Any],
    task_name: str,
    target: str,
    mode: str,
    task_df: pd.DataFrame,
    label_map: dict[str, int] | None,
    split_bundle: dict[str, Any],
):
    sample_ids = feature_bundle["sample_ids"]
    features = feature_bundle["features"]
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

        x_train = features[[id_to_index[sample_id] for sample_id in train_ids]]
        x_val = features[[id_to_index[sample_id] for sample_id in val_ids]]

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
    feature_cache_path_str: str,
    args_payload: dict[str, Any],
):
    try:
        runner.set_gpu_flags()
        args_device = torch.device(device_str)
        if args_device.type == "cuda":
            torch.cuda.set_device(args_device)

        worker_args = argparse.Namespace(**args_payload)
        worker_args.device = device_str

        sample_ids, features, feature_names = load_cached_features(
            Path(feature_cache_path_str)
        )
        feature_bundle = {
            "sample_ids": sample_ids,
            "features": features,
            "feature_names": feature_names,
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
                feature_bundle=feature_bundle,
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
    feature_cache_path: str,
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
                    str(feature_cache_path),
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
        "Prediction requires --targets, or pass --prepare_only to cache features only."
    )


def run_from_args(args):
    runner.set_gpu_flags()
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
    split_root = resolve_target_split_root()
    resolved_devices = runner.normalize_device_list(args.devices, args.device)

    cgm_bundle = prepare_cgm_bundle(cgm_df, args.glucose_unit)
    feature_bundle = extract_or_load_features(
        args=args,
        model_name=model_name,
        dataset_name=dataset_key,
        cgm_bundle=cgm_bundle,
    )

    resolved_config = {
        "dataset": args.dataset,
        "dataset_key": dataset_key,
        "dataset_variant": getattr(args, "dataset_variant", None),
        "sample_filters": getattr(args, "sample_filters", None),
        "targets": target_specs,
        "model_name": model_name,
        "feature_set": args.feature_set,
        "devices": resolved_devices,
        "alpha_values": list(float(value) for value in args.alpha_values),
        "split_root": str(split_root),
        "split_policy": "target_only",
        "runtime": {
            "wait_for_idle_gpus": bool(args.wait_for_idle_gpus),
            "idle_min_memory_gb": float(args.idle_min_memory_gb),
            "idle_max_utilization": float(args.idle_max_utilization),
            "idle_poll_seconds": float(args.idle_poll_seconds),
        },
        "feature_cache": {
            "cache_path": feature_bundle["cache_path"],
            "cache_meta_path": feature_bundle["cache_meta_path"],
            "cache_status": feature_bundle["cache_status"],
            "num_samples": int(len(feature_bundle["sample_ids"])),
            "feature_dim": int(feature_bundle["features"].shape[1]),
            "feature_names": feature_bundle["feature_names"],
            "feature_backend": feature_bundle["feature_backend"],
            "resolved_glucose_unit": cgm_bundle["resolved_glucose_unit"],
            "glucose_scale": cgm_bundle["glucose_scale"],
        },
    }
    runner.save_json(run_root / "_resolved_config.json", resolved_config)

    if args.prepare_only:
        print(
            f"[PREPARE_DONE] dataset={args.dataset} "
            f"cache={feature_bundle['cache_path']}"
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
            "feature_cache_path": feature_bundle["cache_path"],
            "feature_cache_meta_path": feature_bundle["cache_meta_path"],
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
                    "mapping": {
                        str(key): int(value) for key, value in label_map.items()
                    },
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
            feature_cache_path=feature_bundle["cache_path"],
        )
    else:
        device_str = resolved_devices[0]
        args.device = device_str
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
                feature_bundle=feature_bundle,
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
        "feature_cache_path": feature_bundle["cache_path"],
        "feature_cache_meta_path": feature_bundle["cache_meta_path"],
    }


def main():
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
