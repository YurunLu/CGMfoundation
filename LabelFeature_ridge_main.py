import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
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
    "split_root",
    "out_root",
}

CONFIG_SECTIONS = {
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
        "out_root",
        "save_fold_predictions",
        "skip_run_aggregates",
    },
    "training": {
        "device",
        "out_root",
        "save_fold_predictions",
        "skip_run_aggregates",
    },
    "feature": {
        "feature_columns",
        "feature_name",
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
            "Run ridge-based downstream prediction on label-derived feature columns "
            "using prepared or on-the-fly target CV splits."
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
    parser.add_argument(
        "--feature_columns",
        type=str,
        nargs="*",
        default=None,
        help="One or more label columns used as predictor features.",
    )
    parser.add_argument(
        "--feature_name",
        type=str,
        default=None,
        help="Optional short feature-set name for outputs.",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--split_root",
        type=str,
        default=str(REPO_ROOT / "splits" / "lab_target_cv"),
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "label_feature_lab_ridge"),
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
    parser.add_argument("--force_regenerate_splits", action="store_true")
    parser.add_argument("--save_fold_predictions", action="store_true")
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
    if args.feature_columns is None:
        args.feature_columns = []
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
    if not args.feature_columns:
        raise ValueError(
            "feature_columns is required. Pass --feature_columns or provide it in --config."
        )
    return args


def infer_model_name(args) -> str:
    if args.model_name:
        return str(args.model_name)
    if args.feature_name:
        return f"label_ridge_{runner.sanitize_name(args.feature_name).lower()}"
    if len(args.feature_columns) == 1:
        return f"label_ridge_{runner.sanitize_name(args.feature_columns[0]).lower()}"
    return "label_feature_lab_ridge"


def resolve_target_specs(args, labels_df: pd.DataFrame) -> list[dict[str, Any]]:
    if args.targets:
        return runner.parse_target_specs(args.targets, labels_df)
    if args.prepare_all_targets:
        return runner.parse_target_specs(labels_df.columns.tolist(), labels_df)
    raise ValueError("Prediction requires --targets.")


def build_feature_frame(
    labels_df: pd.DataFrame,
    sample_ids: list[str],
    feature_columns: list[str],
) -> pd.DataFrame:
    for column in feature_columns:
        if column not in labels_df.columns:
            raise KeyError(f"Feature column not found: {column}")

    feature_df = labels_df.loc[sample_ids, feature_columns].copy()
    for column in feature_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
    before = len(feature_df)
    feature_df = feature_df.dropna(subset=feature_columns)
    after = len(feature_df)
    print(
        f"[FEATURE_FILTER] columns={feature_columns} kept={after}/{before}"
    )
    return feature_df


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
        f"[TARGET_SPLIT:{status}] {dataset_key}/{task_name} "
        f"-> {len(task_df)} samples"
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
            ("ridge", RidgeClassifierCV(alphas=np.asarray(alpha_values, dtype=np.float64))),
        ]
    )


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
    feature_df: pd.DataFrame,
    task_name: str,
    target: str,
    mode: str,
    task_df: pd.DataFrame,
    label_map: dict[str, int] | None,
    split_bundle: dict[str, Any],
):
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
            if sample_id in labels.index and sample_id in feature_df.index
        ]
        val_ids = [
            sample_id
            for sample_id in fold_info["val_ids"]
            if sample_id in labels.index and sample_id in feature_df.index
        ]

        if len(train_ids) < 2 or len(val_ids) == 0:
            print(
                f"[SKIP_FOLD] {task_name} fold={fold_idx} "
                f"train={len(train_ids)} val={len(val_ids)}"
            )
            continue

        x_train = feature_df.loc[train_ids].to_numpy(dtype=np.float64)
        x_val = feature_df.loc[val_ids].to_numpy(dtype=np.float64)

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
            "feature_columns": list(feature_df.columns),
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
        "feature_columns": ",".join(str(column) for column in feature_df.columns),
    }
    for key, value in summary.items():
        task_summary_row[key] = float(value)
    return {
        "task_summary_row": task_summary_row,
        "fold_metrics_rows": fold_metrics_rows,
    }


def run_from_args(args):
    np.random.seed(args.seed)

    dataset_key = runner.resolve_dataset_key(
        args.dataset,
        dataset_variant=getattr(args, "dataset_variant", None),
        sample_filters=getattr(args, "sample_filters", None),
    )
    _, labels_df = runner.load_dataset_frames(
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

    all_task_summaries: list[dict[str, Any]] = []
    all_fold_metrics: list[dict[str, Any]] = []
    prepared = 0

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

        feature_df = build_feature_frame(
            labels_df=labels_df,
            sample_ids=task_df.index.astype(str).tolist(),
            feature_columns=[str(column) for column in args.feature_columns],
        )
        task_df = task_df.loc[feature_df.index].copy()
        if len(task_df) < args.min_samples:
            print(
                f"[SKIP] dataset={args.dataset} target={target}: "
                f"valid sample count < {args.min_samples} after feature filtering."
            )
            continue
        if mode == "cls" and task_df[target].nunique() < 2:
            print(
                f"[SKIP] dataset={args.dataset} target={target}: "
                "only one class remains after feature filtering."
            )
            continue

        task_name = str(spec.get("task_name") or runner.build_task_name(target, mode))
        split_bundle = ensure_target_split_bundle(
            args=args,
            dataset_key=dataset_key,
            task_name=task_name,
            target=target,
            mode=mode,
            task_df=task_df,
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
                "feature_columns": [str(column) for column in feature_df.columns],
            },
        )

        if args.prepare_only:
            continue

        print(
            f"[TASK_START] dataset={dataset_key} task={task_name} "
            f"split_source={split_bundle['source']}"
        )
        result = run_task_bundle(
            args=args,
            run_root=run_root,
            dataset_key=dataset_key,
            feature_df=feature_df,
            task_name=task_name,
            target=target,
            mode=mode,
            task_df=task_df,
            label_map=label_map,
            split_bundle=split_bundle,
        )
        runner.collect_aggregate_rows(result, all_task_summaries, all_fold_metrics)

    resolved_config = {
        "dataset": args.dataset,
        "dataset_key": dataset_key,
        "dataset_variant": getattr(args, "dataset_variant", None),
        "sample_filters": getattr(args, "sample_filters", None),
        "targets": target_specs,
        "model_name": model_name,
        "device": str(args.device),
        "feature_columns": [str(column) for column in args.feature_columns],
        "feature_name": args.feature_name,
        "alpha_values": list(float(value) for value in args.alpha_values),
        "split_root": str(Path(args.split_root).expanduser().resolve()),
    }
    runner.save_json(run_root / "_resolved_config.json", resolved_config)

    if args.prepare_only:
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
        }

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
    }


def main():
    args = parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
