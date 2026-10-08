from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Export lab CV metrics into Excel workbooks. "
            "One workbook is created per dataset with sheets for regression, "
            "classification, and fold-level details."
        )
    )
    parser.add_argument(
        "--base-root",
        type=str,
        default=str(REPO_ROOT / "outputs" / "CGMfoundation_lab"),
        help="Root directory containing model_name/dataset run outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "outputs" / "CGMfoundation_lab_excel"),
        help="Directory where Excel workbooks will be written.",
    )
    parser.add_argument(
        "--run-roots",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit run_root directories. If omitted, the script scans base-root.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional dataset filter, e.g. zhao hpp.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def discover_run_roots(base_root: Path) -> list[Path]:
    return sorted({path.parent for path in base_root.rglob("_resolved_config.json")})


def list_task_dirs(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(
        path
        for path in run_root.iterdir()
        if path.is_dir() and (path / "task_summary.json").exists()
    )


def available_folds_for_task(task_dir: Path, task_summary: dict[str, Any]) -> list[int]:
    raw = task_summary.get("available_folds")
    if isinstance(raw, list) and raw:
        folds = []
        for item in raw:
            try:
                folds.append(int(item))
            except Exception:
                continue
        if folds:
            return sorted(set(folds))

    folds = []
    for path in task_dir.glob("fold_*"):
        if not path.is_dir():
            continue
        try:
            folds.append(int(path.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(folds))


def collect_fold_rows(
    run_root: Path,
    dataset: str,
    model_name: str,
    task_dir: Path,
) -> list[dict[str, Any]]:
    task_summary = load_json(task_dir / "task_summary.json")
    task_name = str(task_dir.name)
    target = str(task_summary.get("target", task_name))
    task_type = str(task_summary.get("task_type", ""))

    rows = []
    for fold_idx in available_folds_for_task(task_dir, task_summary):
        fold_dir = task_dir / f"fold_{fold_idx}"
        membership_path = fold_dir / "fold_membership.json"
        metrics_path = fold_dir / "best_metrics.json"
        eval_summary_path = fold_dir / "best_eval_summary.json"

        if not membership_path.exists() or not metrics_path.exists():
            continue

        membership = load_json(membership_path)
        metrics = load_json(metrics_path)
        eval_summary = (
            load_json(eval_summary_path) if eval_summary_path.exists() else {}
        )

        row = {
            "dataset": dataset,
            "model_name": model_name,
            "run_root": str(run_root),
            "task_name": task_name,
            "target": target,
            "task_type": task_type,
            "fold": int(fold_idx),
            "device": membership.get("device", task_summary.get("device")),
            "train_size": membership.get("train_size"),
            "val_size": membership.get("val_size"),
            "best_epoch": eval_summary.get("epoch"),
            "best_val_loss": eval_summary.get("val_loss"),
        }
        for key, value in metrics.items():
            if is_number(value):
                row[key] = float(value)

        raw_metrics = eval_summary.get("metrics_raw_scale")
        if isinstance(raw_metrics, dict):
            for key, value in raw_metrics.items():
                if is_number(value):
                    row[f"raw_{key}"] = float(value)
        rows.append(row)
    return rows


def collect_task_summary_row(
    run_root: Path,
    dataset: str,
    model_name: str,
    task_dir: Path,
) -> dict[str, Any] | None:
    task_summary_path = task_dir / "task_summary.json"
    cv_summary_path = task_dir / "cv_summary.json"
    if not task_summary_path.exists() or not cv_summary_path.exists():
        return None

    task_summary = load_json(task_summary_path)
    cv_summary = load_json(cv_summary_path)
    row = {
        "dataset": dataset,
        "model_name": model_name,
        "run_root": str(run_root),
        "task_name": str(task_dir.name),
        "target": task_summary.get("target"),
        "task_type": task_summary.get("task_type"),
        "device": task_summary.get("device"),
        "num_samples": task_summary.get("num_samples"),
        "num_folds": len(available_folds_for_task(task_dir, task_summary)),
    }
    for key, value in cv_summary.items():
        if is_number(value):
            row[key] = float(value)
    return row


def collect_run_rows(
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    resolved_config = load_json(run_root / "_resolved_config.json")
    dataset = str(resolved_config.get("dataset") or run_root.name)
    model_name = str(resolved_config.get("model_name") or run_root.parent.name)

    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for task_dir in list_task_dirs(run_root):
        task_summary_row = collect_task_summary_row(
            run_root, dataset, model_name, task_dir
        )
        if task_summary_row is not None:
            summary_rows.append(task_summary_row)
        fold_rows.extend(collect_fold_rows(run_root, dataset, model_name, task_dir))

    run_meta = {
        "dataset": dataset,
        "model_name": model_name,
        "run_root": str(run_root),
    }
    return run_meta, summary_rows, fold_rows


def flatten_group_stats(
    frame: pd.DataFrame, metric_columns: list[str], key_columns: list[str]
) -> pd.DataFrame:
    if frame.empty or not metric_columns:
        return pd.DataFrame(columns=key_columns)

    aggregated = (
        frame.groupby(key_columns, dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    flattened_columns = []
    for column in aggregated.columns:
        if not isinstance(column, tuple):
            flattened_columns.append(str(column))
            continue
        base, stat = column
        if stat == "":
            flattened_columns.append(str(base))
        elif stat == "mean":
            flattened_columns.append(str(base))
        else:
            flattened_columns.append(f"{base}_{stat}")
    aggregated.columns = flattened_columns
    return aggregated


def merge_regression_raw_summary(
    reg_summary_df: pd.DataFrame,
    reg_fold_df: pd.DataFrame,
) -> pd.DataFrame:
    if reg_summary_df.empty or reg_fold_df.empty:
        return reg_summary_df

    key_columns = [
        "dataset",
        "model_name",
        "run_root",
        "task_name",
        "target",
        "task_type",
    ]
    raw_metric_columns = sorted(
        column for column in reg_fold_df.columns if column.startswith("raw_")
    )
    if not raw_metric_columns:
        return reg_summary_df

    raw_summary_df = flatten_group_stats(reg_fold_df, raw_metric_columns, key_columns)
    return reg_summary_df.merge(raw_summary_df, on=key_columns, how="left")


def reorder_columns(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = [column for column in preferred if column in frame.columns]
    ordered.extend(column for column in frame.columns if column not in ordered)
    return frame.loc[:, ordered]


def build_dataset_frames(
    dataset_name: str,
    summary_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["model_name", "task_type", "task_name"],
            kind="stable",
        ).reset_index(drop=True)
    if not fold_df.empty:
        fold_df = fold_df.sort_values(
            by=["model_name", "task_type", "task_name", "fold"],
            kind="stable",
        ).reset_index(drop=True)

    reg_summary_df = (
        summary_df[summary_df.get("task_type") == "reg"].copy()
        if not summary_df.empty
        else pd.DataFrame()
    )
    cls_summary_df = (
        summary_df[summary_df.get("task_type") == "cls"].copy()
        if not summary_df.empty
        else pd.DataFrame()
    )
    reg_fold_df = (
        fold_df[fold_df.get("task_type") == "reg"].copy()
        if not fold_df.empty
        else pd.DataFrame()
    )

    reg_summary_df = merge_regression_raw_summary(reg_summary_df, reg_fold_df)

    reg_summary_df = reorder_columns(
        reg_summary_df,
        [
            "dataset",
            "model_name",
            "task_name",
            "target",
            "device",
            "num_samples",
            "num_folds",
            "run_root",
            "MAE",
            "MAE_std",
            "RMSE",
            "RMSE_std",
            "MAPE",
            "MAPE_std",
            "R2",
            "R2_std",
            "Pearson_r",
            "Pearson_r_std",
            "Pearson_p",
            "Pearson_p_std",
            "Spearman_r",
            "Spearman_r_std",
            "raw_MAE",
            "raw_MAE_std",
            "raw_RMSE",
            "raw_RMSE_std",
            "raw_MAPE",
            "raw_MAPE_std",
            "raw_R2",
            "raw_R2_std",
            "raw_Pearson_r",
            "raw_Pearson_r_std",
            "raw_Pearson_p",
            "raw_Pearson_p_std",
            "raw_Spearman_r",
            "raw_Spearman_r_std",
        ],
    )
    cls_summary_df = reorder_columns(
        cls_summary_df,
        [
            "dataset",
            "model_name",
            "task_name",
            "target",
            "device",
            "num_samples",
            "num_folds",
            "run_root",
            "Acc",
            "Acc_std",
            "F1",
            "F1_std",
            "MCC",
            "MCC_std",
            "AUC",
            "AUC_std",
        ],
    )
    fold_df = reorder_columns(
        fold_df,
        [
            "dataset",
            "model_name",
            "task_name",
            "target",
            "task_type",
            "fold",
            "device",
            "train_size",
            "val_size",
            "best_epoch",
            "best_val_loss",
            "run_root",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
            "Pearson_r",
            "Pearson_p",
            "Spearman_r",
            "raw_MAE",
            "raw_RMSE",
            "raw_MAPE",
            "raw_R2",
            "raw_Pearson_r",
            "raw_Pearson_p",
            "raw_Spearman_r",
            "Acc",
            "F1",
            "MCC",
            "AUC",
        ],
    )

    return reg_summary_df, cls_summary_df, fold_df


def choose_excel_engine() -> str:
    try:
        import openpyxl  # noqa: F401

        return "openpyxl"
    except Exception:
        try:
            import xlsxwriter  # noqa: F401

            return "xlsxwriter"
        except Exception as exc:
            raise RuntimeError(
                "Neither openpyxl nor xlsxwriter is available. Install one of them first."
            ) from exc


def fit_worksheet(
    writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame, engine: str
):
    worksheet = writer.sheets[sheet_name]
    if engine == "openpyxl":
        worksheet.freeze_panes = "A2"
        for col_idx, column in enumerate(frame.columns, start=1):
            width = max(
                len(str(column)),
                frame[column].astype(str).map(len).max() if not frame.empty else 0,
            )
            width = min(max(width + 2, 10), 40)
            worksheet.column_dimensions[
                worksheet.cell(row=1, column=col_idx).column_letter
            ].width = width
    else:
        worksheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(frame.columns):
            width = max(
                len(str(column)),
                frame[column].astype(str).map(len).max() if not frame.empty else 0,
            )
            width = min(max(width + 2, 10), 40)
            worksheet.set_column(col_idx, col_idx, width)


def write_dataset_workbook(
    dataset_name: str,
    reg_summary_df: pd.DataFrame,
    cls_summary_df: pd.DataFrame,
    fold_df: pd.DataFrame,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{dataset_name}_metrics.xlsx"
    engine = choose_excel_engine()

    with pd.ExcelWriter(output_path, engine=engine) as writer:
        reg_summary_df.to_excel(writer, sheet_name="回归指标", index=False)
        cls_summary_df.to_excel(writer, sheet_name="分类指标", index=False)
        fold_df.to_excel(writer, sheet_name="五折明细", index=False)

        fit_worksheet(writer, "回归指标", reg_summary_df, engine)
        fit_worksheet(writer, "分类指标", cls_summary_df, engine)
        fit_worksheet(writer, "五折明细", fold_df, engine)

    return output_path


def main():
    args = build_parser().parse_args()
    base_root = Path(args.base_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    dataset_filter = {item.strip() for item in args.datasets or [] if item.strip()}

    if args.run_roots:
        run_roots = [Path(path).expanduser().resolve() for path in args.run_roots]
    else:
        if not base_root.exists():
            raise FileNotFoundError(f"Base root does not exist: {base_root}")
        run_roots = discover_run_roots(base_root)

    if not run_roots:
        raise RuntimeError(
            "No run_root directories were found. "
            "Expected directories containing _resolved_config.json."
        )

    dataset_to_summary_rows: dict[str, list[dict[str, Any]]] = {}
    dataset_to_fold_rows: dict[str, list[dict[str, Any]]] = {}

    for run_root in run_roots:
        resolved_config_path = run_root / "_resolved_config.json"
        if not resolved_config_path.exists():
            print(f"[SKIP] missing _resolved_config.json: {run_root}")
            continue

        run_meta, summary_rows, fold_rows = collect_run_rows(run_root)
        dataset_name = str(run_meta["dataset"])
        if dataset_filter and dataset_name not in dataset_filter:
            continue

        dataset_to_summary_rows.setdefault(dataset_name, []).extend(summary_rows)
        dataset_to_fold_rows.setdefault(dataset_name, []).extend(fold_rows)
        print(
            f"[COLLECTED] dataset={dataset_name} model={run_meta['model_name']} "
            f"tasks={len(summary_rows)} folds={len(fold_rows)}"
        )

    if not dataset_to_summary_rows and not dataset_to_fold_rows:
        raise RuntimeError("No metric rows were collected from the selected run roots.")

    for dataset_name in sorted(
        set(dataset_to_summary_rows) | set(dataset_to_fold_rows)
    ):
        reg_summary_df, cls_summary_df, fold_df = build_dataset_frames(
            dataset_name=dataset_name,
            summary_rows=dataset_to_summary_rows.get(dataset_name, []),
            fold_rows=dataset_to_fold_rows.get(dataset_name, []),
        )
        output_path = write_dataset_workbook(
            dataset_name=dataset_name,
            reg_summary_df=reg_summary_df,
            cls_summary_df=cls_summary_df,
            fold_df=fold_df,
            out_dir=out_dir,
        )
        print(
            f"[EXPORTED] dataset={dataset_name} "
            f"reg={len(reg_summary_df)} cls={len(cls_summary_df)} folds={len(fold_df)} "
            f"file={output_path}"
        )


if __name__ == "__main__":
    main()

