from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_ORDER = ["CGMfoundation", "Gluformer", "GMI", "IGLU"]
MODEL_COLORS = {
    "CGMfoundation": "#4F81BD",
    "Gluformer": "#2F2FD7",
    "GMI": "#F4C7CF",
    "IGLU": "#E51C23",
    # "CGMfoundation": "#2F2FD7",
    # "Gluformer": "#F4C7CF",
    # "GMI": "#E51C23",
    # "IGLU": "#C2185B",
}

DEFAULT_RUN_ROOTS = [
    REPO_ROOT / "outputs" / "CGMfoundation_lab" / "CGMfoundation_lab_hpp" / "hpp",
    REPO_ROOT
    / "outputs"
    / "CGMfoundation_lab"
    / "CGMfoundation_lab_nicheng"
    / "nicheng",
    REPO_ROOT / "outputs" / "CGMfoundation_lab" / "CGMfoundation_lab_zhao" / "zhao",
    REPO_ROOT / "outputs" / "gluformer_lab_ridge" / "gluformer_lab_ridge_all" / "hpp",
    REPO_ROOT
    / "outputs"
    / "gluformer_lab_ridge"
    / "gluformer_lab_ridge_all"
    / "nicheng",
    REPO_ROOT / "outputs" / "gluformer_lab_ridge" / "gluformer_lab_ridge_all" / "zhao",
    REPO_ROOT / "outputs" / "gmi_iglu_lab_ridge" / "gmi",
    REPO_ROOT / "outputs" / "gmi_iglu_lab_ridge" / "iglu_all",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect lab prediction results from multiple model outputs and export "
            "an Excel workbook with per-task comparison charts."
        )
    )
    parser.add_argument(
        "--run-roots",
        type=str,
        nargs="*",
        default=[str(path) for path in DEFAULT_RUN_ROOTS],
        help="Output roots to scan recursively for cv_summary.json files.",
    )
    parser.add_argument(
        "--out-xlsx",
        type=str,
        default=str(REPO_ROOT / "outputs" / "lab_model_compare.xlsx"),
        help="Target Excel path.",
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def infer_model_name(path: Path) -> str:
    text = str(path).lower().replace("\\", "/")
    if "/cgmfoundation_lab/" in text:
        return "CGMfoundation"
    if "/gluformer_lab_ridge/" in text:
        return "Gluformer"
    if "/gmi_iglu_lab_ridge/gmi/" in text:
        return "GMI"
    if "/gmi_iglu_lab_ridge/iglu_all/" in text or text.endswith(
        "/gmi_iglu_lab_ridge/iglu_all"
    ):
        return "IGLU"
    return path.parent.name or path.name


def infer_dataset_name(
    path: Path, task_summary: dict[str, Any], cv_summary: dict[str, Any]
) -> str:
    for payload in (task_summary, cv_summary):
        dataset = payload.get("dataset")
        if isinstance(dataset, str) and dataset:
            return dataset

    parts = [part for part in path.parts if part]
    for name in ("hpp", "nicheng", "zhao"):
        if name in parts:
            return name
    return path.parent.parent.name


def infer_task_name(
    path: Path, task_summary: dict[str, Any], cv_summary: dict[str, Any]
) -> str:
    for payload in (task_summary, cv_summary):
        task_name = payload.get("task_name")
        if isinstance(task_name, str) and task_name:
            return task_name
    return path.parent.name


def infer_target_name(
    task_name: str, task_summary: dict[str, Any], cv_summary: dict[str, Any]
) -> str:
    for payload in (task_summary, cv_summary):
        target = payload.get("target")
        if isinstance(target, str) and target:
            return target
    return task_name


def infer_task_type(task_summary: dict[str, Any], cv_summary: dict[str, Any]) -> str:
    for payload in (task_summary, cv_summary):
        task_type = payload.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    return ""


def extract_metric_dict(cv_summary: dict[str, Any]) -> dict[str, Any]:
    nested = cv_summary.get("metrics")
    if isinstance(nested, dict):
        return nested
    return cv_summary


def is_regression(task_type: str) -> bool:
    return str(task_type).lower().startswith("reg")


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if pd.isna(result):
        return None
    return result


def collect_rows(run_roots: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for root_str in run_roots:
        root = Path(root_str).expanduser().resolve()
        if not root.exists():
            print(f"[SKIP] missing root: {root}")
            continue

        for cv_path in sorted(root.rglob("cv_summary.json")):
            task_dir = cv_path.parent
            task_summary_path = task_dir / "task_summary.json"
            task_summary = (
                load_json(task_summary_path) if task_summary_path.exists() else {}
            )
            cv_summary = load_json(cv_path)

            dataset = infer_dataset_name(cv_path, task_summary, cv_summary)
            task_name = infer_task_name(cv_path, task_summary, cv_summary)
            target = infer_target_name(task_name, task_summary, cv_summary)
            task_type = infer_task_type(task_summary, cv_summary)
            metric_name = "Pearson_r" if is_regression(task_type) else "AUC"
            metrics = extract_metric_dict(cv_summary)
            metric_value = safe_float(metrics.get(metric_name))
            model_name = infer_model_name(cv_path)

            if metric_value is None:
                continue

            dedup_key = (dataset, task_name, model_name)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            rows.append(
                {
                    "dataset": dataset,
                    "task_name": task_name,
                    "target": target,
                    "task_type": task_type,
                    "model_name": model_name,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "cv_summary_path": str(cv_path),
                }
            )
    return rows


def sanitize_sheet_name(name: str) -> str:
    cleaned = str(name)
    for ch in ("\\", "/", "*", "[", "]", ":", "?"):
        cleaned = cleaned.replace(ch, "_")
    cleaned = cleaned[:31]
    return cleaned or "Sheet1"


def model_sort_key(model_name: str) -> tuple[int, str]:
    if model_name in MODEL_ORDER:
        return (MODEL_ORDER.index(model_name), model_name)
    return (len(MODEL_ORDER), model_name)


def build_pivot(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index=["target", "task_name"],
        columns="model_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.sort_values(["target", "task_name"]).reset_index(drop=True)
    pivot.columns.name = None
    ordered_columns = ["target", "task_name"]
    ordered_columns.extend(column for column in MODEL_ORDER if column in pivot.columns)
    ordered_columns.extend(
        column for column in pivot.columns if column not in ordered_columns
    )
    pivot = pivot.loc[:, ordered_columns]
    return pivot


def add_chart_sheet(
    workbook: Any,
    writer: pd.ExcelWriter,
    pivot: pd.DataFrame,
    data_sheet_name: str,
    chart_sheet_name: str,
    chart_title_prefix: str,
) -> None:
    if pivot.empty:
        return

    chart_sheet = workbook.add_worksheet(chart_sheet_name)
    writer.sheets[chart_sheet_name] = chart_sheet

    model_columns = list(pivot.columns[2:])
    if not model_columns:
        return

    row_cursor = 0
    col_positions = [0, 8]
    for row_idx in range(len(pivot)):
        excel_row = row_idx + 1
        target = str(pivot.iloc[row_idx]["target"])
        point_styles = []
        for model_name in model_columns:
            color = MODEL_COLORS.get(model_name, "#9C9C9C")
            point_styles.append(
                {
                    "fill": {"color": color},
                    "border": {"color": color},
                }
            )
        chart = workbook.add_chart({"type": "column"})
        chart.add_series(
            {
                "name": target,
                "categories": [data_sheet_name, 0, 2, 0, 1 + len(model_columns)],
                "values": [
                    data_sheet_name,
                    excel_row,
                    2,
                    excel_row,
                    1 + len(model_columns),
                ],
                "data_labels": {"value": True, "num_format": "0.000"},
                "points": point_styles,
                "gap": 260,
            }
        )
        chart.set_title({"name": f"{chart_title_prefix}: {target}"})
        chart.set_legend({"none": True})
        chart.set_chartarea({"border": {"none": True}})
        chart.set_plotarea({"border": {"none": True}, "fill": {"color": "#FFFFFF"}})
        chart.set_y_axis(
            {
                "major_gridlines": {"visible": False},
                "minor_gridlines": {"visible": False},
            }
        )
        chart.set_size({"width": 500, "height": 420})

        insert_col = col_positions[row_idx % 2]
        insert_row = row_cursor
        chart_sheet.insert_chart(insert_row, insert_col, chart)
        if row_idx % 2 == 1:
            row_cursor += 24


def export_excel(rows: list[dict[str, Any]], out_path: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No valid task metrics were found.")

    frame = frame.drop_duplicates(
        subset=["dataset", "task_name", "model_name"], keep="last"
    )
    frame["model_sort"] = frame["model_name"].map(lambda x: model_sort_key(str(x)))
    frame = frame.sort_values(
        ["dataset", "task_type", "target", "model_sort", "model_name"]
    ).reset_index(drop=True)
    frame = frame.drop(columns=["model_sort"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            workbook = writer.book
            summary_sheet = "all_metrics"
            frame.to_excel(writer, sheet_name=summary_sheet, index=False)
            writer.sheets[summary_sheet].freeze_panes(1, 0)

            for dataset in sorted(frame["dataset"].dropna().unique()):
                dataset_df = frame[frame["dataset"] == dataset].copy()
                reg_df = dataset_df[dataset_df["metric_name"] == "Pearson_r"].copy()
                cls_df = dataset_df[dataset_df["metric_name"] == "AUC"].copy()

                reg_pivot = build_pivot(reg_df)
                cls_pivot = build_pivot(cls_df)

                if not reg_pivot.empty:
                    reg_sheet = sanitize_sheet_name(f"{dataset}_reg")
                    reg_pivot.to_excel(writer, sheet_name=reg_sheet, index=False)
                    writer.sheets[reg_sheet].freeze_panes(1, 0)
                    add_chart_sheet(
                        workbook=workbook,
                        writer=writer,
                        pivot=reg_pivot,
                        data_sheet_name=reg_sheet,
                        chart_sheet_name=sanitize_sheet_name(f"{dataset}_reg_chart"),
                        chart_title_prefix=f"{dataset} Pearson_r",
                    )

                if not cls_pivot.empty:
                    cls_sheet = sanitize_sheet_name(f"{dataset}_cls")
                    cls_pivot.to_excel(writer, sheet_name=cls_sheet, index=False)
                    writer.sheets[cls_sheet].freeze_panes(1, 0)
                    add_chart_sheet(
                        workbook=workbook,
                        writer=writer,
                        pivot=cls_pivot,
                        data_sheet_name=cls_sheet,
                        chart_sheet_name=sanitize_sheet_name(f"{dataset}_cls_chart"),
                        chart_title_prefix=f"{dataset} AUC",
                    )
    except ImportError as exc:
        raise ImportError(
            "Please install xlsxwriter first: pip install XlsxWriter"
        ) from exc


def main() -> None:
    args = build_parser().parse_args()
    rows = collect_rows(args.run_roots)
    export_excel(rows, Path(args.out_xlsx).expanduser().resolve())
    print(f"[OK] Excel written to: {Path(args.out_xlsx).expanduser().resolve()}")
    print(f"[OK] Collected task rows: {len(rows)}")


if __name__ == "__main__":
    main()
