from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import EventFinetune_main as event_main
from EventFinetune.Event_Finetune_data_utils import normalize_entity_key
from EventFinetune.Event_Finetune_model import EventFinetuneModel

ROW_LIMIT_PER_SHEET = 900000
TIME_INTERVAL_MINUTES = 5
EXCEL_MAX_DATA_ROWS = 1_048_575
EVENT_COMPARE_BLOCK_ROWS = 3


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute pre-finetune and post-finetune forecasts for event "
            "fine-tuning runs and export Excel workbooks under the original "
            "output directories."
        )
    )
    parser.add_argument(
        "--base-root",
        type=str,
        default=str(REPO_ROOT / "data_event" / "event_finetune_outputs"),
        help="Root directory containing event fine-tuning outputs.",
    )
    parser.add_argument(
        "--summary-paths",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit *_summary.json paths. If omitted, scan --base-root.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional parent-directory filter, e.g. HPP zhao_full.",
    )
    parser.add_argument(
        "--run-tags",
        type=str,
        nargs="*",
        default=None,
        help="Optional run_tag substring filter.",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default=None,
        help="Override the exported split. Defaults to the run's prediction split.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        choices=["auto", "best", "final"],
        default="auto",
        help="Checkpoint used for post-finetune predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device used during export.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Optional workbook filename. Defaults to "
            "<run_tag>_prepost_prediction_<split>.xlsx."
        ),
    )
    parser.add_argument(
        "--include-zero-event-rows",
        action="store_true",
        help=(
            "By default event_detail only keeps rows where at least one event "
            "value is non-zero. Enable this flag to export all event-window rows."
        ),
    )
    return parser


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def discover_summary_paths(base_root: Path) -> list[Path]:
    candidates = sorted(base_root.rglob("*_summary.json"))
    return [
        path
        for path in candidates
        if not path.name.endswith("_split_summary.json")
        and not path.name.startswith("_")
    ]


def choose_excel_engine() -> str:
    try:
        import xlsxwriter  # noqa: F401

        return "xlsxwriter"
    except Exception:
        try:
            import openpyxl  # noqa: F401

            return "openpyxl"
        except Exception as exc:
            raise RuntimeError(
                "Neither xlsxwriter nor openpyxl is available. Install one of them first."
            ) from exc


def fit_sheet(
    writer: pd.ExcelWriter, sheet_name: str, frame: pd.DataFrame, engine: str
):
    worksheet = writer.sheets[sheet_name]
    max_autofit_cols = min(len(frame.columns), 24)
    if engine == "openpyxl":
        worksheet.freeze_panes = "A2"
        for col_idx, column in enumerate(frame.columns[:max_autofit_cols], start=1):
            width = len(str(column))
            if not frame.empty:
                width = max(width, int(frame[column].astype(str).map(len).max()))
            width = min(max(width + 2, 12), 60)
            letter = worksheet.cell(row=1, column=col_idx).column_letter
            worksheet.column_dimensions[letter].width = width
    else:
        worksheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(frame.columns[:max_autofit_cols]):
            width = len(str(column))
            if not frame.empty:
                width = max(width, int(frame[column].astype(str).map(len).max()))
            width = min(max(width + 2, 12), 60)
            worksheet.set_column(col_idx, col_idx, width)


def style_readable_sheet(writer: pd.ExcelWriter, frame: pd.DataFrame, engine: str):
    sheet_name = "readable_compare"
    if sheet_name not in writer.sheets:
        return
    worksheet = writer.sheets[sheet_name]
    if frame.empty:
        return

    event_col_idx = None
    for idx, col_name in enumerate(frame.columns):
        if col_name == "event_summary":
            event_col_idx = idx
            break

    if engine == "openpyxl":
        from openpyxl.styles import Alignment

        wrap_alignment = Alignment(wrap_text=True, vertical="top")
        for row_idx in range(2, len(frame) + 2):
            worksheet.row_dimensions[row_idx].height = 36
        if event_col_idx is not None:
            col_letter = worksheet.cell(row=1, column=event_col_idx + 1).column_letter
            worksheet.column_dimensions[col_letter].width = 48
            for row_idx in range(2, len(frame) + 2):
                worksheet[f"{col_letter}{row_idx}"].alignment = wrap_alignment
    else:
        workbook = writer.book
        wrap_format = workbook.add_format({"text_wrap": True, "valign": "top"})
        for row_idx in range(1, len(frame) + 1):
            worksheet.set_row(row_idx, 36)
        if event_col_idx is not None:
            worksheet.set_column(event_col_idx, event_col_idx, 48, wrap_format)


def style_event_compare_sheet(
    writer: pd.ExcelWriter,
    frame: pd.DataFrame,
    engine: str,
    merge_columns: list[str],
    sheet_name: str = "event_compare",
):
    if sheet_name not in writer.sheets:
        return
    worksheet = writer.sheets[sheet_name]
    if frame.empty:
        return

    column_to_index = {name: idx for idx, name in enumerate(frame.columns)}
    merge_indices = [
        column_to_index[name] for name in merge_columns if name in column_to_index
    ]

    if engine == "openpyxl":
        from openpyxl.styles import Alignment

        top_wrap = Alignment(vertical="top", wrap_text=True)
        for row_idx in range(2, len(frame) + 2):
            worksheet.row_dimensions[row_idx].height = 28

        for col_name in ("event_time", "event_content", "forecast_start_time"):
            if col_name in column_to_index:
                col_letter = worksheet.cell(
                    row=1, column=column_to_index[col_name] + 1
                ).column_letter
                worksheet.column_dimensions[col_letter].width = (
                    28 if col_name != "event_content" else 48
                )

        for block_start in range(0, len(frame), 3):
            start_row = block_start + 2
            end_row = start_row + 2
            for col_idx in merge_indices:
                cell = worksheet.cell(row=start_row, column=col_idx + 1)
                cell.alignment = top_wrap
                worksheet.merge_cells(
                    start_row=start_row,
                    start_column=col_idx + 1,
                    end_row=end_row,
                    end_column=col_idx + 1,
                )
                worksheet.cell(row=start_row, column=col_idx + 1).alignment = top_wrap
    else:
        workbook = writer.book
        merge_format = workbook.add_format({"valign": "top", "text_wrap": True})
        for row_idx in range(1, len(frame) + 1):
            worksheet.set_row(row_idx, 28)

        for col_name in ("event_time", "event_content", "forecast_start_time"):
            if col_name in column_to_index:
                col_idx = column_to_index[col_name]
                width = 28 if col_name != "event_content" else 48
                worksheet.set_column(
                    col_idx,
                    col_idx,
                    width,
                    merge_format if col_name == "event_content" else None,
                )

        for block_start in range(0, len(frame), 3):
            start_row = block_start + 1
            end_row = start_row + 2
            for col_idx in merge_indices:
                value = frame.iloc[block_start, col_idx]
                if pd.isna(value):
                    value = ""
                worksheet.merge_range(
                    start_row,
                    col_idx,
                    end_row,
                    col_idx,
                    value,
                    merge_format,
                )


def write_event_compare_sheets(
    writer: pd.ExcelWriter,
    frame: pd.DataFrame,
    engine: str,
    merge_columns: list[str],
):
    if frame.empty:
        frame.to_excel(writer, index=False, sheet_name="event_compare")
        fit_sheet(writer, "event_compare", frame, engine)
        return 1

    if len(frame) % EVENT_COMPARE_BLOCK_ROWS != 0:
        raise ValueError(
            "event_compare rows must be a multiple of 3 so each event block stays complete."
        )

    rows_per_sheet = (
        EXCEL_MAX_DATA_ROWS // EVENT_COMPARE_BLOCK_ROWS
    ) * EVENT_COMPARE_BLOCK_ROWS
    if rows_per_sheet <= 0:
        raise ValueError("Invalid rows_per_sheet for event_compare export.")

    sheet_count = 0
    for start in range(0, len(frame), rows_per_sheet):
        stop = min(start + rows_per_sheet, len(frame))
        chunk = frame.iloc[start:stop].copy()
        sheet_count += 1
        sheet_name = (
            "event_compare" if sheet_count == 1 else f"event_compare_{sheet_count}"
        )
        chunk.to_excel(writer, index=False, sheet_name=sheet_name)
        fit_sheet(writer, sheet_name, chunk, engine)
        style_event_compare_sheet(
            writer=writer,
            frame=chunk,
            engine=engine,
            merge_columns=merge_columns,
            sheet_name=sheet_name,
        )
    return sheet_count


def write_frame_chunks(
    writer: pd.ExcelWriter,
    frame: pd.DataFrame,
    sheet_base: str,
    engine: str,
    row_limit: int = ROW_LIMIT_PER_SHEET,
):
    if len(frame) <= row_limit:
        frame.to_excel(writer, index=False, sheet_name=sheet_base)
        fit_sheet(writer, sheet_base, frame, engine)
        return

    chunk_count = math.ceil(len(frame) / row_limit)
    for idx in range(chunk_count):
        chunk = frame.iloc[idx * row_limit : (idx + 1) * row_limit].copy()
        sheet_name = f"{sheet_base}_{idx + 1}"
        chunk.to_excel(writer, index=False, sheet_name=sheet_name)
        fit_sheet(writer, sheet_name, chunk, engine)


def maybe_round_float(value, digits: int = 6):
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val):
        return None
    return round(val, digits)


def serialize_numeric_list(values, mask=None, digits: int = 6) -> str:
    out = []
    if mask is None:
        for value in values:
            out.append(maybe_round_float(value, digits))
    else:
        for value, keep in zip(values, mask):
            if bool(keep):
                out.append(maybe_round_float(value, digits))
            else:
                out.append(None)
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def serialize_int_list(values) -> str:
    return json.dumps(
        [int(v) for v in values], ensure_ascii=False, separators=(",", ":")
    )


def build_patch_metric_dict(preds, targets, mask):
    valid = mask.astype(bool)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return {
            "valid_points": 0,
            "mae": None,
            "mse": None,
            "rmse": None,
            "bias": None,
            "max_ae": None,
        }

    diff = preds[valid] - targets[valid]
    mse = float(np.mean(diff**2))
    return {
        "valid_points": valid_count,
        "mae": float(np.mean(np.abs(diff))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(diff)),
        "max_ae": float(np.max(np.abs(diff))),
    }


def format_timestamp(ts) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return pd.Timestamp(ts).isoformat()


def add_minutes(ts, offset_minutes: int):
    if ts is None or pd.isna(ts):
        return None
    return pd.Timestamp(ts) + pd.Timedelta(minutes=int(offset_minutes))


def load_time_metadata(data_csv: Path) -> dict[str, dict[str, Any]]:
    header = pd.read_csv(data_csv, nrows=0)
    available_cols = set(str(col) for col in header.columns)
    usecols = []
    for col in ("device_id", "id", "starttime", "endtime"):
        if col in available_cols:
            usecols.append(col)
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

    mapping: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        keys = []
        if "device_id" in row and not pd.isna(row["device_id"]):
            keys.append(str(row["device_id"]))
        if "id" in row and not pd.isna(row["id"]):
            keys.append(str(row["id"]))
        payload = {
            "starttime": row["starttime"],
            "endtime": row["endtime"],
        }
        for key in keys:
            mapping[key] = payload
    return mapping


def resolve_time_info(
    time_meta: dict[str, dict[str, Any]],
    sample_id: str,
    device_id: str,
    index_value: int | None = None,
) -> dict[str, Any]:
    meta = time_meta.get(str(device_id)) or time_meta.get(str(sample_id)) or {}
    starttime = meta.get("starttime")
    endtime = meta.get("endtime")
    absolute_time = None
    if index_value is not None and starttime is not None and not pd.isna(starttime):
        absolute_time = add_minutes(starttime, index_value * TIME_INTERVAL_MINUTES)
    return {
        "series_start_time": format_timestamp(starttime),
        "series_end_time": format_timestamp(endtime),
        "absolute_time": format_timestamp(absolute_time),
    }


def resolve_training_config(
    summary: dict[str, Any], summary_path: Path
) -> dict[str, Any]:
    resolved_config_path = summary.get("resolved_config_path")
    if resolved_config_path:
        path = Path(str(resolved_config_path)).expanduser()
        if not path.is_absolute():
            path = (summary_path.parent / path).resolve()
        if path.exists():
            return load_json(path)

    config = summary.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Cannot resolve config for summary: {summary_path}")
    return dict(config)


def resolve_checkpoint_path(
    summary: dict[str, Any],
    summary_path: Path,
    checkpoint_kind: str,
) -> tuple[Path, str]:
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
            raise FileNotFoundError(f"best checkpoint is missing in {summary_path}")
        return best_path, "best"
    if checkpoint_kind == "final":
        if final_path is None:
            raise FileNotFoundError(f"final checkpoint is missing in {summary_path}")
        return final_path, "final"

    if best_path is not None:
        return best_path, "best"
    if final_path is not None:
        return final_path, "final"
    raise FileNotFoundError(f"No checkpoint path found in {summary_path}")


def build_key_to_index_map(dataset) -> dict[tuple[str, str], int]:
    mapping: dict[tuple[str, str], int] = {}
    for idx, (sample_id, device_id) in enumerate(zip(dataset.ids, dataset.device_ids)):
        key = (normalize_entity_key(sample_id), normalize_entity_key(device_id))
        if key in mapping:
            raise ValueError(f"Duplicate dataset key encountered: {key}")
        mapping[key] = idx
    return mapping


def resolve_selected_indices(
    dataset,
    config: dict[str, Any],
    summary: dict[str, Any],
    split_name: str,
) -> tuple[list[int], dict[str, list[int]]]:
    split_export = summary.get("split_export")
    key_to_index = build_key_to_index_map(dataset)

    if isinstance(split_export, dict) and split_export.get("assignment_path"):
        assignment_path = Path(str(split_export["assignment_path"])).expanduser()
        if not assignment_path.is_absolute():
            assignment_path = (
                Path(config["out_dir"]).expanduser().resolve() / assignment_path
            ).resolve()
        if assignment_path.exists():
            assignment_df = pd.read_csv(assignment_path)
            split_indices: dict[str, list[int]] = {"train": [], "val": [], "all": []}
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
    normalized = {
        name: [int(idx) for idx in values] for name, values in split_map.items()
    }
    return normalized[split_name], normalized


def apply_saved_event_stats(
    dataset,
    summary: dict[str, Any],
    all_indices: list[int],
):
    event_stats = summary.get("event_stats")
    if not isinstance(event_stats, dict) or not event_stats:
        return
    dataset.apply_event_stats(event_stats, device_indices=all_indices)


def predict_base_device(base_model, sample, model_stride, device, patch_chunk_size):
    patch_array = sample["patches"]
    if patch_array.shape[0] == 0:
        return np.zeros((0, sample["targets"].shape[1]), dtype=np.float32)

    outputs = []
    base_model.eval()
    with torch.no_grad():
        if patch_chunk_size <= 0 or patch_chunk_size >= patch_array.shape[0]:
            patches = torch.from_numpy(patch_array).unsqueeze(0).to(device)
            preds = base_model(patches, start_idx=0, stride=model_stride)
            return preds.squeeze(0).cpu().numpy()

        for start in range(0, patch_array.shape[0], patch_chunk_size):
            stop = min(start + patch_chunk_size, patch_array.shape[0])
            patches = torch.from_numpy(patch_array[start:stop]).unsqueeze(0).to(device)
            preds = base_model(patches, start_idx=start, stride=model_stride)
            outputs.append(preds.squeeze(0).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_models(config: dict[str, Any], device: torch.device):
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
        use_cnn=bool(config["use_cnn"]),
        use_transformer=bool(config["use_transformer"]),
        use_temporal=bool(config["use_temporal"]),
        use_positional=bool(config["use_positional"]),
    ).to(device)
    model.eval()
    model.base.eval()
    return model


def load_post_model_state(model, checkpoint_path: Path):
    checkpoint = load_torch_checkpoint(checkpoint_path)
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
            f"Checkpoint mismatch for {checkpoint_path}. "
            f"missing={missing}, unexpected={unexpected}"
        )


def compute_horizon_metric_rows(preds, targets, masks):
    rows = []
    forecast_steps = targets.shape[1] if targets.ndim == 2 else 0
    for step in range(forecast_steps):
        step_mask = masks[:, step].astype(bool)
        valid_count = int(step_mask.sum())
        if valid_count == 0:
            rows.append(
                {
                    "horizon_step": step + 1,
                    "valid_points": 0,
                    "mae": None,
                    "mse": None,
                    "rmse": None,
                }
            )
            continue
        step_preds = preds[step_mask, step]
        step_targets = targets[step_mask, step]
        diff = step_preds - step_targets
        mse = float(np.mean(diff**2))
        mae = float(np.mean(np.abs(diff)))
        rows.append(
            {
                "horizon_step": step + 1,
                "valid_points": valid_count,
                "mae": mae,
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
            }
        )
    return rows


def build_overall_metrics_frame(
    pre_preds,
    post_preds,
    targets,
    masks,
    checkpoint_kind: str,
) -> pd.DataFrame:
    pre_metrics = event_main.compute_metrics(pre_preds, targets, masks)
    post_metrics = event_main.compute_metrics(post_preds, targets, masks)
    valid_points = int(masks.astype(bool).sum())
    rows = [
        {
            "model_state": "pre_finetune_base",
            "checkpoint_kind": "pretrain_base",
            "valid_points": valid_points,
            "mae": pre_metrics["mae"],
            "mse": pre_metrics["mse"],
            "rmse": pre_metrics["rmse"],
        },
        {
            "model_state": "post_finetune_event",
            "checkpoint_kind": checkpoint_kind,
            "valid_points": valid_points,
            "mae": post_metrics["mae"],
            "mse": post_metrics["mse"],
            "rmse": post_metrics["rmse"],
        },
        {
            "model_state": "delta_post_minus_pre",
            "checkpoint_kind": checkpoint_kind,
            "valid_points": valid_points,
            "mae": (
                None
                if pre_metrics["mae"] is None or post_metrics["mae"] is None
                else post_metrics["mae"] - pre_metrics["mae"]
            ),
            "mse": (
                None
                if pre_metrics["mse"] is None or post_metrics["mse"] is None
                else post_metrics["mse"] - pre_metrics["mse"]
            ),
            "rmse": (
                None
                if pre_metrics["rmse"] is None or post_metrics["rmse"] is None
                else post_metrics["rmse"] - pre_metrics["rmse"]
            ),
        },
    ]
    return pd.DataFrame(rows)


def build_horizon_metrics_frame(pre_preds, post_preds, targets, masks) -> pd.DataFrame:
    pre_rows = compute_horizon_metric_rows(pre_preds, targets, masks)
    post_rows = compute_horizon_metric_rows(post_preds, targets, masks)
    merged_rows = []
    for pre_row, post_row in zip(pre_rows, post_rows):
        merged_rows.append(
            {
                "horizon_step": int(pre_row["horizon_step"]),
                "valid_points": int(pre_row["valid_points"]),
                "pre_mae": pre_row["mae"],
                "post_mae": post_row["mae"],
                "delta_mae": (
                    None
                    if pre_row["mae"] is None or post_row["mae"] is None
                    else post_row["mae"] - pre_row["mae"]
                ),
                "pre_mse": pre_row["mse"],
                "post_mse": post_row["mse"],
                "delta_mse": (
                    None
                    if pre_row["mse"] is None or post_row["mse"] is None
                    else post_row["mse"] - pre_row["mse"]
                ),
                "pre_rmse": pre_row["rmse"],
                "post_rmse": post_row["rmse"],
                "delta_rmse": (
                    None
                    if pre_row["rmse"] is None or post_row["rmse"] is None
                    else post_row["rmse"] - pre_row["rmse"]
                ),
            }
        )
    return pd.DataFrame(merged_rows)


def build_readable_columns(config: dict[str, Any]) -> list[str]:
    patch_size = int(config["patch_size"])
    forecast_steps = int(config["forecast_steps"])
    columns = [
        "sample_id",
        "device_id",
        "patch_index",
        "row_type",
        "patch_start_index",
        "patch_start_time",
        "target_start_index",
        "target_start_time",
        "target_end_index",
        "target_end_time",
        "valid_target_points",
        "event_count",
        "event_summary",
        "row_mae",
        "row_rmse",
        "row_bias",
        "row_max_ae",
        "delta_mae_vs_pre",
        "delta_rmse_vs_pre",
        "delta_bias_vs_pre",
    ]
    columns.extend([f"input_t-{patch_size - idx}" for idx in range(patch_size)])
    columns.extend([f"time_t+{step + 1}" for step in range(forecast_steps)])
    columns.extend([f"value_t+{step + 1}" for step in range(forecast_steps)])
    return columns


def build_event_compare_columns(config: dict[str, Any]) -> list[str]:
    patch_size = int(config["patch_size"])
    forecast_steps = int(config["forecast_steps"])
    columns = [
        "sample_id",
        "device_id",
        "window_index",
    ]
    columns.extend([f"input_{idx + 1}" for idx in range(patch_size)])
    columns.extend(
        [
            "event_phase",
            "event_time",
            "event_content",
            "forecast_start_time",
            "row_type",
        ]
    )
    columns.extend([f"t+{step + 1}" for step in range(forecast_steps)])
    columns.extend(["MAE", "RMSE", "Bias", "MAE_change"])
    return columns


def format_event_content(row_values: np.ndarray, event_cols: list[str]) -> str:
    parts = []
    for event_idx, event_name in enumerate(event_cols):
        value = maybe_round_float(row_values[event_idx])
        if value is None or value == 0:
            continue
        parts.append(f"{event_name}={value}")
    return "; ".join(parts) if parts else "无事件"


def choose_representative_patch_index(
    starts: list[int],
    event_index: int,
    patch_size: int,
    forecast_steps: int,
) -> int | None:
    best_score = None
    best_patch_index = None
    for patch_index, patch_start in enumerate(starts):
        target_start = int(patch_start + patch_size)
        target_end = int(target_start + forecast_steps - 1)
        if event_index < patch_start or event_index > target_end:
            continue

        in_input = event_index < target_start
        if in_input:
            score = (0, target_start - event_index, patch_index)
        else:
            score = (1, event_index - target_start, patch_index)

        if best_score is None or score < best_score:
            best_score = score
            best_patch_index = patch_index
    return best_patch_index


def build_event_summary_text(
    event_window: np.ndarray,
    event_cols: list[str],
    patch_start: int,
    patch_size: int,
    sample_id: str,
    device_id: str,
    time_meta: dict[str, dict[str, Any]],
) -> tuple[int, str]:
    lines = []
    event_count = 0
    for rel_offset in range(event_window.shape[0]):
        row_values = event_window[rel_offset]
        nonzero_pairs = []
        for event_idx, event_name in enumerate(event_cols):
            value = row_values[event_idx]
            rounded = maybe_round_float(value)
            if rounded is None or rounded == 0:
                continue
            nonzero_pairs.append(f"{event_name}={rounded}")
        if not nonzero_pairs:
            continue
        abs_index = int(patch_start + rel_offset)
        time_info = resolve_time_info(time_meta, sample_id, device_id, abs_index)
        phase = "input" if rel_offset < patch_size else "forecast"
        target_offset = rel_offset - patch_size
        if target_offset < 0:
            rel_text = f"t{target_offset}"
        else:
            rel_text = f"t+{target_offset + 1}"
        line = (
            f"[{phase}] {rel_text} | idx={abs_index} | "
            f"time={time_info['absolute_time'] or 'NA'} | " + "; ".join(nonzero_pairs)
        )
        lines.append(line)
        event_count += 1
    return event_count, "\n".join(lines)


def build_offset_reference_frame(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    patch_size = int(config["patch_size"])
    forecast_steps = int(config["forecast_steps"])
    for rel_idx in range(-patch_size, 0):
        rows.append(
            {
                "kind": "input_patch",
                "relative_offset": rel_idx,
                "minutes_from_target_start": rel_idx * TIME_INTERVAL_MINUTES,
                "meaning": "input glucose point before the first forecast step",
            }
        )
    for rel_idx in range(0, forecast_steps):
        rows.append(
            {
                "kind": "forecast_horizon",
                "relative_offset": rel_idx,
                "minutes_from_target_start": rel_idx * TIME_INTERVAL_MINUTES,
                "meaning": f"forecast step t+{rel_idx + 1}",
            }
        )
    return pd.DataFrame(rows)


def build_run_metadata_frame(
    summary_path: Path,
    summary: dict[str, Any],
    config: dict[str, Any],
    workbook_path: Path,
    split_name: str,
    checkpoint_path: Path,
    checkpoint_kind: str,
    num_selected_devices: int,
    patch_count: int,
    valid_forecast_rows: int,
    include_zero_event_rows: bool,
) -> pd.DataFrame:
    rows = [
        {"section": "run", "key": "summary_path", "value": str(summary_path.resolve())},
        {"section": "run", "key": "run_tag", "value": str(summary.get("run_tag", ""))},
        {
            "section": "run",
            "key": "workbook_path",
            "value": str(workbook_path.resolve()),
        },
        {"section": "run", "key": "split", "value": split_name},
        {"section": "run", "key": "checkpoint_kind", "value": checkpoint_kind},
        {
            "section": "run",
            "key": "checkpoint_path",
            "value": str(checkpoint_path.resolve()),
        },
        {"section": "data", "key": "data_csv", "value": str(config["data_csv"])},
        {"section": "data", "key": "onehot_csv", "value": str(config["onehot_csv"])},
        {
            "section": "data",
            "key": "pretrain_ckpt",
            "value": str(config["pretrain_ckpt"]),
        },
        {
            "section": "data",
            "key": "selected_devices",
            "value": str(num_selected_devices),
        },
        {"section": "data", "key": "patch_rows", "value": str(patch_count)},
        {
            "section": "data",
            "key": "valid_forecast_rows",
            "value": str(valid_forecast_rows),
        },
        {"section": "config", "key": "patch_size", "value": str(config["patch_size"])},
        {
            "section": "config",
            "key": "forecast_steps",
            "value": str(config["forecast_steps"]),
        },
        {"section": "config", "key": "stride", "value": str(config["stride"])},
        {
            "section": "config",
            "key": "model_stride",
            "value": str(config["model_stride"]),
        },
        {
            "section": "config",
            "key": "normalize_continuous_events",
            "value": str(config.get("normalize_continuous_events", False)),
        },
        {
            "section": "config",
            "key": "interval_minutes",
            "value": str(TIME_INTERVAL_MINUTES),
        },
        {
            "section": "event",
            "key": "event_columns",
            "value": json.dumps(list(config["event_cols"]), ensure_ascii=False),
        },
        {
            "section": "event",
            "key": "include_zero_event_rows",
            "value": str(include_zero_event_rows),
        },
        {
            "section": "note",
            "key": "patch_core",
            "value": (
                "input_values_json stores the model input patch values after the same glucose interpolation "
                "used in training; input_absolute_indices_json and target_absolute_indices_json keep the "
                "point-by-point positions on the original 5-minute grid."
            ),
        },
        {
            "section": "note",
            "key": "event_detail",
            "value": (
                "event_detail exports raw event feature values before any optional train-only normalization; "
                "each row is one patch/time position."
            ),
        },
        {
            "section": "note",
            "key": "forecast_detail",
            "value": (
                "forecast_detail places true_value, pred_pre_value and pred_post_value side by side "
                "with the absolute target time."
            ),
        },
        {
            "section": "note",
            "key": "readable_compare",
            "value": (
                "readable_compare expands each prediction window into three rows: 真实值, 微调前, 微调后; "
                "the input points are spread across input_t-* columns and event_summary lists event time/name/value."
            ),
        },
    ]
    return pd.DataFrame(rows)


def export_single_run(
    summary_path: Path,
    split_override: str | None,
    checkpoint_kind: str,
    device: torch.device,
    output_name: str | None,
    include_zero_event_rows: bool,
):
    summary = load_json(summary_path)
    config = resolve_training_config(summary, summary_path)
    run_tag = str(summary.get("run_tag", summary_path.stem.replace("_summary", "")))
    prediction_export = summary.get("prediction_export")
    prediction_export = prediction_export if isinstance(prediction_export, dict) else {}
    split_name = (
        split_override
        or prediction_export.get("split")
        or config.get("prediction_split", "val")
    )

    raw_dataset = event_main.build_dataset(config)
    model_dataset = event_main.build_dataset(config)
    selected_indices, split_map = resolve_selected_indices(
        dataset=model_dataset,
        config=config,
        summary=summary,
        split_name=split_name,
    )
    apply_saved_event_stats(
        dataset=model_dataset,
        summary=summary,
        all_indices=list(split_map["all"]),
    )

    checkpoint_path, resolved_ckpt_kind = resolve_checkpoint_path(
        summary=summary,
        summary_path=summary_path,
        checkpoint_kind=checkpoint_kind,
    )
    time_meta = load_time_metadata(Path(str(config["data_csv"])))

    if device.type == "cuda":
        torch.cuda.set_device(device)

    pre_model = build_models(config, device)
    pre_base = pre_model.base
    del pre_model

    post_model = build_models(config, device)
    load_post_model_state(post_model, checkpoint_path)
    post_model.eval()
    post_model.base.eval()

    patch_chunk_size = int(config.get("prediction_patch_chunk_size", 0))
    forecast_rows: list[dict[str, Any]] = []
    patch_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    readable_rows: list[dict[str, Any]] = []
    event_compare_rows: list[dict[str, Any]] = []

    all_pre_preds = []
    all_post_preds = []
    all_targets = []
    all_masks = []

    event_cols = list(config["event_cols"])
    patch_size = int(config["patch_size"])
    forecast_steps = int(config["forecast_steps"])

    for dataset_index in selected_indices:
        model_sample = model_dataset[dataset_index]
        raw_sample = raw_dataset[dataset_index]
        pre_preds = predict_base_device(
            base_model=pre_base,
            sample=model_sample,
            model_stride=int(config["model_stride"]),
            device=device,
            patch_chunk_size=patch_chunk_size,
        )
        post_preds = event_main.predict_device(
            model=post_model,
            sample=model_sample,
            model_stride=int(config["model_stride"]),
            device=device,
            patch_chunk_size=patch_chunk_size,
        )

        targets = raw_sample["targets"]
        masks = raw_sample["masks"]
        patches = raw_sample["patches"]
        events = raw_sample["events"]
        starts = raw_dataset.device_patch_starts[dataset_index][: pre_preds.shape[0]]

        if pre_preds.shape != post_preds.shape or pre_preds.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape mismatch for {raw_sample['device_id']}: "
                f"pre={pre_preds.shape}, post={post_preds.shape}, target={targets.shape}"
            )

        all_pre_preds.append(pre_preds)
        all_post_preds.append(post_preds)
        all_targets.append(targets)
        all_masks.append(masks)

        sample_id = str(raw_sample["id"])
        device_id = str(raw_sample["device_id"])

        for patch_index, patch_start in enumerate(starts):
            patch_end = int(patch_start + patch_size - 1)
            target_start = int(patch_start + patch_size)
            target_end = int(target_start + forecast_steps - 1)
            patch_time_info = resolve_time_info(
                time_meta, sample_id, device_id, patch_start
            )
            patch_end_time_info = resolve_time_info(
                time_meta, sample_id, device_id, patch_end
            )
            target_time_info = resolve_time_info(
                time_meta, sample_id, device_id, target_start
            )
            target_end_time_info = resolve_time_info(
                time_meta, sample_id, device_id, target_end
            )

            patch_mask = masks[patch_index].astype(bool)
            target_values = serialize_numeric_list(
                targets[patch_index], mask=patch_mask
            )
            pre_values = serialize_numeric_list(pre_preds[patch_index], mask=patch_mask)
            post_values = serialize_numeric_list(
                post_preds[patch_index], mask=patch_mask
            )
            pre_abs_errors = serialize_numeric_list(
                np.abs(pre_preds[patch_index] - targets[patch_index]),
                mask=patch_mask,
            )
            post_abs_errors = serialize_numeric_list(
                np.abs(post_preds[patch_index] - targets[patch_index]),
                mask=patch_mask,
            )

            event_window = events[patch_index]
            event_nonzero_rows = np.any(np.abs(event_window) > 0, axis=1)
            event_count, event_summary = build_event_summary_text(
                event_window=event_window,
                event_cols=event_cols,
                patch_start=int(patch_start),
                patch_size=patch_size,
                sample_id=sample_id,
                device_id=device_id,
                time_meta=time_meta,
            )

            input_values = [maybe_round_float(value) for value in patches[patch_index]]
            pred_time_values = []
            for horizon_step in range(forecast_steps):
                abs_index = int(target_start + horizon_step)
                target_point_time = resolve_time_info(
                    time_meta, sample_id, device_id, abs_index
                )
                pred_time_values.append(target_point_time["absolute_time"])

            pre_patch_metrics = build_patch_metric_dict(
                pre_preds[patch_index], targets[patch_index], patch_mask
            )
            post_patch_metrics = build_patch_metric_dict(
                post_preds[patch_index], targets[patch_index], patch_mask
            )

            patch_rows.append(
                {
                    "sample_id": sample_id,
                    "device_id": device_id,
                    "patch_index": int(patch_index),
                    "patch_start_index": int(patch_start),
                    "patch_end_index": patch_end,
                    "target_start_index": target_start,
                    "target_end_index": target_end,
                    "series_start_time": patch_time_info["series_start_time"],
                    "series_end_time": patch_time_info["series_end_time"],
                    "patch_start_time": patch_time_info["absolute_time"],
                    "patch_end_time": patch_end_time_info["absolute_time"],
                    "target_start_time": target_time_info["absolute_time"],
                    "target_end_time": target_end_time_info["absolute_time"],
                    "num_valid_target_points": int(patch_mask.sum()),
                    "num_nonzero_event_rows": int(event_nonzero_rows.sum()),
                    "num_nonzero_event_cells": int(np.count_nonzero(event_window)),
                    "input_absolute_indices_json": serialize_int_list(
                        range(int(patch_start), int(patch_start + patch_size))
                    ),
                    "target_absolute_indices_json": serialize_int_list(
                        range(int(target_start), int(target_start + forecast_steps))
                    ),
                    "input_values_json": serialize_numeric_list(patches[patch_index]),
                    "mask_json": serialize_int_list(patch_mask.astype(np.int32)),
                    "true_values_json": target_values,
                    "pred_pre_values_json": pre_values,
                    "pred_post_values_json": post_values,
                    "pre_abs_errors_json": pre_abs_errors,
                    "post_abs_errors_json": post_abs_errors,
                    "event_summary": event_summary,
                }
            )

            base_readable = {
                "sample_id": sample_id,
                "device_id": device_id,
                "patch_index": int(patch_index),
                "patch_start_index": int(patch_start),
                "patch_start_time": patch_time_info["absolute_time"],
                "target_start_index": target_start,
                "target_start_time": target_time_info["absolute_time"],
                "target_end_index": target_end,
                "target_end_time": target_end_time_info["absolute_time"],
                "valid_target_points": int(patch_mask.sum()),
                "event_count": int(event_count),
                "event_summary": event_summary,
            }
            for idx, value in enumerate(input_values):
                base_readable[f"input_t-{patch_size - idx}"] = value
            for idx, value in enumerate(pred_time_values):
                base_readable[f"time_t+{idx + 1}"] = value

            true_row = dict(base_readable)
            true_row["row_type"] = "真实值"
            true_row["row_mae"] = None
            true_row["row_rmse"] = None
            true_row["row_bias"] = None
            true_row["row_max_ae"] = None
            true_row["delta_mae_vs_pre"] = None
            true_row["delta_rmse_vs_pre"] = None
            true_row["delta_bias_vs_pre"] = None
            for idx in range(forecast_steps):
                true_row[f"value_t+{idx + 1}"] = (
                    maybe_round_float(targets[patch_index, idx])
                    if patch_mask[idx]
                    else None
                )
            readable_rows.append(true_row)

            pre_row = dict(base_readable)
            pre_row["row_type"] = "微调前"
            pre_row["row_mae"] = maybe_round_float(pre_patch_metrics["mae"])
            pre_row["row_rmse"] = maybe_round_float(pre_patch_metrics["rmse"])
            pre_row["row_bias"] = maybe_round_float(pre_patch_metrics["bias"])
            pre_row["row_max_ae"] = maybe_round_float(pre_patch_metrics["max_ae"])
            pre_row["delta_mae_vs_pre"] = (
                0.0 if pre_patch_metrics["mae"] is not None else None
            )
            pre_row["delta_rmse_vs_pre"] = (
                0.0 if pre_patch_metrics["rmse"] is not None else None
            )
            pre_row["delta_bias_vs_pre"] = (
                0.0 if pre_patch_metrics["bias"] is not None else None
            )
            for idx in range(forecast_steps):
                pre_row[f"value_t+{idx + 1}"] = (
                    maybe_round_float(pre_preds[patch_index, idx])
                    if patch_mask[idx]
                    else None
                )
            readable_rows.append(pre_row)

            post_row = dict(base_readable)
            post_row["row_type"] = "微调后"
            post_row["row_mae"] = maybe_round_float(post_patch_metrics["mae"])
            post_row["row_rmse"] = maybe_round_float(post_patch_metrics["rmse"])
            post_row["row_bias"] = maybe_round_float(post_patch_metrics["bias"])
            post_row["row_max_ae"] = maybe_round_float(post_patch_metrics["max_ae"])
            post_row["delta_mae_vs_pre"] = (
                maybe_round_float(post_patch_metrics["mae"] - pre_patch_metrics["mae"])
                if post_patch_metrics["mae"] is not None
                and pre_patch_metrics["mae"] is not None
                else None
            )
            post_row["delta_rmse_vs_pre"] = (
                maybe_round_float(
                    post_patch_metrics["rmse"] - pre_patch_metrics["rmse"]
                )
                if post_patch_metrics["rmse"] is not None
                and pre_patch_metrics["rmse"] is not None
                else None
            )
            post_row["delta_bias_vs_pre"] = (
                maybe_round_float(
                    post_patch_metrics["bias"] - pre_patch_metrics["bias"]
                )
                if post_patch_metrics["bias"] is not None
                and pre_patch_metrics["bias"] is not None
                else None
            )
            for idx in range(forecast_steps):
                post_row[f"value_t+{idx + 1}"] = (
                    maybe_round_float(post_preds[patch_index, idx])
                    if patch_mask[idx]
                    else None
                )
            readable_rows.append(post_row)

            for horizon_step in range(forecast_steps):
                if not patch_mask[horizon_step]:
                    continue
                abs_index = int(target_start + horizon_step)
                target_point_time = resolve_time_info(
                    time_meta, sample_id, device_id, abs_index
                )
                true_value = float(targets[patch_index, horizon_step])
                pre_value = float(pre_preds[patch_index, horizon_step])
                post_value = float(post_preds[patch_index, horizon_step])
                pre_abs_error = abs(pre_value - true_value)
                post_abs_error = abs(post_value - true_value)
                forecast_rows.append(
                    {
                        "sample_id": sample_id,
                        "device_id": device_id,
                        "patch_index": int(patch_index),
                        "patch_start_index": int(patch_start),
                        "target_time_index": abs_index,
                        "target_time": target_point_time["absolute_time"],
                        "horizon_step": int(horizon_step + 1),
                        "true_value": maybe_round_float(true_value),
                        "pred_pre_value": maybe_round_float(pre_value),
                        "pred_post_value": maybe_round_float(post_value),
                        "pre_abs_error": maybe_round_float(pre_abs_error),
                        "post_abs_error": maybe_round_float(post_abs_error),
                        "delta_abs_error_post_minus_pre": maybe_round_float(
                            post_abs_error - pre_abs_error
                        ),
                    }
                )

            row_selector = np.ones(event_window.shape[0], dtype=bool)
            if not include_zero_event_rows:
                row_selector = event_nonzero_rows
            for rel_offset, keep_row in enumerate(row_selector):
                if not keep_row:
                    continue
                abs_index = int(patch_start + rel_offset)
                time_info = resolve_time_info(
                    time_meta, sample_id, device_id, abs_index
                )
                event_time_value = time_info["absolute_time"] or f"index={abs_index}"
                forecast_start_value = (
                    target_time_info["absolute_time"] or f"index={target_start}"
                )
                event_phase_value = "输入期" if rel_offset < patch_size else "预测期"
                event_content_value = format_event_content(
                    event_window[rel_offset], event_cols
                )
                row = {
                    "sample_id": sample_id,
                    "device_id": device_id,
                    "patch_index": int(patch_index),
                    "patch_start_index": int(patch_start),
                    "relative_offset": int(rel_offset - patch_size),
                    "absolute_time_index": abs_index,
                    "timestamp": time_info["absolute_time"],
                    "window_region": "input" if rel_offset < patch_size else "forecast",
                }
                for event_col_idx, event_col in enumerate(event_cols):
                    row[event_col] = maybe_round_float(
                        event_window[rel_offset, event_col_idx]
                    )
                event_rows.append(row)

                shared_compare = {
                    "sample_id": sample_id,
                    "device_id": device_id,
                    "window_index": int(patch_index),
                    "event_phase": event_phase_value,
                    "event_time": event_time_value,
                    "event_content": event_content_value,
                    "forecast_start_time": forecast_start_value,
                }

                true_row = dict(shared_compare)
                true_row["row_type"] = "真实值"
                true_row["MAE"] = None
                true_row["RMSE"] = None
                true_row["Bias"] = None
                true_row["MAE_change"] = None
                for idx in range(patch_size):
                    true_row[f"input_{idx + 1}"] = input_values[idx]
                for idx in range(forecast_steps):
                    true_row[f"t+{idx + 1}"] = (
                        maybe_round_float(targets[patch_index, idx])
                        if patch_mask[idx]
                        else None
                    )
                event_compare_rows.append(true_row)

                pre_row = dict(shared_compare)
                pre_row["row_type"] = "微调前"
                pre_row["MAE"] = maybe_round_float(pre_patch_metrics["mae"])
                pre_row["RMSE"] = maybe_round_float(pre_patch_metrics["rmse"])
                pre_row["Bias"] = maybe_round_float(pre_patch_metrics["bias"])
                pre_row["MAE_change"] = None
                for idx in range(patch_size):
                    pre_row[f"input_{idx + 1}"] = None
                for idx in range(forecast_steps):
                    pre_row[f"t+{idx + 1}"] = (
                        maybe_round_float(pre_preds[patch_index, idx])
                        if patch_mask[idx]
                        else None
                    )
                event_compare_rows.append(pre_row)

                post_row = dict(shared_compare)
                post_row["row_type"] = "微调后"
                post_row["MAE"] = maybe_round_float(post_patch_metrics["mae"])
                post_row["RMSE"] = maybe_round_float(post_patch_metrics["rmse"])
                post_row["Bias"] = maybe_round_float(post_patch_metrics["bias"])
                post_row["MAE_change"] = (
                    maybe_round_float(
                        post_patch_metrics["mae"] - pre_patch_metrics["mae"]
                    )
                    if post_patch_metrics["mae"] is not None
                    and pre_patch_metrics["mae"] is not None
                    else None
                )
                for idx in range(patch_size):
                    post_row[f"input_{idx + 1}"] = None
                for idx in range(forecast_steps):
                    post_row[f"t+{idx + 1}"] = (
                        maybe_round_float(post_preds[patch_index, idx])
                        if patch_mask[idx]
                        else None
                    )
                event_compare_rows.append(post_row)

    if all_pre_preds:
        pre_matrix = np.concatenate(all_pre_preds, axis=0)
        post_matrix = np.concatenate(all_post_preds, axis=0)
        target_matrix = np.concatenate(all_targets, axis=0)
        mask_matrix = np.concatenate(all_masks, axis=0)
    else:
        pre_matrix = np.zeros((0, forecast_steps), dtype=np.float32)
        post_matrix = np.zeros((0, forecast_steps), dtype=np.float32)
        target_matrix = np.zeros((0, forecast_steps), dtype=np.float32)
        mask_matrix = np.zeros((0, forecast_steps), dtype=np.float32)

    overall_metrics = build_overall_metrics_frame(
        pre_preds=pre_matrix,
        post_preds=post_matrix,
        targets=target_matrix,
        masks=mask_matrix,
        checkpoint_kind=resolved_ckpt_kind,
    )
    horizon_metrics = build_horizon_metrics_frame(
        pre_preds=pre_matrix,
        post_preds=post_matrix,
        targets=target_matrix,
        masks=mask_matrix,
    )

    output_dir = summary_path.parent
    workbook_name = (
        output_name
        if output_name
        else f"{run_tag}_prepost_prediction_{split_name}.xlsx"
    )
    workbook_path = output_dir / workbook_name

    metadata_frame = build_run_metadata_frame(
        summary_path=summary_path,
        summary=summary,
        config=config,
        workbook_path=workbook_path,
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=resolved_ckpt_kind,
        num_selected_devices=len(selected_indices),
        patch_count=len(patch_rows),
        valid_forecast_rows=len(forecast_rows),
        include_zero_event_rows=include_zero_event_rows,
    )
    offset_reference = build_offset_reference_frame(config)
    forecast_frame = pd.DataFrame(
        forecast_rows,
        columns=[
            "sample_id",
            "device_id",
            "patch_index",
            "patch_start_index",
            "target_time_index",
            "target_time",
            "horizon_step",
            "true_value",
            "pred_pre_value",
            "pred_post_value",
            "pre_abs_error",
            "post_abs_error",
            "delta_abs_error_post_minus_pre",
        ],
    )
    patch_frame = pd.DataFrame(
        patch_rows,
        columns=[
            "sample_id",
            "device_id",
            "patch_index",
            "patch_start_index",
            "patch_end_index",
            "target_start_index",
            "target_end_index",
            "series_start_time",
            "series_end_time",
            "patch_start_time",
            "patch_end_time",
            "target_start_time",
            "target_end_time",
            "num_valid_target_points",
            "num_nonzero_event_rows",
            "num_nonzero_event_cells",
            "input_absolute_indices_json",
            "target_absolute_indices_json",
            "input_values_json",
            "mask_json",
            "true_values_json",
            "pred_pre_values_json",
            "pred_post_values_json",
            "pre_abs_errors_json",
            "post_abs_errors_json",
            "event_summary",
        ],
    )
    event_frame = pd.DataFrame(
        event_rows,
        columns=[
            "sample_id",
            "device_id",
            "patch_index",
            "patch_start_index",
            "relative_offset",
            "absolute_time_index",
            "timestamp",
            "window_region",
            *event_cols,
        ],
    )
    event_compare_frame = pd.DataFrame(
        event_compare_rows,
        columns=build_event_compare_columns(config),
    )
    readable_frame = pd.DataFrame(
        readable_rows,
        columns=build_readable_columns(config),
    )

    engine = choose_excel_engine()
    engine_kwargs = {}
    if engine == "xlsxwriter":
        engine_kwargs = {"options": {"strings_to_urls": False}}
    with pd.ExcelWriter(
        workbook_path, engine=engine, engine_kwargs=engine_kwargs
    ) as writer:
        event_compare_sheet_count = write_event_compare_sheets(
            writer=writer,
            frame=event_compare_frame,
            engine=engine,
            merge_columns=[
                "sample_id",
                "device_id",
                "window_index",
                "event_phase",
                "event_time",
                "event_content",
                "forecast_start_time",
            ],
        )

        overall_metrics.to_excel(writer, index=False, sheet_name="metrics", startrow=1)
        horizon_metrics.to_excel(
            writer,
            index=False,
            sheet_name="metrics",
            startrow=len(overall_metrics) + 5,
        )
        metrics_sheet = writer.sheets["metrics"]
        if engine == "openpyxl":
            metrics_sheet["A1"] = "overall_metrics"
            metrics_sheet.cell(
                row=len(overall_metrics) + 5, column=1
            ).value = "horizon_metrics"
            metrics_sheet.cell(
                row=len(overall_metrics) + len(horizon_metrics) + 8, column=1
            ).value = f"event_compare_sheets={event_compare_sheet_count}"
        else:
            metrics_sheet.write(0, 0, "overall_metrics")
            metrics_sheet.write(len(overall_metrics) + 4, 0, "horizon_metrics")
            metrics_sheet.write(
                len(overall_metrics) + len(horizon_metrics) + 7,
                0,
                f"event_compare_sheets={event_compare_sheet_count}",
            )
        fit_sheet(
            writer,
            "metrics",
            pd.concat(
                [overall_metrics, horizon_metrics],
                axis=0,
                ignore_index=True,
                sort=False,
            ),
            engine,
        )

    del pre_base
    del post_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    return {
        "summary_path": str(summary_path),
        "workbook_path": str(workbook_path),
        "run_tag": run_tag,
        "split": split_name,
        "checkpoint_kind": resolved_ckpt_kind,
        "patch_rows": len(patch_rows),
        "forecast_rows": len(forecast_rows),
        "event_rows": len(event_rows),
    }


def should_keep_summary(
    summary_path: Path,
    summary_payload: dict[str, Any],
    task_filter: set[str] | None,
    run_tag_filters: list[str] | None,
) -> bool:
    if task_filter is not None and summary_path.parent.name not in task_filter:
        return False

    if run_tag_filters:
        run_tag = str(summary_payload.get("run_tag", summary_path.stem))
        lower_tag = run_tag.lower()
        if not any(token.lower() in lower_tag for token in run_tag_filters):
            return False
    return True


def main():
    args = build_parser().parse_args()

    if args.summary_paths:
        summary_paths = [
            Path(item).expanduser().resolve() for item in args.summary_paths
        ]
    else:
        base_root = Path(args.base_root).expanduser().resolve()
        if not base_root.exists():
            raise FileNotFoundError(f"Base root not found: {base_root}")
        summary_paths = discover_summary_paths(base_root)

    if not summary_paths:
        raise FileNotFoundError("No event fine-tuning summary files were found.")

    task_filter = set(args.tasks) if args.tasks else None
    selected_summary_paths = []
    for summary_path in summary_paths:
        payload = load_json(summary_path)
        if should_keep_summary(summary_path, payload, task_filter, args.run_tags):
            selected_summary_paths.append(summary_path)

    if not selected_summary_paths:
        raise FileNotFoundError("No summary files matched the requested filters.")

    device = torch.device(args.device)
    results = []
    for summary_path in selected_summary_paths:
        print(f"[EXPORT] {summary_path}")
        result = export_single_run(
            summary_path=summary_path,
            split_override=args.split,
            checkpoint_kind=args.checkpoint_kind,
            device=device,
            output_name=args.output_name,
            include_zero_event_rows=bool(args.include_zero_event_rows),
        )
        results.append(result)
        print(
            f"[DONE] run_tag={result['run_tag']} split={result['split']} "
            f"patch_rows={result['patch_rows']} forecast_rows={result['forecast_rows']} "
            f"event_rows={result['event_rows']} workbook={result['workbook_path']}"
        )

    print(f"[SUMMARY] Exported {len(results)} workbook(s).")


if __name__ == "__main__":
    main()


