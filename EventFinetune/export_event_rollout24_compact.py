from __future__ import annotations

import argparse
import csv
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CODE_ROOT = REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import EventFinetune_main as event_main
from EventFinetune import export_event_prepost_prediction_excel as helpers
from EventFinetune.Event_Finetune_data_utils import fill_nan_linear

COL_SAMPLE_ID = "样本ID"
COL_DEVICE_ID = "设备ID"
COL_WINDOW_INDEX = "窗口编号"
COL_WINDOW_START = "窗口起点"
COL_INPUT_START_TIME = "输入开始时间"
COL_FORECAST_START_TIME = "预测开始时间"
COL_INPUT_EVENT_COUNT = "输入期事件数"
COL_FORECAST_EVENT_COUNT = "预测期事件数"
COL_ALL_EVENT_SUMMARY = "全部事件摘要"
COL_ROW_TYPE = "行类型"
COL_MAE = "MAE"
COL_RMSE = "RMSE"
COL_BIAS = "Bias"
COL_MAE_DELTA = "MAE变化"

ROW_TRUE = "真实值"
ROW_PRE = "微调前"
ROW_POST = "微调后"

METRIC_SCOPE = "范围"
METRIC_NAME = "名称"
METRIC_VALUE = "值"
METRIC_VALID_POINTS = "有效点数"
METRIC_PRE_MAE = "微调前MAE"
METRIC_POST_MAE = "微调后MAE"
METRIC_PRE_RMSE = "微调前RMSE"
METRIC_POST_RMSE = "微调后RMSE"
METRIC_PRE_BIAS = "微调前Bias"
METRIC_POST_BIAS = "微调后Bias"
METRIC_RMSE_DELTA = "RMSE变化"
METRIC_BIAS_DELTA = "Bias变化"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Autoregressively roll current EventFinetune models to 24h and export "
            "compact comparison CSV files. The current model architecture expects "
            "future event slices, so rollout uses the recorded future events."
        )
    )
    parser.add_argument(
        "--base-root",
        type=str,
        default=str(REPO_ROOT / "data_event" / "event_finetune_outputs"),
    )
    parser.add_argument("--summary-paths", type=str, nargs="*", default=None)
    parser.add_argument("--tasks", type=str, nargs="*", default=None)
    parser.add_argument("--run-tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "all"],
        default="all",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        choices=["auto", "best", "final"],
        default="auto",
    )
    parser.add_argument(
        "--rollout-horizon-steps",
        type=int,
        default=288,
        help="Default 288 = 24h for 5-minute CGM.",
    )
    parser.add_argument(
        "--allow-partial-horizon",
        action="store_true",
        help="Keep windows near the end even when the full 24h target is unavailable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=256,
        help="Number of windows to autoregress in parallel on the selected device.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Default: <run_tag>_event_compare_compact_rollout24_<split>.csv",
    )
    parser.add_argument(
        "--metrics-output-name",
        type=str,
        default=None,
        help="Default: <run_tag>_event_compare_compact_rollout24_<split>_metrics.csv",
    )
    return parser


def should_keep_summary(
    summary_path: Path,
    summary_payload: dict,
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


def input_event_label(offset: int, patch_size: int) -> str:
    rel = patch_size - offset - 1
    return "t+0" if rel == 0 else f"t-{rel}"


def build_main_columns(config: dict, rollout_horizon_steps: int) -> list[str]:
    patch_size = int(config["patch_size"])
    columns = [
        COL_SAMPLE_ID,
        COL_DEVICE_ID,
        COL_WINDOW_INDEX,
        COL_WINDOW_START,
    ]
    columns.extend([f"输入点{idx + 1}" for idx in range(patch_size)])
    columns.extend(
        [
            COL_INPUT_START_TIME,
            COL_FORECAST_START_TIME,
            COL_INPUT_EVENT_COUNT,
            COL_FORECAST_EVENT_COUNT,
            COL_ALL_EVENT_SUMMARY,
            COL_ROW_TYPE,
        ]
    )
    columns.extend([f"t+{step + 1}" for step in range(rollout_horizon_steps)])
    columns.extend([COL_MAE, COL_RMSE, COL_BIAS, COL_MAE_DELTA])
    columns.extend(
        [f"输入事件_{input_event_label(idx, patch_size)}" for idx in range(patch_size)]
    )
    columns.extend([f"预测事件_t+{step + 1}" for step in range(rollout_horizon_steps)])
    return columns


def build_metrics_columns() -> list[str]:
    return [
        "run_tag",
        "split",
        METRIC_SCOPE,
        METRIC_NAME,
        METRIC_VALUE,
        METRIC_VALID_POINTS,
        "checkpoint_kind",
        "rollout_horizon_steps",
        "chunk_steps",
        METRIC_PRE_MAE,
        METRIC_POST_MAE,
        COL_MAE_DELTA,
        METRIC_PRE_RMSE,
        METRIC_POST_RMSE,
        METRIC_RMSE_DELTA,
        METRIC_PRE_BIAS,
        METRIC_POST_BIAS,
        METRIC_BIAS_DELTA,
    ]


def format_event_content(row_values: np.ndarray, event_cols: list[str]) -> str:
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
        rounded = helpers.maybe_round_float(value)
        parts.append(f"{event_name}={rounded}")
    return "; ".join(parts)


def slice_and_pad_1d(values: np.ndarray, start: int, length: int, pad_value=np.nan):
    out = np.full(length, pad_value, dtype=np.float32)
    if start >= len(values):
        return out
    stop = min(start + length, len(values))
    chunk = np.asarray(values[start:stop], dtype=np.float32)
    out[: len(chunk)] = chunk
    return out


def slice_and_pad_2d(values: np.ndarray, start: int, length: int) -> np.ndarray:
    out = np.zeros((length, values.shape[1]), dtype=np.float32)
    if start >= len(values):
        return out
    stop = min(start + length, len(values))
    chunk = np.asarray(values[start:stop], dtype=np.float32)
    out[: len(chunk)] = chunk
    return out


def build_target_window(
    raw_glucose: np.ndarray,
    target_start: int,
    rollout_horizon_steps: int,
):
    target_values = slice_and_pad_1d(raw_glucose, target_start, rollout_horizon_steps)
    target_mask = ~np.isnan(target_values)
    target_filled = target_values.copy()
    target_filled[~target_mask] = 0.0
    return target_filled.astype(np.float32), target_mask.astype(np.float32)


def build_event_timeline(
    full_events: np.ndarray,
    event_cols: list[str],
    sample_id: str,
    device_id: str,
    time_meta: dict,
    event_content_cache: dict[int, str],
    time_text_cache: dict[int, str],
    start: int,
    length: int,
    patch_size: int,
    is_forecast: bool,
):
    cells: list[str | None] = []
    summary_parts: list[str] = []
    count = 0
    for offset in range(length):
        abs_index = int(start + offset)
        if abs_index >= full_events.shape[0]:
            cells.append(None)
            continue
        content = event_content_cache.get(abs_index)
        if content is None:
            content = format_event_content(
                np.asarray(full_events[abs_index]), event_cols
            )
            event_content_cache[abs_index] = content
        if not content:
            cells.append(None)
            continue

        cells.append(content)
        count += 1
        if is_forecast:
            rel_label = f"t+{offset + 1}"
        else:
            rel_label = input_event_label(offset, patch_size)
        time_text = time_text_cache.get(abs_index)
        if time_text is None:
            time_text = (
                helpers.resolve_time_info(time_meta, sample_id, device_id, abs_index)[
                    "absolute_time"
                ]
                or f"index={abs_index}"
            )
            time_text_cache[abs_index] = time_text
        summary_parts.append(f"{rel_label}@{time_text}:{content}")
    return cells, count, summary_parts


def fill_forecast_columns(
    row: dict,
    values: np.ndarray,
    mask: np.ndarray,
    rollout_horizon_steps: int,
):
    valid = mask.astype(bool)
    for idx in range(rollout_horizon_steps):
        row[f"t+{idx + 1}"] = (
            helpers.maybe_round_float(values[idx]) if valid[idx] else None
        )


def fill_event_columns(
    row: dict,
    input_event_cells: list[str | None],
    forecast_event_cells: list[str | None],
    patch_size: int,
):
    for idx, value in enumerate(input_event_cells):
        row[f"输入事件_{input_event_label(idx, patch_size)}"] = value
    for idx, value in enumerate(forecast_event_cells):
        row[f"预测事件_t+{idx + 1}"] = value


def build_kept_patch_starts(
    dataset,
    dataset_index: int,
    patch_size: int,
    forecast_steps: int,
) -> list[int]:
    # Keep the exact dense patch grid produced by EventDataset.__getitem__.
    # Re-filtering starts here breaks alignment with raw_sample["patches"] after
    # the dataset switched from sparse kept patches to a dense patch grid.
    raw_glucose = dataset.glucose[dataset_index]
    max_len = len(raw_glucose)
    return [
        int(start)
        for start in dataset.device_patch_starts[dataset_index]
        if int(start) + patch_size + forecast_steps <= max_len
    ]


def rollout_base_model_batch(
    base_model,
    initial_patches: np.ndarray,
    patch_starts: np.ndarray,
    chunk_steps: int,
    rollout_horizon_steps: int,
    model_stride: int,
    device: torch.device,
) -> np.ndarray:
    batch_size = int(initial_patches.shape[0])
    preds = torch.empty(
        (batch_size, rollout_horizon_steps),
        dtype=torch.float32,
        device=device,
    )
    # Keep the autoregressive state on-device so each rollout step avoids host/device copies.
    current_patch = (
        torch.as_tensor(initial_patches, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .contiguous()
    )
    patch_start_tensor = torch.as_tensor(
        patch_starts,
        dtype=torch.long,
        device=device,
    ).contiguous()
    offset = 0
    base_model.eval()
    with torch.inference_mode():
        while offset < rollout_horizon_steps:
            chunk = (
                base_model(
                    current_patch,
                    start_idx=patch_start_tensor + offset,
                    stride=model_stride,
                )
                .squeeze(1)
                .to(dtype=torch.float32)
            )
            take = min(chunk_steps, rollout_horizon_steps - offset)
            preds[:, offset : offset + take].copy_(chunk[:, :take])
            offset += take
            if offset >= rollout_horizon_steps:
                break
            current_patch = torch.cat(
                [current_patch[..., take:], chunk[:, :take].unsqueeze(1)],
                dim=-1,
            )
    return preds.cpu().numpy()


def rollout_event_model_batch(
    model,
    initial_patches: np.ndarray,
    full_events: torch.Tensor,
    patch_starts: np.ndarray,
    patch_size: int,
    chunk_steps: int,
    rollout_horizon_steps: int,
    model_stride: int,
    device: torch.device,
) -> np.ndarray:
    batch_size = int(initial_patches.shape[0])
    preds = torch.empty(
        (batch_size, rollout_horizon_steps),
        dtype=torch.float32,
        device=device,
    )
    current_patch = (
        torch.as_tensor(initial_patches, dtype=torch.float32, device=device)
        .unsqueeze(1)
        .contiguous()
    )
    patch_start_tensor = torch.as_tensor(
        patch_starts,
        dtype=torch.long,
        device=device,
    ).contiguous()
    event_window_len = patch_size + chunk_steps
    event_offsets = torch.arange(
        event_window_len,
        dtype=torch.long,
        device=device,
    ).view(1, -1)
    offset = 0
    model.eval()
    with torch.inference_mode():
        while offset < rollout_horizon_steps:
            event_start = patch_start_tensor + offset
            event_tensor = full_events[
                event_start.view(-1, 1) + event_offsets
            ].unsqueeze(1)
            chunk = (
                model(
                    current_patch,
                    event_tensor,
                    start_idx=event_start,
                    stride=model_stride,
                )
                .squeeze(1)
                .to(dtype=torch.float32)
            )
            take = min(chunk_steps, rollout_horizon_steps - offset)
            preds[:, offset : offset + take].copy_(chunk[:, :take])
            offset += take
            if offset >= rollout_horizon_steps:
                break
            current_patch = torch.cat(
                [current_patch[..., take:], chunk[:, :take].unsqueeze(1)],
                dim=-1,
            )
    return preds.cpu().numpy()


def init_metric_state(rollout_horizon_steps: int) -> dict:
    zeros_f = np.zeros(rollout_horizon_steps, dtype=np.float64)
    zeros_i = np.zeros(rollout_horizon_steps, dtype=np.int64)
    return {
        "rollout_horizon_steps": rollout_horizon_steps,
        "overall_count": 0,
        "pre_abs_sum": 0.0,
        "pre_sq_sum": 0.0,
        "pre_bias_sum": 0.0,
        "post_abs_sum": 0.0,
        "post_sq_sum": 0.0,
        "post_bias_sum": 0.0,
        "horizon_count": zeros_i.copy(),
        "pre_horizon_abs_sum": zeros_f.copy(),
        "pre_horizon_sq_sum": zeros_f.copy(),
        "pre_horizon_bias_sum": zeros_f.copy(),
        "post_horizon_abs_sum": zeros_f.copy(),
        "post_horizon_sq_sum": zeros_f.copy(),
        "post_horizon_bias_sum": zeros_f.copy(),
    }


def accumulate_metrics(
    state: dict,
    pre_preds: np.ndarray,
    post_preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
):
    valid = mask.astype(bool)
    if not np.any(valid):
        return

    diff_pre = pre_preds - targets
    diff_post = post_preds - targets

    state["overall_count"] += int(valid.sum())
    state["pre_abs_sum"] += float(np.abs(diff_pre[valid]).sum())
    state["pre_sq_sum"] += float((diff_pre[valid] ** 2).sum())
    state["pre_bias_sum"] += float(diff_pre[valid].sum())
    state["post_abs_sum"] += float(np.abs(diff_post[valid]).sum())
    state["post_sq_sum"] += float((diff_post[valid] ** 2).sum())
    state["post_bias_sum"] += float(diff_post[valid].sum())

    valid_i = valid.astype(np.int64)
    state["horizon_count"] += valid_i
    state["pre_horizon_abs_sum"] += np.where(valid, np.abs(diff_pre), 0.0)
    state["pre_horizon_sq_sum"] += np.where(valid, diff_pre**2, 0.0)
    state["pre_horizon_bias_sum"] += np.where(valid, diff_pre, 0.0)
    state["post_horizon_abs_sum"] += np.where(valid, np.abs(diff_post), 0.0)
    state["post_horizon_sq_sum"] += np.where(valid, diff_post**2, 0.0)
    state["post_horizon_bias_sum"] += np.where(valid, diff_post, 0.0)


def safe_mean(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(total) / float(count)


def safe_rmse(total_sq: float, count: int) -> float | None:
    if count <= 0:
        return None
    return float(np.sqrt(float(total_sq) / float(count)))


def maybe_delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return after - before


def build_metrics_rows(
    run_tag: str,
    split_name: str,
    checkpoint_kind: str,
    rollout_horizon_steps: int,
    chunk_steps: int,
    metric_state: dict,
    exported_windows: int,
    valid_forecast_points: int,
) -> list[dict]:
    rows: list[dict] = []
    rows.append(
        {
            "run_tag": run_tag,
            "split": split_name,
            METRIC_SCOPE: "summary",
            METRIC_NAME: "exported_windows",
            METRIC_VALUE: exported_windows,
            "rollout_horizon_steps": rollout_horizon_steps,
            "chunk_steps": chunk_steps,
        }
    )
    rows.append(
        {
            "run_tag": run_tag,
            "split": split_name,
            METRIC_SCOPE: "summary",
            METRIC_NAME: "valid_forecast_points",
            METRIC_VALUE: valid_forecast_points,
            "rollout_horizon_steps": rollout_horizon_steps,
            "chunk_steps": chunk_steps,
        }
    )

    overall_count = int(metric_state["overall_count"])
    pre_mae = safe_mean(metric_state["pre_abs_sum"], overall_count)
    post_mae = safe_mean(metric_state["post_abs_sum"], overall_count)
    pre_rmse = safe_rmse(metric_state["pre_sq_sum"], overall_count)
    post_rmse = safe_rmse(metric_state["post_sq_sum"], overall_count)
    pre_bias = safe_mean(metric_state["pre_bias_sum"], overall_count)
    post_bias = safe_mean(metric_state["post_bias_sum"], overall_count)

    rows.append(
        {
            "run_tag": run_tag,
            "split": split_name,
            METRIC_SCOPE: "overall",
            METRIC_NAME: "all",
            METRIC_VALID_POINTS: overall_count,
            "checkpoint_kind": checkpoint_kind,
            "rollout_horizon_steps": rollout_horizon_steps,
            "chunk_steps": chunk_steps,
            METRIC_PRE_MAE: helpers.maybe_round_float(pre_mae),
            METRIC_POST_MAE: helpers.maybe_round_float(post_mae),
            COL_MAE_DELTA: helpers.maybe_round_float(maybe_delta(post_mae, pre_mae)),
            METRIC_PRE_RMSE: helpers.maybe_round_float(pre_rmse),
            METRIC_POST_RMSE: helpers.maybe_round_float(post_rmse),
            METRIC_RMSE_DELTA: helpers.maybe_round_float(
                maybe_delta(post_rmse, pre_rmse)
            ),
            METRIC_PRE_BIAS: helpers.maybe_round_float(pre_bias),
            METRIC_POST_BIAS: helpers.maybe_round_float(post_bias),
            METRIC_BIAS_DELTA: helpers.maybe_round_float(
                maybe_delta(post_bias, pre_bias)
            ),
        }
    )

    for step_idx in range(metric_state["rollout_horizon_steps"]):
        count = int(metric_state["horizon_count"][step_idx])
        pre_step_mae = safe_mean(metric_state["pre_horizon_abs_sum"][step_idx], count)
        post_step_mae = safe_mean(metric_state["post_horizon_abs_sum"][step_idx], count)
        pre_step_rmse = safe_rmse(metric_state["pre_horizon_sq_sum"][step_idx], count)
        post_step_rmse = safe_rmse(metric_state["post_horizon_sq_sum"][step_idx], count)
        pre_step_bias = safe_mean(metric_state["pre_horizon_bias_sum"][step_idx], count)
        post_step_bias = safe_mean(
            metric_state["post_horizon_bias_sum"][step_idx], count
        )

        rows.append(
            {
                "run_tag": run_tag,
                "split": split_name,
                METRIC_SCOPE: "horizon",
                METRIC_NAME: f"t+{step_idx + 1}",
                METRIC_VALID_POINTS: count,
                "checkpoint_kind": checkpoint_kind,
                "rollout_horizon_steps": rollout_horizon_steps,
                "chunk_steps": chunk_steps,
                METRIC_PRE_MAE: helpers.maybe_round_float(pre_step_mae),
                METRIC_POST_MAE: helpers.maybe_round_float(post_step_mae),
                COL_MAE_DELTA: helpers.maybe_round_float(
                    maybe_delta(post_step_mae, pre_step_mae)
                ),
                METRIC_PRE_RMSE: helpers.maybe_round_float(pre_step_rmse),
                METRIC_POST_RMSE: helpers.maybe_round_float(post_step_rmse),
                METRIC_RMSE_DELTA: helpers.maybe_round_float(
                    maybe_delta(post_step_rmse, pre_step_rmse)
                ),
                METRIC_PRE_BIAS: helpers.maybe_round_float(pre_step_bias),
                METRIC_POST_BIAS: helpers.maybe_round_float(post_step_bias),
                METRIC_BIAS_DELTA: helpers.maybe_round_float(
                    maybe_delta(post_step_bias, pre_step_bias)
                ),
            }
        )

    return rows


def has_any_event(full_events: np.ndarray, start: int, length: int) -> bool:
    if start >= len(full_events):
        return False
    stop = min(start + length, len(full_events))
    return bool(np.any(np.abs(full_events[start:stop]) > 0))


def choose_representative_patch_index(
    starts: list[int],
    event_index: int,
    patch_size: int,
    forecast_steps: int,
) -> int | None:
    best_score = None
    best_patch_index = None

    input_idx = bisect_right(starts, event_index) - 1
    if input_idx >= 0:
        input_start = int(starts[input_idx])
        if event_index - patch_size + 1 <= input_start <= event_index:
            target_start = input_start + patch_size
            best_score = (0, target_start - event_index, input_idx)
            best_patch_index = input_idx

    forecast_idx = bisect_right(starts, event_index - patch_size) - 1
    if forecast_idx >= 0:
        forecast_start = int(starts[forecast_idx])
        min_forecast_start = event_index - patch_size - forecast_steps + 1
        max_forecast_start = event_index - patch_size
        if min_forecast_start <= forecast_start <= max_forecast_start:
            target_start = forecast_start + patch_size
            forecast_score = (1, event_index - target_start, forecast_idx)
            if best_score is None or forecast_score < best_score:
                best_score = forecast_score
                best_patch_index = forecast_idx

    return best_patch_index


def build_selected_patch_indices(
    full_events: np.ndarray,
    kept_starts: list[int],
    patch_size: int,
    rollout_horizon_steps: int,
) -> list[int]:
    nonzero_event_indices = np.flatnonzero(
        np.any(np.abs(full_events) > 0, axis=1)
    ).tolist()
    if not nonzero_event_indices:
        return []

    selected_patch_indices: set[int] = set()
    for event_abs_index in nonzero_event_indices:
        representative_patch_index = choose_representative_patch_index(
            starts=kept_starts,
            event_index=int(event_abs_index),
            patch_size=patch_size,
            forecast_steps=rollout_horizon_steps,
        )
        if representative_patch_index is not None:
            selected_patch_indices.add(int(representative_patch_index))
    return sorted(selected_patch_indices)


def export_one_run(
    summary_path: Path,
    split_name: str,
    checkpoint_kind: str,
    rollout_horizon_steps: int,
    allow_partial_horizon: bool,
    device: torch.device,
    rollout_batch_size: int,
    output_name: str | None,
    metrics_output_name: str | None,
):
    summary = helpers.load_json(summary_path)
    config = helpers.resolve_training_config(summary, summary_path)
    run_tag = str(summary.get("run_tag", summary_path.stem.replace("_summary", "")))

    raw_dataset = event_main.build_dataset(config)
    model_dataset = event_main.build_dataset(config)
    selected_indices, split_map = helpers.resolve_selected_indices(
        dataset=model_dataset,
        config=config,
        summary=summary,
        split_name=split_name,
    )
    helpers.apply_saved_event_stats(
        dataset=model_dataset,
        summary=summary,
        all_indices=list(split_map["all"]),
    )

    checkpoint_path, resolved_ckpt_kind = helpers.resolve_checkpoint_path(
        summary=summary,
        summary_path=summary_path,
        checkpoint_kind=checkpoint_kind,
    )
    time_meta = helpers.load_time_metadata(Path(str(config["data_csv"])))

    if device.type == "cuda":
        torch.cuda.set_device(device)

    pre_model = helpers.build_models(config, device)
    pre_base = pre_model.base
    del pre_model

    post_model = helpers.build_models(config, device)
    helpers.load_post_model_state(post_model, checkpoint_path)
    post_model.eval()
    post_model.base.eval()

    event_cols = list(config["event_cols"])
    patch_size = int(config["patch_size"])
    chunk_steps = int(config["forecast_steps"])
    model_stride = int(config["model_stride"])
    if chunk_steps <= 0:
        raise ValueError("forecast_steps must be positive for autoregressive rollout.")
    if rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive.")

    output_dir = summary_path.parent
    main_csv_name = (
        output_name
        if output_name
        else f"{run_tag}_event_compare_compact_rollout24_{split_name}.csv"
    )
    metrics_csv_name = (
        metrics_output_name
        if metrics_output_name
        else f"{run_tag}_event_compare_compact_rollout24_{split_name}_metrics.csv"
    )
    main_csv_path = output_dir / main_csv_name
    metrics_csv_path = output_dir / metrics_csv_name
    main_columns = build_main_columns(config, rollout_horizon_steps)
    input_value_columns = main_columns[4 : 4 + patch_size]

    metric_state = init_metric_state(rollout_horizon_steps)
    exported_windows = 0
    valid_forecast_points_total = 0

    with main_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_columns)
        writer.writeheader()

        for device_order, dataset_index in enumerate(selected_indices, start=1):
            raw_sample = raw_dataset[dataset_index]
            device_id = str(raw_sample["device_id"])
            sample_id = str(raw_sample["id"])
            raw_glucose = np.asarray(
                raw_dataset.glucose[dataset_index], dtype=np.float32
            )
            raw_events = raw_dataset.events_map.get(device_id)
            if raw_events is None:
                raw_events = np.zeros(
                    (raw_dataset.L, len(event_cols)), dtype=np.float32
                )
            raw_events = np.asarray(raw_events, dtype=np.float32)

            model_events = model_dataset.events_map.get(device_id)
            if model_events is None:
                model_events = np.zeros(
                    (model_dataset.L, len(event_cols)), dtype=np.float32
                )
            model_events = np.asarray(model_events, dtype=np.float32)
            model_events = fill_nan_linear(model_events.copy())
            model_events_tensor = torch.as_tensor(
                model_events,
                dtype=torch.float32,
                device=device,
            )
            model_events_tensor = torch.cat(
                [
                    model_events_tensor,
                    torch.zeros(
                        (patch_size + chunk_steps, model_events_tensor.shape[1]),
                        dtype=torch.float32,
                        device=device,
                    ),
                ],
                dim=0,
            )
            # Overlapping windows hit the same event rows and timestamps repeatedly.
            event_content_cache: dict[int, str] = {}
            time_text_cache: dict[int, str] = {}

            kept_starts = build_kept_patch_starts(
                raw_dataset,
                dataset_index,
                patch_size,
                chunk_steps,
            )
            if len(kept_starts) != raw_sample["patches"].shape[0]:
                raise RuntimeError(
                    f"Patch start count mismatch for device_id={device_id}: "
                    f"kept_starts={len(kept_starts)} sample_patches={raw_sample['patches'].shape[0]}"
                )

            valid_len = int(raw_dataset.valid_lengths[dataset_index])
            selected_patch_indices = build_selected_patch_indices(
                full_events=raw_events,
                kept_starts=kept_starts,
                patch_size=patch_size,
                rollout_horizon_steps=rollout_horizon_steps,
            )
            print(
                f"[INFO] run_tag={run_tag} device={device_order}/{len(selected_indices)} "
                f"sample_id={sample_id} selected_windows={len(selected_patch_indices)}",
                flush=True,
            )

            window_items: list[dict] = []
            for patch_index in selected_patch_indices:
                patch_start = int(kept_starts[patch_index])
                target_start = int(patch_start + patch_size)
                target_end = int(target_start + rollout_horizon_steps)
                if not allow_partial_horizon and target_end > valid_len:
                    continue

                target_values, target_mask = build_target_window(
                    raw_glucose,
                    target_start,
                    rollout_horizon_steps,
                )
                if (
                    not allow_partial_horizon
                    and int(target_mask.sum()) < rollout_horizon_steps
                ):
                    continue
                if int(target_mask.sum()) == 0:
                    continue

                window_items.append(
                    {
                        "patch_index": int(patch_index),
                        "patch_start": int(patch_start),
                        "target_start": int(target_start),
                        "initial_patch": np.asarray(
                            raw_sample["patches"][patch_index],
                            dtype=np.float32,
                        ),
                        "target_values": target_values,
                        "target_mask": target_mask,
                        "target_mask_bool": target_mask.astype(bool),
                    }
                )

            for batch_start in range(0, len(window_items), rollout_batch_size):
                batch_items = window_items[
                    batch_start : batch_start + rollout_batch_size
                ]
                batch_initial_patches = np.stack(
                    [item["initial_patch"] for item in batch_items],
                    axis=0,
                )
                batch_patch_starts = np.asarray(
                    [item["patch_start"] for item in batch_items],
                    dtype=np.int64,
                )
                pre_rollouts = rollout_base_model_batch(
                    base_model=pre_base,
                    initial_patches=batch_initial_patches,
                    patch_starts=batch_patch_starts,
                    chunk_steps=chunk_steps,
                    rollout_horizon_steps=rollout_horizon_steps,
                    model_stride=model_stride,
                    device=device,
                )
                post_rollouts = rollout_event_model_batch(
                    model=post_model,
                    initial_patches=batch_initial_patches,
                    full_events=model_events_tensor,
                    patch_starts=batch_patch_starts,
                    patch_size=patch_size,
                    chunk_steps=chunk_steps,
                    rollout_horizon_steps=rollout_horizon_steps,
                    model_stride=model_stride,
                    device=device,
                )

                for item, pre_rollout, post_rollout in zip(
                    batch_items,
                    pre_rollouts,
                    post_rollouts,
                ):
                    patch_index = int(item["patch_index"])
                    patch_start = int(item["patch_start"])
                    target_start = int(item["target_start"])
                    initial_patch = item["initial_patch"]
                    target_values = item["target_values"]
                    target_mask = item["target_mask"]
                    target_mask_bool = item["target_mask_bool"]

                    accumulate_metrics(
                        metric_state,
                        pre_preds=pre_rollout,
                        post_preds=post_rollout,
                        targets=target_values,
                        mask=target_mask,
                    )
                    valid_forecast_points_total += int(target_mask_bool.sum())

                    input_values = [
                        helpers.maybe_round_float(value)
                        for value in initial_patch.tolist()
                    ]
                    input_start_time = time_text_cache.get(int(patch_start))
                    if input_start_time is None:
                        input_start_time = (
                            helpers.resolve_time_info(
                                time_meta, sample_id, device_id, int(patch_start)
                            )["absolute_time"]
                            or f"index={int(patch_start)}"
                        )
                        time_text_cache[int(patch_start)] = input_start_time
                    forecast_start_time = time_text_cache.get(int(target_start))
                    if forecast_start_time is None:
                        forecast_start_time = (
                            helpers.resolve_time_info(
                                time_meta, sample_id, device_id, int(target_start)
                            )["absolute_time"]
                            or f"index={int(target_start)}"
                        )
                        time_text_cache[int(target_start)] = forecast_start_time

                    input_event_cells, input_event_count, input_summary_parts = (
                        build_event_timeline(
                            full_events=raw_events,
                            event_cols=event_cols,
                            sample_id=sample_id,
                            device_id=device_id,
                            time_meta=time_meta,
                            event_content_cache=event_content_cache,
                            time_text_cache=time_text_cache,
                            start=int(patch_start),
                            length=patch_size,
                            patch_size=patch_size,
                            is_forecast=False,
                        )
                    )
                    (
                        forecast_event_cells,
                        forecast_event_count,
                        forecast_summary_parts,
                    ) = build_event_timeline(
                        full_events=raw_events,
                        event_cols=event_cols,
                        sample_id=sample_id,
                        device_id=device_id,
                        time_meta=time_meta,
                        event_content_cache=event_content_cache,
                        time_text_cache=time_text_cache,
                        start=int(target_start),
                        length=rollout_horizon_steps,
                        patch_size=patch_size,
                        is_forecast=True,
                    )
                    all_event_summary = " | ".join(
                        input_summary_parts + forecast_summary_parts
                    )

                    pre_metrics = helpers.build_patch_metric_dict(
                        pre_rollout,
                        target_values,
                        target_mask_bool,
                    )
                    post_metrics = helpers.build_patch_metric_dict(
                        post_rollout,
                        target_values,
                        target_mask_bool,
                    )

                    shared = {
                        COL_SAMPLE_ID: sample_id,
                        COL_DEVICE_ID: device_id,
                        COL_WINDOW_INDEX: int(patch_index),
                        COL_WINDOW_START: int(patch_start),
                        COL_INPUT_START_TIME: input_start_time,
                        COL_FORECAST_START_TIME: forecast_start_time,
                        COL_INPUT_EVENT_COUNT: int(input_event_count),
                        COL_FORECAST_EVENT_COUNT: int(forecast_event_count),
                        COL_ALL_EVENT_SUMMARY: all_event_summary,
                    }
                    for column_name, value in zip(input_value_columns, input_values):
                        shared[column_name] = value

                    true_row = dict(shared)
                    true_row[COL_ROW_TYPE] = ROW_TRUE
                    true_row[COL_MAE] = None
                    true_row[COL_RMSE] = None
                    true_row[COL_BIAS] = None
                    true_row[COL_MAE_DELTA] = None
                    fill_forecast_columns(
                        true_row,
                        target_values,
                        target_mask,
                        rollout_horizon_steps,
                    )
                    fill_event_columns(
                        true_row,
                        input_event_cells,
                        forecast_event_cells,
                        patch_size,
                    )
                    writer.writerow(true_row)

                    pre_row = dict(shared)
                    pre_row[COL_ROW_TYPE] = ROW_PRE
                    pre_row[COL_MAE] = helpers.maybe_round_float(pre_metrics["mae"])
                    pre_row[COL_RMSE] = helpers.maybe_round_float(pre_metrics["rmse"])
                    pre_row[COL_BIAS] = helpers.maybe_round_float(pre_metrics["bias"])
                    pre_row[COL_MAE_DELTA] = None
                    fill_forecast_columns(
                        pre_row,
                        pre_rollout,
                        target_mask,
                        rollout_horizon_steps,
                    )
                    fill_event_columns(
                        pre_row,
                        input_event_cells,
                        forecast_event_cells,
                        patch_size,
                    )
                    writer.writerow(pre_row)

                    post_row = dict(shared)
                    post_row[COL_ROW_TYPE] = ROW_POST
                    post_row[COL_MAE] = helpers.maybe_round_float(post_metrics["mae"])
                    post_row[COL_RMSE] = helpers.maybe_round_float(post_metrics["rmse"])
                    post_row[COL_BIAS] = helpers.maybe_round_float(post_metrics["bias"])
                    post_row[COL_MAE_DELTA] = helpers.maybe_round_float(
                        maybe_delta(post_metrics["mae"], pre_metrics["mae"])
                    )
                    fill_forecast_columns(
                        post_row,
                        post_rollout,
                        target_mask,
                        rollout_horizon_steps,
                    )
                    fill_event_columns(
                        post_row,
                        input_event_cells,
                        forecast_event_cells,
                        patch_size,
                    )
                    writer.writerow(post_row)
                    exported_windows += 1

            print(
                f"[INFO] run_tag={run_tag} finished_device={device_order}/{len(selected_indices)} "
                f"exported_windows_total={exported_windows}",
                flush=True,
            )
            continue

            for patch_index in selected_patch_indices:
                patch_start = int(kept_starts[patch_index])
                target_start = int(patch_start + patch_size)
                target_end = int(target_start + rollout_horizon_steps)
                if not allow_partial_horizon and target_end > valid_len:
                    continue

                target_values, target_mask = build_target_window(
                    raw_glucose,
                    target_start,
                    rollout_horizon_steps,
                )
                if (
                    not allow_partial_horizon
                    and int(target_mask.sum()) < rollout_horizon_steps
                ):
                    continue
                if int(target_mask.sum()) == 0:
                    continue
                target_mask_bool = target_mask.astype(bool)

                initial_patch = np.asarray(
                    raw_sample["patches"][patch_index], dtype=np.float32
                )
                pre_rollout = rollout_base_model(
                    base_model=pre_base,
                    initial_patch=initial_patch,
                    patch_start=int(patch_start),
                    chunk_steps=chunk_steps,
                    rollout_horizon_steps=rollout_horizon_steps,
                    model_stride=model_stride,
                    device=device,
                )
                post_rollout = rollout_event_model(
                    model=post_model,
                    initial_patch=initial_patch,
                    full_events=model_events_tensor,
                    patch_start=int(patch_start),
                    patch_size=patch_size,
                    chunk_steps=chunk_steps,
                    rollout_horizon_steps=rollout_horizon_steps,
                    model_stride=model_stride,
                    device=device,
                )

                accumulate_metrics(
                    metric_state,
                    pre_preds=pre_rollout,
                    post_preds=post_rollout,
                    targets=target_values,
                    mask=target_mask,
                )
                valid_forecast_points_total += int(target_mask_bool.sum())

                input_values = [
                    helpers.maybe_round_float(value) for value in initial_patch.tolist()
                ]
                input_start_time = time_text_cache.get(int(patch_start))
                if input_start_time is None:
                    input_start_time = (
                        helpers.resolve_time_info(
                            time_meta, sample_id, device_id, int(patch_start)
                        )["absolute_time"]
                        or f"index={int(patch_start)}"
                    )
                    time_text_cache[int(patch_start)] = input_start_time
                forecast_start_time = time_text_cache.get(int(target_start))
                if forecast_start_time is None:
                    forecast_start_time = (
                        helpers.resolve_time_info(
                            time_meta, sample_id, device_id, int(target_start)
                        )["absolute_time"]
                        or f"index={int(target_start)}"
                    )
                    time_text_cache[int(target_start)] = forecast_start_time

                input_event_cells, input_event_count, input_summary_parts = (
                    build_event_timeline(
                        full_events=raw_events,
                        event_cols=event_cols,
                        sample_id=sample_id,
                        device_id=device_id,
                        time_meta=time_meta,
                        event_content_cache=event_content_cache,
                        time_text_cache=time_text_cache,
                        start=int(patch_start),
                        length=patch_size,
                        patch_size=patch_size,
                        is_forecast=False,
                    )
                )
                forecast_event_cells, forecast_event_count, forecast_summary_parts = (
                    build_event_timeline(
                        full_events=raw_events,
                        event_cols=event_cols,
                        sample_id=sample_id,
                        device_id=device_id,
                        time_meta=time_meta,
                        event_content_cache=event_content_cache,
                        time_text_cache=time_text_cache,
                        start=int(target_start),
                        length=rollout_horizon_steps,
                        patch_size=patch_size,
                        is_forecast=True,
                    )
                )
                all_event_summary = " | ".join(
                    input_summary_parts + forecast_summary_parts
                )

                pre_metrics = helpers.build_patch_metric_dict(
                    pre_rollout,
                    target_values,
                    target_mask_bool,
                )
                post_metrics = helpers.build_patch_metric_dict(
                    post_rollout,
                    target_values,
                    target_mask_bool,
                )

                shared = {
                    COL_SAMPLE_ID: sample_id,
                    COL_DEVICE_ID: device_id,
                    COL_WINDOW_INDEX: int(patch_index),
                    COL_WINDOW_START: int(patch_start),
                    COL_INPUT_START_TIME: input_start_time,
                    COL_FORECAST_START_TIME: forecast_start_time,
                    COL_INPUT_EVENT_COUNT: int(input_event_count),
                    COL_FORECAST_EVENT_COUNT: int(forecast_event_count),
                    COL_ALL_EVENT_SUMMARY: all_event_summary,
                }
                for idx, value in enumerate(input_values, start=1):
                    shared[f"输入点{idx}"] = value

                true_row = dict(shared)
                true_row[COL_ROW_TYPE] = ROW_TRUE
                true_row[COL_MAE] = None
                true_row[COL_RMSE] = None
                true_row[COL_BIAS] = None
                true_row[COL_MAE_DELTA] = None
                fill_forecast_columns(
                    true_row,
                    target_values,
                    target_mask,
                    rollout_horizon_steps,
                )
                fill_event_columns(
                    true_row,
                    input_event_cells,
                    forecast_event_cells,
                    patch_size,
                )
                writer.writerow(true_row)

                pre_row = dict(shared)
                pre_row[COL_ROW_TYPE] = ROW_PRE
                pre_row[COL_MAE] = helpers.maybe_round_float(pre_metrics["mae"])
                pre_row[COL_RMSE] = helpers.maybe_round_float(pre_metrics["rmse"])
                pre_row[COL_BIAS] = helpers.maybe_round_float(pre_metrics["bias"])
                pre_row[COL_MAE_DELTA] = None
                fill_forecast_columns(
                    pre_row,
                    pre_rollout,
                    target_mask,
                    rollout_horizon_steps,
                )
                fill_event_columns(
                    pre_row,
                    input_event_cells,
                    forecast_event_cells,
                    patch_size,
                )
                writer.writerow(pre_row)

                post_row = dict(shared)
                post_row[COL_ROW_TYPE] = ROW_POST
                post_row[COL_MAE] = helpers.maybe_round_float(post_metrics["mae"])
                post_row[COL_RMSE] = helpers.maybe_round_float(post_metrics["rmse"])
                post_row[COL_BIAS] = helpers.maybe_round_float(post_metrics["bias"])
                post_row[COL_MAE_DELTA] = helpers.maybe_round_float(
                    maybe_delta(post_metrics["mae"], pre_metrics["mae"])
                )
                fill_forecast_columns(
                    post_row,
                    post_rollout,
                    target_mask,
                    rollout_horizon_steps,
                )
                fill_event_columns(
                    post_row,
                    input_event_cells,
                    forecast_event_cells,
                    patch_size,
                )
                writer.writerow(post_row)
                exported_windows += 1
            print(
                f"[INFO] run_tag={run_tag} finished_device={device_order}/{len(selected_indices)} "
                f"exported_windows_total={exported_windows}",
                flush=True,
            )

    metrics_rows = build_metrics_rows(
        run_tag=run_tag,
        split_name=split_name,
        checkpoint_kind=resolved_ckpt_kind,
        rollout_horizon_steps=rollout_horizon_steps,
        chunk_steps=chunk_steps,
        metric_state=metric_state,
        exported_windows=exported_windows,
        valid_forecast_points=valid_forecast_points_total,
    )
    with metrics_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=build_metrics_columns())
        writer.writeheader()
        writer.writerows(metrics_rows)

    del pre_base
    del post_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    return {
        "summary_path": str(summary_path),
        "main_csv_path": str(main_csv_path),
        "metrics_csv_path": str(metrics_csv_path),
        "run_tag": run_tag,
        "split": split_name,
        "exported_windows": exported_windows,
    }


def main():
    args = build_parser().parse_args()

    if args.rollout_horizon_steps <= 0:
        raise ValueError("--rollout-horizon-steps must be positive.")
    if args.rollout_batch_size <= 0:
        raise ValueError("--rollout-batch-size must be positive.")

    if args.summary_paths:
        summary_paths = [
            Path(item).expanduser().resolve() for item in args.summary_paths
        ]
    else:
        base_root = Path(args.base_root).expanduser().resolve()
        if not base_root.exists():
            raise FileNotFoundError(f"Base root not found: {base_root}")
        summary_paths = helpers.discover_summary_paths(base_root)

    if not summary_paths:
        raise FileNotFoundError("No event fine-tuning summary files were found.")

    task_filter = set(args.tasks) if args.tasks else None
    selected_paths = []
    for summary_path in summary_paths:
        payload = helpers.load_json(summary_path)
        if should_keep_summary(summary_path, payload, task_filter, args.run_tags):
            selected_paths.append(summary_path)

    if not selected_paths:
        raise FileNotFoundError("No summary files matched the requested filters.")

    device = torch.device(args.device)
    results = []
    for summary_path in selected_paths:
        print(f"[EXPORT] {summary_path}")
        result = export_one_run(
            summary_path=summary_path,
            split_name=args.split,
            checkpoint_kind=args.checkpoint_kind,
            rollout_horizon_steps=int(args.rollout_horizon_steps),
            allow_partial_horizon=bool(args.allow_partial_horizon),
            device=device,
            rollout_batch_size=int(args.rollout_batch_size),
            output_name=args.output_name,
            metrics_output_name=args.metrics_output_name,
        )
        results.append(result)
        print(
            f"[DONE] run_tag={result['run_tag']} split={result['split']} "
            f"exported_windows={result['exported_windows']} "
            f"main_csv={result['main_csv_path']} metrics_csv={result['metrics_csv_path']}"
        )

    print(f"[SUMMARY] Exported {len(results)} run(s).")


if __name__ == "__main__":
    main()
