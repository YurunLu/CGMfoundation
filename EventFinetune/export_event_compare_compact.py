from __future__ import annotations

import argparse
import csv
import sys
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

COL_SAMPLE_ID = "\u6837\u672cID"
COL_WINDOW_INDEX = "\u7a97\u53e3\u7f16\u53f7"
COL_EVENT_ID = "\u4e8b\u4ef6\u7f16\u53f7"
COL_EVENT_PHASE = "\u4e8b\u4ef6\u9636\u6bb5"
COL_EVENT_TIME = "\u4e8b\u4ef6\u65f6\u95f4"
COL_EVENT_CONTENT = "\u4e8b\u4ef6\u5185\u5bb9"
COL_FORECAST_START_TIME = "\u9884\u6d4b\u5f00\u59cb\u65f6\u95f4"
COL_ROW_TYPE = "\u884c\u7c7b\u578b"
COL_MAE = "MAE"
COL_RMSE = "RMSE"
COL_BIAS = "Bias"
COL_MAE_DELTA = "MAE\u53d8\u5316"

ROW_TRUE = "\u771f\u5b9e\u503c"
ROW_PRE = "\u5fae\u8c03\u524d"
ROW_POST = "\u5fae\u8c03\u540e"
PHASE_INPUT = "\u8f93\u5165\u671f"
PHASE_FORECAST = "\u9884\u6d4b\u671f"

METRIC_SCOPE = "\u8303\u56f4"
METRIC_NAME = "\u540d\u79f0"
METRIC_VALUE = "\u503c"
METRIC_VALID_POINTS = "\u6709\u6548\u70b9\u6570"
METRIC_PRE_MAE = "\u5fae\u8c03\u524dMAE"
METRIC_POST_MAE = "\u5fae\u8c03\u540eMAE"
METRIC_PRE_RMSE = "\u5fae\u8c03\u524dRMSE"
METRIC_POST_RMSE = "\u5fae\u8c03\u540eRMSE"
METRIC_PRE_BIAS = "\u5fae\u8c03\u524dBias"
METRIC_POST_BIAS = "\u5fae\u8c03\u540eBias"
METRIC_RMSE_DELTA = "RMSE\u53d8\u5316"
METRIC_BIAS_DELTA = "Bias\u53d8\u5316"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Export compact event comparison CSV files. "
            "Only rows with real events are exported. "
            "Each event is written as 3 rows: true / pre-finetune / post-finetune."
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
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Default: <run_tag>_event_compare_compact_<split>.csv",
    )
    parser.add_argument(
        "--metrics-output-name",
        type=str,
        default=None,
        help="Default: <run_tag>_event_compare_compact_<split>_metrics.csv",
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


def build_main_columns(config: dict) -> list[str]:
    patch_size = int(config["patch_size"])
    forecast_steps = int(config["forecast_steps"])
    columns = [COL_SAMPLE_ID, COL_WINDOW_INDEX, COL_EVENT_ID, COL_EVENT_PHASE]
    columns.extend([f"\u8f93\u5165\u70b9{idx + 1}" for idx in range(patch_size)])
    columns.extend(
        [
            COL_EVENT_TIME,
            COL_EVENT_CONTENT,
            COL_FORECAST_START_TIME,
            COL_ROW_TYPE,
        ]
    )
    columns.extend([f"t+{step + 1}" for step in range(forecast_steps)])
    columns.extend([COL_MAE, COL_RMSE, COL_BIAS, COL_MAE_DELTA])
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


def build_kept_patch_starts(
    dataset,
    dataset_index: int,
    patch_size: int,
    forecast_steps: int,
) -> list[int]:
    raw_glucose = dataset.glucose[dataset_index]
    kept_starts: list[int] = []
    for start in dataset.device_patch_starts[dataset_index]:
        start = int(start)
        if start + patch_size + forecast_steps > len(raw_glucose):
            continue
        x_raw_patch = raw_glucose[start : start + patch_size]
        valid_ratio = float(np.sum(~np.isnan(x_raw_patch))) / float(patch_size)
        if valid_ratio < 0.5:
            continue
        kept_starts.append(start)
    return kept_starts


def split_consecutive_indices(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append((start, prev + 1))
        start = value
        prev = value
    ranges.append((start, prev + 1))
    return ranges


def predict_selected_base(
    base_model,
    sample: dict,
    selected_patch_indices: list[int],
    model_stride: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    outputs: dict[int, np.ndarray] = {}
    if not selected_patch_indices:
        return outputs

    patch_array = sample["patches"]
    base_model.eval()
    with torch.no_grad():
        for start, stop in split_consecutive_indices(selected_patch_indices):
            patches = torch.from_numpy(patch_array[start:stop]).unsqueeze(0).to(device)
            preds = base_model(patches, start_idx=start, stride=model_stride)
            preds_np = preds.squeeze(0).detach().cpu().numpy()
            for offset, patch_index in enumerate(range(start, stop)):
                outputs[int(patch_index)] = preds_np[offset]
    return outputs


def predict_selected_event(
    model,
    sample: dict,
    selected_patch_indices: list[int],
    model_stride: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    outputs: dict[int, np.ndarray] = {}
    if not selected_patch_indices:
        return outputs

    patch_array = sample["patches"]
    event_array = sample["events"]
    model.eval()
    with torch.no_grad():
        for start, stop in split_consecutive_indices(selected_patch_indices):
            patches = torch.from_numpy(patch_array[start:stop]).unsqueeze(0).to(device)
            events = torch.from_numpy(event_array[start:stop]).unsqueeze(0).to(device)
            preds = model(patches, events, start_idx=start, stride=model_stride)
            preds_np = preds.squeeze(0).detach().cpu().numpy()
            for offset, patch_index in enumerate(range(start, stop)):
                outputs[int(patch_index)] = preds_np[offset]
    return outputs


def init_metric_state(forecast_steps: int) -> dict:
    zeros_f = np.zeros(forecast_steps, dtype=np.float64)
    zeros_i = np.zeros(forecast_steps, dtype=np.int64)
    return {
        "forecast_steps": forecast_steps,
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
    metric_state: dict,
    selected_patch_rows: int,
    valid_forecast_points: int,
    event_blocks: int,
) -> list[dict]:
    rows: list[dict] = []
    rows.append(
        {
            "run_tag": run_tag,
            "split": split_name,
            METRIC_SCOPE: "summary",
            METRIC_NAME: "selected_patch_rows",
            METRIC_VALUE: selected_patch_rows,
        }
    )
    rows.append(
        {
            "run_tag": run_tag,
            "split": split_name,
            METRIC_SCOPE: "summary",
            METRIC_NAME: "valid_forecast_points",
            METRIC_VALUE: valid_forecast_points,
        }
    )
    rows.append(
        {
            "run_tag": run_tag,
            "split": split_name,
            METRIC_SCOPE: "summary",
            METRIC_NAME: "event_blocks",
            METRIC_VALUE: event_blocks,
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

    for step_idx in range(metric_state["forecast_steps"]):
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


def build_shared_row(
    sample_id: str,
    window_index: int,
    event_id: int,
    event_phase: str,
    input_values: list[float | None],
    event_time: str,
    event_content: str,
    forecast_start_time: str,
) -> dict:
    row = {
        COL_SAMPLE_ID: sample_id,
        COL_WINDOW_INDEX: window_index,
        COL_EVENT_ID: event_id,
        COL_EVENT_PHASE: event_phase,
        COL_EVENT_TIME: event_time,
        COL_EVENT_CONTENT: event_content,
        COL_FORECAST_START_TIME: forecast_start_time,
    }
    for idx, value in enumerate(input_values, start=1):
        row[f"\u8f93\u5165\u70b9{idx}"] = value
    return row


def fill_forecast_columns(
    row: dict,
    values: np.ndarray,
    mask: np.ndarray,
    forecast_steps: int,
):
    valid = mask.astype(bool)
    for idx in range(forecast_steps):
        row[f"t+{idx + 1}"] = (
            helpers.maybe_round_float(values[idx]) if valid[idx] else None
        )


def export_one_run(
    summary_path: Path,
    split_override: str | None,
    checkpoint_kind: str,
    device: torch.device,
    output_name: str | None,
    metrics_output_name: str | None,
):
    summary = helpers.load_json(summary_path)
    config = helpers.resolve_training_config(summary, summary_path)
    run_tag = str(summary.get("run_tag", summary_path.stem.replace("_summary", "")))
    split_name = split_override or "all"

    raw_dataset = event_main.build_dataset(config)
    model_dataset = event_main.build_dataset(config)
    selected_indices, split_map = helpers.resolve_selected_indices(
        dataset=model_dataset,
        config=config,
        summary=summary,
        split_name=split_name,
    )
    print(
        f"[INFO] run_tag={run_tag} split={split_name} "
        f"selected_devices={len(selected_indices)} total_devices={len(split_map['all'])}"
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
    forecast_steps = int(config["forecast_steps"])
    model_stride = int(config["model_stride"])
    event_id_counter = 0
    event_blocks = 0
    selected_patch_rows_total = 0
    valid_forecast_points_total = 0
    metric_state = init_metric_state(forecast_steps)

    output_dir = summary_path.parent
    main_csv_name = (
        output_name
        if output_name
        else f"{run_tag}_event_compare_compact_{split_name}.csv"
    )
    metrics_csv_name = (
        metrics_output_name
        if metrics_output_name
        else f"{run_tag}_event_compare_compact_{split_name}_metrics.csv"
    )
    main_csv_path = output_dir / main_csv_name
    metrics_csv_path = output_dir / metrics_csv_name
    main_columns = build_main_columns(config)

    with main_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_columns)
        writer.writeheader()

        for device_order, dataset_index in enumerate(selected_indices, start=1):
            raw_sample = raw_dataset[dataset_index]
            model_sample = model_dataset[dataset_index]

            if raw_sample["patches"].shape[0] == 0:
                continue

            sample_id = str(raw_sample["id"])
            device_id = str(raw_sample["device_id"])
            full_events = raw_dataset.events_map.get(device_id)
            if full_events is None:
                continue

            kept_starts = build_kept_patch_starts(
                raw_dataset,
                dataset_index,
                patch_size,
                forecast_steps,
            )
            if len(kept_starts) != raw_sample["patches"].shape[0]:
                raise RuntimeError(
                    f"Patch start count mismatch for device_id={device_id}: "
                    f"kept_starts={len(kept_starts)} sample_patches={raw_sample['patches'].shape[0]}"
                )

            nonzero_event_indices = np.flatnonzero(
                np.any(np.abs(full_events) > 0, axis=1)
            ).tolist()
            if not nonzero_event_indices:
                continue

            event_to_patch: dict[int, int] = {}
            for event_abs_index in nonzero_event_indices:
                representative_patch_index = helpers.choose_representative_patch_index(
                    starts=kept_starts,
                    event_index=int(event_abs_index),
                    patch_size=patch_size,
                    forecast_steps=forecast_steps,
                )
                if representative_patch_index is None:
                    continue
                event_to_patch[int(event_abs_index)] = int(representative_patch_index)

            if not event_to_patch:
                continue

            selected_patch_indices = sorted(set(event_to_patch.values()))
            pre_pred_map = predict_selected_base(
                base_model=pre_base,
                sample=model_sample,
                selected_patch_indices=selected_patch_indices,
                model_stride=model_stride,
                device=device,
            )
            post_pred_map = predict_selected_event(
                model=post_model,
                sample=model_sample,
                selected_patch_indices=selected_patch_indices,
                model_stride=model_stride,
                device=device,
            )

            targets = raw_sample["targets"]
            masks = raw_sample["masks"]
            patches = raw_sample["patches"]

            selected_patch_rows_total += len(selected_patch_indices)
            for patch_index in selected_patch_indices:
                if patch_index not in pre_pred_map or patch_index not in post_pred_map:
                    raise RuntimeError(
                        f"Missing sparse prediction for device_id={device_id}, patch_index={patch_index}"
                    )
                accumulate_metrics(
                    metric_state,
                    pre_preds=np.asarray(pre_pred_map[patch_index], dtype=np.float32),
                    post_preds=np.asarray(post_pred_map[patch_index], dtype=np.float32),
                    targets=np.asarray(targets[patch_index], dtype=np.float32),
                    mask=np.asarray(masks[patch_index], dtype=np.float32),
                )
                valid_forecast_points_total += int(
                    np.asarray(masks[patch_index]).astype(bool).sum()
                )

            for event_abs_index in sorted(event_to_patch):
                patch_index = event_to_patch[event_abs_index]
                patch_start = int(kept_starts[patch_index])
                target_start = int(patch_start + patch_size)
                patch_mask = np.asarray(masks[patch_index]).astype(bool)
                input_values = [
                    helpers.maybe_round_float(value)
                    for value in np.asarray(patches[patch_index])
                ]
                pre_patch_metrics = helpers.build_patch_metric_dict(
                    np.asarray(pre_pred_map[patch_index], dtype=np.float32),
                    np.asarray(targets[patch_index], dtype=np.float32),
                    patch_mask,
                )
                post_patch_metrics = helpers.build_patch_metric_dict(
                    np.asarray(post_pred_map[patch_index], dtype=np.float32),
                    np.asarray(targets[patch_index], dtype=np.float32),
                    patch_mask,
                )

                event_time = (
                    helpers.resolve_time_info(
                        time_meta, sample_id, device_id, int(event_abs_index)
                    )["absolute_time"]
                    or f"index={int(event_abs_index)}"
                )
                forecast_start_time = (
                    helpers.resolve_time_info(
                        time_meta, sample_id, device_id, target_start
                    )["absolute_time"]
                    or f"index={target_start}"
                )
                event_content = helpers.format_event_content(
                    np.asarray(full_events[event_abs_index]),
                    event_cols,
                )
                event_phase = (
                    PHASE_INPUT
                    if int(event_abs_index) < target_start
                    else PHASE_FORECAST
                )

                event_id_counter += 1
                shared = build_shared_row(
                    sample_id=sample_id,
                    window_index=patch_index,
                    event_id=event_id_counter,
                    event_phase=event_phase,
                    input_values=input_values,
                    event_time=event_time,
                    event_content=event_content,
                    forecast_start_time=forecast_start_time,
                )

                true_row = dict(shared)
                true_row[COL_ROW_TYPE] = ROW_TRUE
                true_row[COL_MAE] = None
                true_row[COL_RMSE] = None
                true_row[COL_BIAS] = None
                true_row[COL_MAE_DELTA] = None
                fill_forecast_columns(
                    true_row,
                    np.asarray(targets[patch_index], dtype=np.float32),
                    patch_mask,
                    forecast_steps,
                )
                writer.writerow(true_row)

                pre_row = dict(shared)
                pre_row[COL_ROW_TYPE] = ROW_PRE
                pre_row[COL_MAE] = helpers.maybe_round_float(pre_patch_metrics["mae"])
                pre_row[COL_RMSE] = helpers.maybe_round_float(pre_patch_metrics["rmse"])
                pre_row[COL_BIAS] = helpers.maybe_round_float(pre_patch_metrics["bias"])
                pre_row[COL_MAE_DELTA] = None
                fill_forecast_columns(
                    pre_row,
                    np.asarray(pre_pred_map[patch_index], dtype=np.float32),
                    patch_mask,
                    forecast_steps,
                )
                writer.writerow(pre_row)

                post_row = dict(shared)
                post_row[COL_ROW_TYPE] = ROW_POST
                post_row[COL_MAE] = helpers.maybe_round_float(post_patch_metrics["mae"])
                post_row[COL_RMSE] = helpers.maybe_round_float(
                    post_patch_metrics["rmse"]
                )
                post_row[COL_BIAS] = helpers.maybe_round_float(
                    post_patch_metrics["bias"]
                )
                post_row[COL_MAE_DELTA] = helpers.maybe_round_float(
                    maybe_delta(post_patch_metrics["mae"], pre_patch_metrics["mae"])
                )
                fill_forecast_columns(
                    post_row,
                    np.asarray(post_pred_map[patch_index], dtype=np.float32),
                    patch_mask,
                    forecast_steps,
                )
                writer.writerow(post_row)
                event_blocks += 1

            if device_order % 50 == 0:
                print(
                    f"[INFO] run_tag={run_tag} processed_devices={device_order}/{len(selected_indices)} "
                    f"event_blocks={event_blocks}"
                )

    metrics_rows = build_metrics_rows(
        run_tag=run_tag,
        split_name=split_name,
        checkpoint_kind=resolved_ckpt_kind,
        metric_state=metric_state,
        selected_patch_rows=selected_patch_rows_total,
        valid_forecast_points=valid_forecast_points_total,
        event_blocks=event_blocks,
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
        "event_blocks": event_blocks,
        "selected_patch_rows": selected_patch_rows_total,
    }


def main():
    args = build_parser().parse_args()

    summary_paths = []
    if args.summary_paths:
        summary_paths = [Path(item).expanduser().resolve() for item in args.summary_paths]
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
            split_override=args.split,
            checkpoint_kind=args.checkpoint_kind,
            device=device,
            output_name=args.output_name,
            metrics_output_name=args.metrics_output_name,
        )
        results.append(result)
        print(
            f"[DONE] run_tag={result['run_tag']} split={result['split']} "
            f"event_blocks={result['event_blocks']} selected_patch_rows={result['selected_patch_rows']} "
            f"main_csv={result['main_csv_path']} metrics_csv={result['metrics_csv_path']}"
        )

    print(f"[SUMMARY] Exported {len(results)} run(s).")


if __name__ == "__main__":
    main()
