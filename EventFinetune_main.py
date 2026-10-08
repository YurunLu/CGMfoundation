import argparse
import csv
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EventFinetune.Event_Finetune_data_utils import (
    EventDataset,
    collate_patches,
    normalize_entity_key,
)
from EventFinetune.Event_Finetune_model import EventFinetuneModel
from EventFinetune.Event_loss_utils import masked_mae_loss

DEFAULT_CONFIG = {
    "run_name": "event_finetune",
    "data_csv": "CGMacros_data_withevent_simple_5min_3456.csv",
    "onehot_csv": "CGMacros_OneHot_all_simple_5min_3456.csv",
    "pretrain_ckpt": "",
    "out_dir": "data_event/event_finetune_outputs",
    "epochs": 3000,
    "batch_size": 16,
    "patch_size": 144,
    "stride": 3,
    "forecast_steps": 12,
    "embed_dim": 256,
    "depth": 12,
    "num_heads": 8,
    "hidden_dim": 1024,
    "max_seq_len": 3456,
    "model_stride": 3,
    "lr": 5e-4,
    "base_lr": 1e-6,
    "weight_decay": 1e-4,
    "freeze_epochs": 50,
    "eval_every": 1,
    "num_workers": 4,
    "train_ratio": 0.8,
    "seed": 42,
    "glucose_scale": 1.0,
    "dropout": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "event_cols": [
        "Diet",
        "Calories",
        "Carbohydrates",
        "Protein",
        "Fat",
        "Fiber",
    ],
    "use_cnn": True,
    "use_transformer": True,
    "use_temporal": True,
    "use_positional": True,
    "normalize_continuous_events": False,
    "save_split_datasets": True,
    "save_best_checkpoint": True,
    "save_final_checkpoint": True,
    "export_predictions": True,
    "prediction_split": "val",
    "prediction_patch_chunk_size": 0,
}

INT_KEYS = {
    "epochs",
    "batch_size",
    "patch_size",
    "stride",
    "forecast_steps",
    "embed_dim",
    "depth",
    "num_heads",
    "hidden_dim",
    "max_seq_len",
    "model_stride",
    "freeze_epochs",
    "eval_every",
    "num_workers",
    "seed",
    "prediction_patch_chunk_size",
}
FLOAT_KEYS = {
    "lr",
    "base_lr",
    "weight_decay",
    "train_ratio",
    "glucose_scale",
    "dropout",
}
BOOL_KEYS = {
    "use_cnn",
    "use_transformer",
    "use_temporal",
    "use_positional",
    "normalize_continuous_events",
    "save_split_datasets",
    "save_best_checkpoint",
    "save_final_checkpoint",
    "export_predictions",
}
PATH_KEYS = {"data_csv", "onehot_csv", "pretrain_ckpt"}

CKPT_MODEL_KEY_MAP = {
    "patch_size": "patch_size",
    "stride": "stride",
    "forecast_steps": "forecast_steps",
    "embed_dim": "embed_dim",
    "depth": "depth",
    "num_heads": "num_heads",
    "hidden_dim": "hidden_dim",
    "max_seq_len": "max_seq_len",
    "dropout": "dropout",
    "use_cnn": "use_cnn",
    "use_transformer": "use_transformer",
    "use_temporal": "use_temporal",
    "use_positional": "use_positional",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EventFinetune with a JSON config file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/event_finetune_config.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--prediction_split",
        type=str,
        choices=["train", "val", "all"],
        default=None,
    )
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def load_json_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_existing_path(raw_value, config_dir: Path):
    path = Path(raw_value)
    if path.is_absolute():
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        return str(path)
    if path.exists():
        return str(path.resolve())
    candidate = (config_dir / path).resolve()
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Path not found: {raw_value}")


def resolve_pretrain_checkpoint_path(raw_value, config_dir: Path):
    resolved = Path(resolve_existing_path(raw_value, config_dir))
    if resolved.is_file():
        return str(resolved)

    candidates = sorted(resolved.glob("best_forecast*.pt"))
    if not candidates:
        candidates = sorted(resolved.glob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint file found under directory: {resolved}")

    selected = max(candidates, key=lambda item: item.stat().st_mtime)
    print(f"[INFO] Resolved pretrain checkpoint directory {resolved} -> {selected}")
    return str(selected)


def load_pretrain_model_hparams(pretrain_ckpt_path: str):
    try:
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(pretrain_ckpt_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        return {}
    for key in ("model_hparams", "config", "args"):
        model_hparams = ckpt.get(key)
        if isinstance(model_hparams, dict):
            return model_hparams
    return {}


def apply_pretrain_model_config(config):
    model_hparams = load_pretrain_model_hparams(config["pretrain_ckpt"])
    if not model_hparams:
        if config["model_stride"] != config["stride"]:
            print(
                f"[INFO] Aligning model_stride={config['model_stride']} "
                f"to stride={config['stride']}"
            )
            config["model_stride"] = config["stride"]
        return

    for ckpt_key, config_key in CKPT_MODEL_KEY_MAP.items():
        if ckpt_key not in model_hparams or model_hparams[ckpt_key] is None:
            continue
        new_value = model_hparams[ckpt_key]
        old_value = config.get(config_key)
        if old_value != new_value:
            print(
                f"[INFO] Using {config_key}={new_value} from pretrained checkpoint "
                f"(config was {old_value})"
            )
        config[config_key] = new_value

    if config["model_stride"] != config["stride"]:
        print(
            f"[INFO] Aligning model_stride={config['model_stride']} "
            f"to pretrained stride={config['stride']}"
        )
        config["model_stride"] = config["stride"]


def resolve_config(cli_args):
    config_path = Path(cli_args.config).expanduser().resolve()
    file_config = load_json_config(config_path)
    config = deepcopy(DEFAULT_CONFIG)
    config.update(file_config)

    for key in [
        "device",
        "epochs",
        "batch_size",
        "out_dir",
        "run_name",
        "prediction_split",
        "num_workers",
    ]:
        value = getattr(cli_args, key)
        if value is not None:
            config[key] = value

    for key in INT_KEYS:
        config[key] = int(config[key])
    for key in FLOAT_KEYS:
        config[key] = float(config[key])
    for key in BOOL_KEYS:
        config[key] = bool(config[key])

    if not config.get("event_cols"):
        raise ValueError("Config field 'event_cols' cannot be empty.")
    config["event_cols"] = [str(col) for col in config["event_cols"]]

    if not 0.0 < config["train_ratio"] < 1.0:
        raise ValueError("Config field 'train_ratio' must be between 0 and 1.")
    if config["eval_every"] <= 0:
        raise ValueError("Config field 'eval_every' must be positive.")
    if config["freeze_epochs"] < 0:
        raise ValueError("Config field 'freeze_epochs' cannot be negative.")
    if config["prediction_split"] not in {"train", "val", "all"}:
        raise ValueError("Config field 'prediction_split' must be train, val or all.")

    for key in PATH_KEYS:
        if not config.get(key):
            raise ValueError(f"Config field '{key}' is required.")
        if key == "pretrain_ckpt":
            config[key] = resolve_pretrain_checkpoint_path(
                config[key], config_path.parent
            )
        else:
            config[key] = resolve_existing_path(config[key], config_path.parent)

    apply_pretrain_model_config(config)

    config["config_path"] = str(config_path)
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config):
    return EventDataset(
        config["data_csv"],
        config["onehot_csv"],
        event_columns=config["event_cols"],
        patch_size=config["patch_size"],
        stride=config["stride"],
        forecast_steps=config["forecast_steps"],
        glucose_scale=config["glucose_scale"],
    )


def split_indices(dataset, train_ratio):
    n_devices = len(dataset.device_ids)
    train_stop = int(train_ratio * n_devices)
    train_indices = list(range(0, train_stop))
    val_indices = list(range(train_stop, n_devices))
    return {
        "train": train_indices,
        "val": val_indices,
        "all": list(range(0, n_devices)),
    }


def build_loader(subset, batch_size, num_workers, shuffle):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_patches,
        num_workers=num_workers,
    )


def evaluate(model, dataloader, device, model_stride):
    if len(dataloader) == 0:
        return float("nan")

    model.eval()
    total_loss = 0.0
    n_items = 0
    with torch.no_grad():
        for patches, events, targets, mask, patch_starts in dataloader:
            patches = patches.to(device)
            events = events.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            patch_starts = patch_starts.to(device)
            preds = model(
                patches,
                events,
                start_idx=patch_starts,
                stride=model_stride,
            )
            loss = masked_mae_loss(preds, targets, mask)
            total_loss += loss.item() * patches.shape[0]
            n_items += patches.shape[0]
    return total_loss / max(n_items, 1)


def evaluate_base(base_model, dataloader, device, model_stride):
    if len(dataloader) == 0:
        return float("nan")

    base_model.eval()
    total_loss = 0.0
    n_items = 0
    with torch.no_grad():
        for patches, _, targets, mask, patch_starts in dataloader:
            patches = patches.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            patch_starts = patch_starts.to(device)
            preds = base_model(
                patches,
                start_idx=patch_starts,
                stride=model_stride,
            )
            loss = masked_mae_loss(preds, targets, mask)
            total_loss += loss.item() * patches.shape[0]
            n_items += patches.shape[0]
    return total_loss / max(n_items, 1)


def compute_metrics(preds, targets, mask):
    valid = mask.astype(bool)
    if valid.sum() == 0:
        return {"mae": None, "mse": None, "rmse": None}

    diff = preds[valid] - targets[valid]
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    return {"mae": mae, "mse": mse, "rmse": float(np.sqrt(mse))}


def write_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def resolve_data_split_key_series(data_df: pd.DataFrame):
    if "device_id" in data_df.columns:
        return data_df["device_id"].map(normalize_entity_key)
    if "id" in data_df.columns:
        return data_df["id"].map(normalize_entity_key)
    return data_df.iloc[:, 0].map(normalize_entity_key)


def export_split_artifacts(dataset, split_map, config, out_dir, run_tag):
    data_df = pd.read_csv(config["data_csv"])
    data_key = resolve_data_split_key_series(data_df)

    onehot_df = pd.read_csv(config["onehot_csv"], index_col=0)
    if "device_id" not in onehot_df.columns:
        raise ValueError("One-hot CSV must contain a 'device_id' column.")
    onehot_key = onehot_df["device_id"].map(normalize_entity_key)

    index_to_split = {}
    for split_name in ("train", "val"):
        for dataset_index in split_map[split_name]:
            index_to_split[dataset_index] = split_name

    assignment_rows = []
    for dataset_index in split_map["all"]:
        assignment_rows.append(
            {
                "dataset_index": int(dataset_index),
                "split": index_to_split.get(dataset_index, "all"),
                "id": dataset.ids[dataset_index],
                "device_id": dataset.device_ids[dataset_index],
                "valid_length": int(dataset.valid_lengths[dataset_index]),
                "num_patches": int(len(dataset.device_patch_starts[dataset_index])),
            }
        )

    assignment_path = out_dir / f"{run_tag}_split_assignment.csv"
    write_csv(
        assignment_path,
        assignment_rows,
        [
            "dataset_index",
            "split",
            "id",
            "device_id",
            "valid_length",
            "num_patches",
        ],
    )

    split_exports = {}
    for split_name, indices in split_map.items():
        selected_device_ids = {
            normalize_entity_key(dataset.device_ids[idx]) for idx in indices
        }
        split_data_df = data_df.loc[data_key.isin(selected_device_ids)].copy()
        split_onehot_df = onehot_df.loc[onehot_key.isin(selected_device_ids)].copy()

        data_path = out_dir / f"{run_tag}_{split_name}_data.csv"
        onehot_path = out_dir / f"{run_tag}_{split_name}_onehot.csv"

        split_data_df.to_csv(data_path, index=False, encoding="utf-8")
        split_onehot_df.to_csv(onehot_path, encoding="utf-8")

        split_exports[split_name] = {
            "num_devices": len(indices),
            "data_csv": str(data_path),
            "onehot_csv": str(onehot_path),
        }

    summary = {
        "assignment_path": str(assignment_path),
        "splits": split_exports,
    }
    summary_path = out_dir / f"{run_tag}_split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "assignment_path": str(assignment_path),
        "summary_path": str(summary_path),
        "splits": split_exports,
    }


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, best_val, config, extra=None
):
    serializable_config = {
        key: value for key, value in config.items() if key != "config_path"
    }
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "args": serializable_config,
        "config": serializable_config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def predict_device(model, sample, model_stride, device, patch_chunk_size):
    patch_array = sample["patches"]
    event_array = sample["events"]
    patch_starts = sample["patch_starts"]
    if patch_array.shape[0] == 0:
        return np.zeros((0, sample["targets"].shape[1]), dtype=np.float32)

    outputs = []
    with torch.no_grad():
        if patch_chunk_size <= 0 or patch_chunk_size >= patch_array.shape[0]:
            patches = torch.from_numpy(patch_array).unsqueeze(0).to(device)
            events = torch.from_numpy(event_array).unsqueeze(0).to(device)
            starts = torch.from_numpy(patch_starts).unsqueeze(0).to(device)
            preds = model(patches, events, start_idx=starts, stride=model_stride)
            return preds.squeeze(0).cpu().numpy()

        for start in range(0, patch_array.shape[0], patch_chunk_size):
            stop = min(start + patch_chunk_size, patch_array.shape[0])
            patches = torch.from_numpy(patch_array[start:stop]).unsqueeze(0).to(device)
            events = torch.from_numpy(event_array[start:stop]).unsqueeze(0).to(device)
            starts = torch.from_numpy(patch_starts[start:stop]).unsqueeze(0).to(device)
            preds = model(patches, events, start_idx=starts, stride=model_stride)
            outputs.append(preds.squeeze(0).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def export_prediction_artifacts(
    model, dataset, selected_indices, split_name, config, out_dir, run_tag, device
):
    model.eval()
    matrix_rows = []
    long_rows = []
    horizon_errors = [[] for _ in range(config["forecast_steps"])]
    all_preds = []
    all_targets = []
    all_masks = []

    matrix_headers = [
        "sample_id",
        "device_id",
        "patch_index",
        "patch_start",
        *[f"true_t+{step + 1}" for step in range(config["forecast_steps"])],
        *[f"pred_t+{step + 1}" for step in range(config["forecast_steps"])],
        *[f"abs_err_t+{step + 1}" for step in range(config["forecast_steps"])],
        *[f"mask_t+{step + 1}" for step in range(config["forecast_steps"])],
    ]
    long_headers = [
        "sample_id",
        "device_id",
        "patch_index",
        "patch_start",
        "target_time_index",
        "horizon_step",
        "true_value",
        "pred_value",
        "abs_error",
    ]

    for dataset_index in selected_indices:
        sample = dataset[dataset_index]
        preds = predict_device(
            model=model,
            sample=sample,
            model_stride=config["model_stride"],
            device=device,
            patch_chunk_size=config["prediction_patch_chunk_size"],
        )
        targets = sample["targets"]
        masks = sample["masks"]
        starts = sample["patch_starts"][: preds.shape[0]]

        if preds.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape mismatch for {sample['device_id']}: "
                f"{preds.shape} vs {targets.shape}"
            )

        all_preds.append(preds)
        all_targets.append(targets)
        all_masks.append(masks)

        abs_err = np.abs(preds - targets)
        for step in range(config["forecast_steps"]):
            step_mask = masks[:, step].astype(bool)
            if step_mask.any():
                horizon_errors[step].append(abs_err[step_mask, step])

        for patch_index, patch_start in enumerate(starts):
            matrix_row = {
                "sample_id": sample["id"],
                "device_id": sample["device_id"],
                "patch_index": patch_index,
                "patch_start": int(patch_start),
            }
            for step in range(config["forecast_steps"]):
                is_valid = bool(masks[patch_index, step])
                matrix_row[f"mask_t+{step + 1}"] = int(is_valid)
                if is_valid:
                    true_value = float(targets[patch_index, step])
                    pred_value = float(preds[patch_index, step])
                    error_value = float(abs_err[patch_index, step])
                    matrix_row[f"true_t+{step + 1}"] = true_value
                    matrix_row[f"pred_t+{step + 1}"] = pred_value
                    matrix_row[f"abs_err_t+{step + 1}"] = error_value
                    long_rows.append(
                        {
                            "sample_id": sample["id"],
                            "device_id": sample["device_id"],
                            "patch_index": patch_index,
                            "patch_start": int(patch_start),
                            "target_time_index": int(
                                patch_start + config["patch_size"] + step
                            ),
                            "horizon_step": step + 1,
                            "true_value": true_value,
                            "pred_value": pred_value,
                            "abs_error": error_value,
                        }
                    )
                else:
                    matrix_row[f"true_t+{step + 1}"] = ""
                    matrix_row[f"pred_t+{step + 1}"] = ""
                    matrix_row[f"abs_err_t+{step + 1}"] = ""
            matrix_rows.append(matrix_row)

    if all_preds:
        pred_matrix = np.concatenate(all_preds, axis=0)
        target_matrix = np.concatenate(all_targets, axis=0)
        mask_matrix = np.concatenate(all_masks, axis=0)
    else:
        pred_matrix = np.zeros((0, config["forecast_steps"]), dtype=np.float32)
        target_matrix = np.zeros((0, config["forecast_steps"]), dtype=np.float32)
        mask_matrix = np.zeros((0, config["forecast_steps"]), dtype=np.float32)

    metrics = compute_metrics(pred_matrix, target_matrix, mask_matrix)
    horizon_rows = []
    for step in range(config["forecast_steps"]):
        mae = None
        if horizon_errors[step]:
            mae = float(np.mean(np.concatenate(horizon_errors[step], axis=0)))
        horizon_rows.append({"horizon_step": step + 1, "mae": mae})

    matrix_path = out_dir / f"{run_tag}_comparison_matrix_{split_name}.csv"
    long_path = out_dir / f"{run_tag}_comparison_long_{split_name}.csv"
    horizon_path = out_dir / f"{run_tag}_horizon_metrics_{split_name}.csv"

    write_csv(matrix_path, matrix_rows, matrix_headers)
    write_csv(long_path, long_rows, long_headers)
    write_csv(horizon_path, horizon_rows, ["horizon_step", "mae"])

    return {
        "split": split_name,
        "num_devices": len(selected_indices),
        "num_patch_rows": int(pred_matrix.shape[0]),
        "num_valid_forecast_rows": len(long_rows),
        "metrics": metrics,
        "matrix_path": str(matrix_path),
        "long_path": str(long_path),
        "horizon_metrics_path": str(horizon_path),
    }


def main():
    cli_args = parse_args()
    config = resolve_config(cli_args)
    set_seed(config["seed"])

    out_dir = Path(config["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_tag = f"{config['run_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    resolved_config_path = out_dir / f"{run_tag}_resolved_config.json"
    resolved_config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset = build_dataset(config)
    split_map = split_indices(dataset, config["train_ratio"])
    train_set = dataset.subset_by_device_indices(split_map["train"])
    val_set = dataset.subset_by_device_indices(split_map["val"])

    split_export_info = None
    if config["save_split_datasets"]:
        split_export_info = export_split_artifacts(
            dataset=dataset,
            split_map=split_map,
            config=config,
            out_dir=out_dir,
            run_tag=run_tag,
        )
        print(
            f"[INFO] Wrote split assignment to {split_export_info['assignment_path']}"
        )
        print(f"[INFO] Wrote split summary to {split_export_info['summary_path']}")

    continuous_cols = train_set.parent.infer_continuous_event_columns()
    print(f"[INFO] Continuous event columns: {continuous_cols}")
    event_stats = None
    if config["normalize_continuous_events"] and continuous_cols:
        event_stats = train_set.compute_event_stats(
            continuous_event_columns=continuous_cols
        )
        train_set.apply_event_stats(event_stats)
        val_set.apply_event_stats(event_stats)
        print("[INFO] Applied train-only event normalization.")

    train_loader = build_loader(
        train_set,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
    )
    val_loader = build_loader(
        val_set,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )

    device = torch.device(config["device"])
    model = EventFinetuneModel(
        pretrained_ckpt_path=config["pretrain_ckpt"],
        num_event_types=len(config["event_cols"]),
        patch_size=config["patch_size"],
        forecast_size=config["forecast_steps"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"],
        use_cnn=config["use_cnn"],
        use_transformer=config["use_transformer"],
        use_temporal=config["use_temporal"],
        use_positional=config["use_positional"],
    ).to(device)

    raw_train_loss = evaluate_base(
        model.base,
        train_loader,
        device,
        model_stride=config["model_stride"],
    )
    raw_val_loss = evaluate_base(
        model.base,
        val_loader,
        device,
        model_stride=config["model_stride"],
    )
    print(f"[INFO] Raw train loss: {raw_train_loss:.6f}")
    print(f"[INFO] Raw validation loss: {raw_val_loss:.6f}")

    for parameter in model.base.parameters():
        parameter.requires_grad = False

    optimizer = torch.optim.AdamW(
        [
            {"params": model.base.parameters(), "lr": config["base_lr"]},
            {
                "params": list(model.event_emb.parameters())
                + list(model.past_event_proj.parameters())
                + list(model.future_event_proj.parameters())
                + list(model.future_mlp.parameters())
                + list(model.past_attn_score.parameters())
                + [model.alpha_past, model.alpha_future],
                "lr": config["lr"],
            },
        ],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_val = float("inf")
    best_epoch = 0
    final_train_loss = float("nan")
    final_val_loss = float("nan")
    history_rows = []

    best_ckpt_path = out_dir / f"{run_tag}_best.pt"
    final_ckpt_path = out_dir / f"{run_tag}_final.pt"

    print("[INFO] Starting fine-tuning.")
    for epoch in range(1, config["epochs"] + 1):
        if epoch == config["freeze_epochs"] + 1:
            for parameter in model.base.parameters():
                parameter.requires_grad = True
            print(f"[INFO] Unfroze base model at epoch {epoch}")

        model.train()
        total_loss = 0.0
        n_items = 0
        for patches, events, targets, mask, patch_starts in train_loader:
            patches = patches.to(device)
            events = events.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            patch_starts = patch_starts.to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = model(
                patches,
                events,
                start_idx=patch_starts,
                stride=config["model_stride"],
            )
            loss = masked_mae_loss(preds, targets, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * patches.shape[0]
            n_items += patches.shape[0]

        final_train_loss = total_loss / max(n_items, 1)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": final_train_loss,
            "val_loss": "",
            "alpha_past": float(model.alpha_past.item()),
            "alpha_future": float(model.alpha_future.item()),
        }

        should_eval = (epoch % config["eval_every"] == 0) or (epoch == config["epochs"])
        if should_eval:
            final_val_loss = evaluate(
                model,
                val_loader,
                device,
                model_stride=config["model_stride"],
            )
            record["val_loss"] = final_val_loss
            print(
                f"Epoch {epoch:4d} | train={final_train_loss:.5f} | "
                f"val={final_val_loss:.5f} | "
                f"alpha_past={model.alpha_past.item():.4f} "
                f"alpha_future={model.alpha_future.item():.4f}"
            )
            if final_val_loss < best_val:
                best_val = final_val_loss
                best_epoch = epoch
                if config["save_best_checkpoint"]:
                    save_checkpoint(
                        best_ckpt_path,
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_val,
                        config,
                        extra={
                            "raw_train_loss": raw_train_loss,
                            "raw_val_loss": raw_val_loss,
                        },
                    )
                    print(f"[INFO] Saved best checkpoint to {best_ckpt_path}")
        else:
            print(f"Epoch {epoch:4d} | train={final_train_loss:.5f}")

        history_rows.append(record)

    if config["save_final_checkpoint"]:
        save_checkpoint(
            final_ckpt_path,
            model,
            optimizer,
            scheduler,
            config["epochs"],
            best_val,
            config,
            extra={
                "best_epoch": best_epoch,
                "raw_train_loss": raw_train_loss,
                "raw_val_loss": raw_val_loss,
            },
        )
        print(f"[INFO] Saved final checkpoint to {final_ckpt_path}")

    history_path = out_dir / f"{run_tag}_history.csv"
    write_csv(
        history_path,
        history_rows,
        ["epoch", "train_loss", "val_loss", "alpha_past", "alpha_future"],
    )

    export_info = None
    if config["export_predictions"]:
        export_info = export_prediction_artifacts(
            model=model,
            dataset=dataset,
            selected_indices=split_map[config["prediction_split"]],
            split_name=config["prediction_split"],
            config=config,
            out_dir=out_dir,
            run_tag=run_tag,
            device=device,
        )
        print(f"[INFO] Wrote comparison matrix to {export_info['matrix_path']}")
        print(f"[INFO] Wrote comparison table to {export_info['long_path']}")
        print(f"[INFO] Wrote horizon metrics to {export_info['horizon_metrics_path']}")

    summary = {
        "run_tag": run_tag,
        "config_path": config["config_path"],
        "resolved_config_path": str(resolved_config_path),
        "history_path": str(history_path),
        "best_checkpoint_path": str(best_ckpt_path)
        if config["save_best_checkpoint"]
        else None,
        "final_checkpoint_path": str(final_ckpt_path)
        if config["save_final_checkpoint"]
        else None,
        "num_devices": len(dataset.device_ids),
        "train_devices": len(split_map["train"]),
        "val_devices": len(split_map["val"]),
        "continuous_event_columns": continuous_cols,
        "event_stats": event_stats,
        "raw_train_loss": raw_train_loss,
        "raw_val_loss": raw_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "split_export": split_export_info,
        "prediction_export": export_info,
        "config": {key: value for key, value in config.items() if key != "config_path"},
    }

    summary_path = out_dir / f"{run_tag}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[INFO] Saved training history to {history_path}")
    print(f"[INFO] Saved summary to {summary_path}")
    print(f"[INFO] Best validation loss: {best_val:.5f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
