# Pretraining and forecast evaluation

This guide covers the CSV dataset builder, forecast pretraining, direct forecast
evaluation with patient identifiers, and full-context 24-hour autoregressive (AR)
evaluation. Run the commands from the repository root. Shell examples use Bash on
a Linux host; select devices and paths appropriate for your environment.

## Environment

The pretraining environment uses Python 3.10 and PyTorch 2.6.0 with CUDA 12.4.
`requirements-pretrain.txt` specifies the direct Python dependency versions,
including Accelerate, Hugging Face Datasets, and Weights & Biases (W&B).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-pretrain.txt
```

The CUDA wheel command follows the
[official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/).
Use a compatible NVIDIA driver, or install a PyTorch build for the intended CPU
or CUDA environment. The optional `requirements-tested.txt` file specifies
dependencies for automated workflow checks.

The trainer initializes W&B on the main process. To record runs locally without an
online W&B session, set offline mode before launching training:

```bash
export WANDB_MODE=offline
```

On PowerShell, the equivalent setting is `$env:WANDB_MODE = "offline"`.
Online logging can instead use an authenticated W&B session. Project and run names
are configured under `wandb` in the YAML file.

## CSV input

Each row represents one CGM sequence. The first CSV column is read as the row index
and should contain a sequence or patient identifier. The remaining CGM columns
represent consecutive **5-minute samples in mmol/L**, ordered by numeric time index.
Both zero-based (`0` through `3455`) and one-based (`1` through `3456`) column names
are accepted by the dataset builder for the supplied 3456-point configuration.
Keep missing time positions as empty values rather than removing their columns.
Keep other metadata outside the CGM columns.

An abbreviated zero-based example is:

```csv
id,0,1,2,3,4
sequence_001,5.5,5.6,,5.7,5.8
sequence_002,7.1,,7.3,7.2,7.0
```

This example illustrates the format only. Supply the full sequence columns for your
configured length; the short example does not satisfy the default validity filter.
The default `max_seq_len` is 3456 points, corresponding to 12 days. Missing positions
are padded with missing values, and positions outside the configured range are
excluded during dataset construction.

The standalone evaluators also read the first column as an index. If a separate
`id` column is present, it is used for patient identifiers; otherwise, the index is
used. The direct evaluator orders CGM columns numerically and aligns the sequence
to its 3456-point evaluation layout.

## Cleaning, masks, and data splits

Dataset construction applies the following steps:

1. Convert CGM entries to numeric values. Missing or nonnumeric entries, values
   `<= 0`, and values `> 60.0` mmol/L are treated as missing.
2. Clip remaining positive values to the sensor range `[1.11, 33.3]` mmol/L.
3. Retain rows with at least 80% observed values over `max_seq_len` after cleaning
   and column alignment.
4. Create `orig_mask` before imputation, then fill gaps with the last observation
   carried forward (LOCF). Leading gaps without a previous observation are filled
   with `5.5` mmol/L.
5. Shuffle rows with the configured seed, split them, and save `seq_filled` and
   `orig_mask` as a Hugging Face `DatasetDict`.

The model receives a single channel of filled glucose values. During pretraining,
the original observation mask is used for forecast-target losses and metrics;
the historical mask is not supplied as an additional model input channel.
Patches are extracted from the stored full sequences on the fly during training.

The supplied configurations request **90% training / 10% validation / 0% internal
test**. The builder computes integer cut points from `train_ratio` and `val_ratio`;
any remainder belongs to `test`. Splitting is performed at the row level. An empty
test split is supported, and the trainer skips its final internal test evaluation
when no test rows are available. Evaluate external cohorts separately with the CSV
evaluation commands below.

## Configuration

Two YAML files are supplied:

| File | Role |
| --- | --- |
| `configs/config.yaml` | General configuration and the default builder/trainer configuration |
| `configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml` | Pretraining settings with 12-hour patches, a 1-hour forecast, and a maximum of 500 epochs |

Choose a configuration and keep the dataset layout compatible with its model
settings. The supplied 12-hour configuration
uses `patch_size=144`, `stride=3`, `forecast_steps=12`, `embed_dim=256`, `depth=12`,
`num_heads=8`, `hidden_dim=1024`, `dropout=0.1`, positional encoding enabled, and
temporal embedding disabled.

Set these paths for the input data and output directories:

| Key | Local value to provide |
| --- | --- |
| `data.csv_dir` | Directory containing the source sequence CSV files |
| `data.hf_dataset_dir` | Directory where the builder saves and the trainer loads the dataset |
| `training.save_dir` | Directory for checkpoints and training logs |

Use the same configuration for dataset construction and training. Keep
`data.hf_dataset_dir`, `max_seq_len`, and the intended data split consistent.
The builder reads all `*.csv` files directly inside `data.csv_dir`.

You may edit the YAML paths or make local copies and pass their paths with `--config`.
Both entry points default to the repository's `configs/config.yaml` when no
configuration argument is supplied. Explicit relative config paths are resolved
from the current working directory. Training hyperparameters, architecture flags,
output location, seed, worker count, and W&B names are configured in YAML.

## Build the dataset

```bash
python build_dataset.py --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml
```

The builder writes the `train`, `validation`, and `test` splits under
`data.hf_dataset_dir`. This preprocessing step does not train a model. Rebuild the
dataset when changing the source cohort, cleaning inputs, sequence layout, or split
settings.

## Train

The following single-GPU example starts from random initialization and uses the
12-hour pretraining configuration:

```bash
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0 python pretrain_main.py \
  --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml
```

For four GPUs on one host:

```bash
export WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 \
  pretrain_main.py \
  --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml
```

Adjust the visible GPU IDs and process count together. `training.batch_size` is the
batch size per process; the trainer scales the configured learning rate by the
square root of the process count. GPU count also affects optimizer updates per
epoch and the wrapped scheduler. Record GPU count, batch size, learning rate,
warmup settings, and selected epoch when reporting or comparing runs.

The training objective is masked clinical-weighted L1 loss. Default glucose
thresholds are 3.9 and 10.0 mmol/L, with weights 2.0, 1.2, and 1.0 for low, high,
and other observed target values. The supplied configuration sets a maximum of 500
epochs and patience of 20: training stops after 20 consecutive epochs without a
strict improvement over the best validation loss. A new best resets
the counter.

Outputs under `training.save_dir` include:

- `best_forecast_<timestamp>_dim<embed>_patch<patch>_stride<stride>.pt`:
  checkpoint selected by validation loss, with model settings and optimizer/scheduler states.
- `latest_checkpoint_<epoch>.pt`: periodic checkpoints saved every 10 epochs.
- `log_<timestamp>.csv`: epoch losses and validation metrics.

W&B writes its local run records separately. Use the saved best checkpoint for the
standalone evaluation commands; replace `/path/to/best_forecast.pt` below with its
actual filename.

## Direct forecast and patient-level evaluation

```bash
python EvaluatePretrain_with_patient_id_patient_metrics.py \
  --csv_path /path/to/external_cohort.csv \
  --checkpoint_path /path/to/best_forecast.pt \
  --output_dir ./outputs/direct_eval \
  --device cuda \
  --batch_size 32
```

The evaluator restores model settings from checkpoint metadata, builds forecast
windows from the external CSV, and evaluates observed targets. Its row filter
requires at least `patch_size + forecast_steps` valid observations. The supplied
configuration predicts 12 steps (1 hour) at each patch position, with earlier
history patches available through causal attention.

This evaluator uses a fixed `max_seq_len=3456`, matching the supplied configurations.
Adapt the evaluation layout when using another positional-buffer length.

Outputs include `time_slice_metrics.csv`, `mae_per_patch.csv`, `rmae_per_patch.csv`,
`patient_level_metrics.csv`, and `patient_time_slice_metrics.csv`.
`raw_predictions.pt` stores predictions, targets, masks, and retained patient IDs;
`patient_ids.csv` records the row-to-ID mapping, and `flattened_predictions.csv`
contains valid prediction/target pairs with patient, window, and step identifiers.

## Full-context 24-hour AR evaluation

```bash
python Eval_AR_24h_FullContext_clamped.py \
  --csv_path /path/to/external_cohort.csv \
  --checkpoint_path /path/to/best_forecast.pt \
  --output_dir ./outputs/ar_24h_eval \
  --device cuda \
  --stride 3 \
  --ar_step_size 3 \
  --workers 1 \
  --min_history_hours 24 \
  --min_history_valid_ratio 0.7
```

For each eligible row, the final 288 points ending at its last observed value form
the 24-hour target; preceding values provide the history. The default criteria
require at least 24 hours of history, at least 70% observed history points, and at
least 50% observed target points. The available model context is limited by the
checkpoint's `max_seq_len`.

The example appends 3 predictions (15 minutes) per iteration and feeds them back
into the context. Keep `stride` consistent with the training checkpoint and use a
rollout step no greater than its forecast output length. The example step size
matches the patch stride and divides the 288-point horizon. Predictions are clipped
before feedback and evaluation to `[2.22, 33.33]` mmol/L by default; the bounds can
be changed with `--clip_min` and `--clip_max`.

For `ar_step_size=3`, outputs are:

- `AR_24h_step_3_clin_metrics.csv`: patient-level horizon and clinical metrics.
- `Global_vs_Indiv_TimeDecay_step3.csv`: pooled and mean patient-level MAE/correlation
  for horizons from 30 minutes to 24 hours.
- `Pop_Metric_Corr_step3.csv`: correlations between predicted and observed patient
  clinical summaries.
- `AR_24h_step_3_curves.pt`: patient prediction curves, targets, and masks.

## Data and weight availability

This repository provides source code and configurations. Participant data and
pretrained weights are not included. Training requires your own appropriately
formatted cohort; evaluation requires both an external CSV and a compatible
checkpoint. See the [downstream guide](DOWNSTREAM.md) for clinical and
event-conditioned tasks and [workflow checks](DOWNSTREAM_VALIDATION.md) for
automated execution checks. Frozen DGP reference parameters and their usage are
described in the [DGP guide](PAPER_DGP.md).
