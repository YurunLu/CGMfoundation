# CGMfoundation

**Foundation modeling of continuous glucose monitoring data for glucose forecasting, clinical prediction, and dynamic glycemic profiling.**

CGMfoundation learns representations of continuous glucose monitoring (CGM) time
series through forecast pretraining. The model combines convolutional patch
encoding with a Transformer using causal attention to capture temporal patterns
in glucose regulation. The learned representations support glucose forecasting,
clinical classification and regression, event-conditioned prediction, and
analysis of individual glucose dynamics.

This repository provides model implementations, training and evaluation scripts,
experiment configurations, and tools for computing the **Dynamic Glucore Profile
(DGP)** from latent CGM trajectories.

## Overview

| Workflow | Description |
| --- | --- |
| Forecast pretraining | Learn CGM representations by predicting future glucose from overlapping history patches |
| Glucose forecasting | Evaluate direct forecasts and 24-hour autoregressive trajectories |
| Clinical prediction | Perform classification and regression with fine-tuned models or frozen representations |
| Event-conditioned prediction | Model glucose responses around annotated events |
| Dynamic glycemic profiling | Quantify basin occupancy, displacement, variability, pulse response, and stability from latent trajectories |

The supplied pretraining configuration uses 12-hour history patches, a 15-minute
patch stride, and a 1-hour prediction horizon at 5-minute resolution. Architecture
and task settings are configurable.

## Installation

Clone the repository and install the dependencies in a dedicated Python environment:

```bash
git clone https://github.com/YurunLu/CGMfoundation.git
cd CGMfoundation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
Install a PyTorch build compatible with your CPU or CUDA environment. The
pretraining environment uses Python 3.10 and PyTorch 2.6.0 with CUDA 12.4;
`requirements-pretrain.txt` specifies its direct dependency versions. See the
[installation guide](docs/PRETRAINING.md#environment) for the corresponding setup.

## Data and model requirements

CGM measurements are expressed in **mmol/L** at **5-minute intervals**. Input
tables contain one recording per row and numerically ordered glucose sample
columns. Preserve missing time positions rather than removing them. Recording
identifiers should distinguish repeated visits from the same participant.

Participant datasets and pretrained checkpoints are not included in this
repository. Training requires user-supplied CGM data; evaluation and representation
extraction additionally require a trained checkpoint. The frozen reference and
calibration used in the **2026-08-26 manuscript results** are included in
[`artifacts/dgp/paper_20260826/`](artifacts/dgp/paper_20260826/README.md).
These parameters require the exact manuscript checkpoint; its filename and
SHA-256 are recorded in the accompanying manifest.

Before running an experiment, set the data, checkpoint, output, and device paths
in its configuration. For pretraining, configure `data.csv_dir`,
`data.hf_dataset_dir`, and `training.save_dir`; dataset construction and training
must use the same `data.hf_dataset_dir`. Detailed input formats and preprocessing
are described in the [pretraining guide](docs/PRETRAINING.md#csv-input) and
[trajectory guide](docs/TRAJECTORY_DGP.md#export) and
[manuscript DGP guide](docs/PAPER_DGP.md).

## Usage

Run commands from the repository root. Example paths refer to user-supplied data
and checkpoints.

### Pretraining

Prepare the dataset and train the forecast model:

```bash
python build_dataset.py --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml
python pretrain_main.py --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml
```

The dataset builder performs CGM cleaning, creates observation masks, and saves a
Hugging Face dataset. Training uses masked forecast losses. The configuration
controls data splits, model architecture, optimization, and logging.

### Forecast evaluation

Evaluate direct forecasts and export recording-level metrics:

```bash
python EvaluatePretrain_with_patient_id_patient_metrics.py --csv_path data/evaluation.csv --checkpoint_path checkpoints/forecast.pt --output_dir outputs/forecast
```

Generate and evaluate 24-hour autoregressive forecasts:

```bash
python Eval_AR_24h_FullContext_clamped.py --csv_path data/evaluation.csv --checkpoint_path checkpoints/forecast.pt --output_dir outputs/forecast_24h
```

See the [forecast evaluation guide](docs/PRETRAINING.md) for checkpoint
compatibility, forecast bounds, masks, and output definitions.

### Clinical and event-conditioned prediction

Select a task configuration and provide its required cohort inputs and checkpoint:

```bash
python LabelPredictionFinetune_CV_main.py --config configs/lab/hpp_new_lab_cgmfoundation_finetune.json
python EventFinetune_main.py --config configs/event_finetune_tasks/CGMacros.json
python GluFormer_ridge_main.py --config configs/gluformer_lab_ridge/hpp_ridge.json
```

Clinical workflows include cross-validation, fine-tuning, and prediction from
frozen CGMfoundation or GluFormer representations. GMI/IGLU and clinical-feature
baselines are also provided. See the [downstream guide](docs/DOWNSTREAM.md) for
task configurations and input conventions.

### Latent trajectory export

For manuscript DGP analysis, place the matching checkpoint in `checkpoints/`,
verify it, and export the ordered patch embeddings using the frozen configuration:

```bash
python Others/verify_paper_dgp_artifacts.py --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt
python ExportTrajectory.py --csv-path data/cohort.csv --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt --config artifacts/dgp/paper_20260826/export_config.json --output-dir outputs/patches/cohort
```

The export contains patch embeddings, original glucose patches, validity and
observation masks, recording IDs, and metadata. The export records the checkpoint
hash, which the manuscript DGP command checks before calculation. The frozen
manuscript configuration requires sample columns `0` through `3455`; see the
[trajectory guide](docs/TRAJECTORY_DGP.md#export) for the complete format.

### Dynamic Glucore Profile

The 2026-08-26 manuscript DGP summarizes latent CGM trajectories using six indices:

| Index | Property characterized |
| --- | --- |
| TOB | Fraction of latent states outside the healthy basin |
| HAD | Mean displacement toward the high-glycemia direction |
| DynVar | Within-recording variability along the glycemic axis |
| DynHBGI | Positive-tail burden in the high-glycemia region |
| TPLE | One-hour growth of separation between matched pulse-response trajectories |
| SLI | Log-transformed nocturnal relaxation time of the Mahalanobis radius |

Apply the published reference and calibration directly to the exported cohort.
No evaluation labels or refitting are required:

```bash
python DGP_paper.py --patch-dir outputs/patches/cohort --reference artifacts/dgp/paper_20260826/glycemic_basin_reference.npz --calibration artifacts/dgp/paper_20260826/calibration.json --output-dir outputs/dgp/cohort
```

The analysis also exports the dynamic risk score:

```text
DRS = z_NGT(HAD) + 0.25 × [(1 h) × TPLE + SLI]
```

Here `z_NGT(HAD)` uses the frozen Nicheng NGT median and robust scale.
DynVar and DynHBGI use frozen NGT means and sample standard deviations.
TPLE is measured in `h^-1`; SLI is `log(1 + tau)`, where `tau` is the
night-fixed lag-one relaxation time in 15-minute steps. Missing trajectory-pair
or relaxation estimates remain missing. DRS uses the same fixed scale across
cohorts. The warning formula and follow-up window were selected exploratorily;
prospective performance requires independent validation.

The DGP implementation assumes 15-minute patch spacing and a first patch endpoint
at noon, corresponding to the supplied 12-hour patches and a midnight recording
origin. Outputs include `dgp_metrics.csv`, `calibration.json`, and
`run_manifest.json`. See the [manuscript DGP guide](docs/PAPER_DGP.md) for
definitions, checkpoint verification, missing-value rules, and reuse with a new
encoder. Trajectory extraction uses offline imputation; interpret the profiles
within that preprocessing framework. The earlier radial PLE and expansion/speed
SLI workflow remains documented in the [legacy guide](docs/TRAJECTORY_DGP.md);
its `DGP_main.py` command uses a different metric definition and calibration.
Observed CPU replay differences from the archived trajectories are recorded in
the [validation report](artifacts/dgp/paper_20260826/validation.json).

## Repository structure

| Path | Contents |
| --- | --- |
| `models/` | CGMfoundation model components |
| `utils/` | Data processing, losses, metrics, and training utilities |
| `configs/` | Pretraining and downstream experiment configurations |
| `LabelPredictionFinetune/` | Clinical prediction and cross-validation |
| `EventFinetune/` | Event-conditioned models and result exports |
| `SampleRepresentation/`, `Trajectory/` | Representation learning and trajectory modeling |
| `GlucosePredictionBaseline/` | Glucose forecasting baselines |
| `DGP/` | Dynamic glycemic profile calculation |
| `artifacts/dgp/paper_20260826/` | Frozen manuscript DGP reference, calibration, and export configuration |
| `docs/` | Detailed installation, input, and workflow documentation |

For questions about the code or reproducibility, please
[open an issue](https://github.com/YurunLu/CGMfoundation/issues).
