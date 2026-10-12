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
extraction additionally require a compatible trained checkpoint. DGP analysis
requires a healthy reference and calibration, which can be fitted using the
provided scripts and a designated reference cohort.

Before running an experiment, set the data, checkpoint, output, and device paths
in its configuration. For pretraining, configure `data.csv_dir`,
`data.hf_dataset_dir`, and `training.save_dir`; dataset construction and training
must use the same `data.hf_dataset_dir`. Detailed input formats and preprocessing
are described in the [pretraining guide](docs/PRETRAINING.md#csv-input) and
[trajectory and DGP guide](docs/TRAJECTORY_DGP.md).

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

Export the ordered patch embeddings for a cohort:

```bash
python ExportTrajectory.py --csv-path data/cohort.csv --checkpoint-path checkpoints/forecast.pt --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml --output-dir outputs/patches/cohort
```

The export contains patch embeddings, original glucose patches, validity and
observation masks, recording IDs, and metadata. Use the training configuration
associated with the checkpoint. This entry point requires sample columns `1`
through `max_seq_len - 1`; see the
[trajectory guide](docs/TRAJECTORY_DGP.md#export) for the complete format.

### Dynamic Glucore Profile

DGP summarizes latent CGM trajectories using six complementary indices:

| Index | Property characterized |
| --- | --- |
| TOB | Fraction of latent states outside the healthy basin |
| HAD | Mean displacement toward the high-glycemia direction |
| DynVar | Within-recording variability along the glycemic axis |
| DynHBGI | Positive-tail burden in the high-glycemia region |
| PLE | Finite-time radial response following detected latent pulses |
| SLI | Local stability loss from radial expansion and trajectory speed |

Fit the reference and calibration on designated training/reference recordings,
then apply the frozen files to evaluation cohorts. Export the reference recordings
to `outputs/patches/train` using the same encoder and configuration as above.
The reference label table requires a unique `sample_id` (or `id`) and `Type`
values of `NGT`, `PreDM`, or `DM`.

```bash
python DGP_reference.py --patch-dir outputs/patches/train --labels data/train_labels.csv --output-dir outputs/reference
python DGP_main.py --patch-dir outputs/patches/train --reference outputs/reference/glycemic_basin_reference.npz --calibrate-labels data/train_labels.csv --output-dir outputs/dgp/train
python DGP_main.py --patch-dir outputs/patches/cohort --reference outputs/reference/glycemic_basin_reference.npz --calibration outputs/dgp/train/calibration.json --output-dir outputs/dgp/cohort
```

The analysis also exports the dynamic risk score:

```text
DRS = z(HAD) + 0.5 × max(z(PLE), 0) × max(z(SLI), 0)
```

By default, DRS uses the evaluated cohort's mean and standard deviation. Use
`--drs-reference training` to apply frozen training normalization. Compare scores
using a consistent normalization scheme and fit references without evaluation
labels. Missing pulse-response estimates are retained as missing values.

The DGP implementation assumes 15-minute patch spacing and a first patch endpoint
at noon, corresponding to the supplied 12-hour patches and a midnight recording
origin. It retains the signed PLE and SLI scales. Outputs include
`dgp_metrics.csv`, `calibration.json`, and `run_manifest.json`. See the
[DGP guide](docs/TRAJECTORY_DGP.md#numerical-definitions) for formulas, timing,
calibration, and interpretation. Trajectory extraction uses offline imputation;
the exported profiles should be interpreted within that preprocessing framework.

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
| `docs/` | Detailed installation, input, and workflow documentation |

For questions about the code or reproducibility, please
[open an issue](https://github.com/YurunLu/CGMfoundation/issues).
