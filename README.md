# CGMfoundation

Research code for foundation models of continuous glucose monitoring (CGM) data,
including forecast pretraining, patient-level forecast evaluation, 24-hour
autoregressive evaluation, and clinical downstream experiments.

## Pretraining and forecast evaluation

The pretraining model encodes overlapping CGM history patches with a CNN and a
Transformer with causal attention. Positional and temporal embeddings are configurable.
The supplied experiment configuration uses 12-hour patches and predicts the next hour
at 5-minute resolution. Missing history values are imputed before entering the model;
observed-target masks control the training loss and evaluation metrics.

The [pretraining guide](docs/PRETRAINING.md) describes the historical software
environment, CSV format, preprocessing, configuration, training, and both evaluation
workflows. After installing dependencies and adapting the paths in the YAML files,
run commands from the repository root:

```bash
python build_dataset.py --config configs/config.yaml
python pretrain_main.py --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml
```

Use `EvaluatePretrain_with_patient_id_patient_metrics.py` for direct forecast metrics
and patient-level outputs, and `Eval_AR_24h_FullContext_clamped.py` for a 24-hour
autoregressive rollout with bounded predictions. Both require an external evaluation
CSV and a compatible trained checkpoint.

## Repository layout

| Path | Purpose |
| --- | --- |
| `build_dataset.py` | Clean CSV sequences and save a Hugging Face dataset with observation masks |
| `pretrain_main.py` | Train the forecast model with Accelerate |
| `EvaluatePretrain_with_patient_id_patient_metrics.py` | Direct forecast evaluation and patient-level metrics |
| `Eval_AR_24h_FullContext_clamped.py` | Full-context 24-hour autoregressive evaluation |
| `models/base_model.py` | Pretraining model and embedding components |
| `utils/data_cleaning.py`, `utils/loss_utils.py`, `utils/train_utils.py` | CGM cleaning, masked losses and metrics, and learning-rate scheduling |
| `configs/` | Pretraining YAML files and downstream experiment configurations |
| `LabelPredictionFinetune/` | Clinical cross-validation runner and result exports |
| `EventFinetune/` | Event model, data preparation, losses, and result exports |
| `SampleRepresentation/`, `Trajectory/` | Representation and trajectory workflows |
| `GlucosePredictionBaseline/` | Forecast baselines, including PatchTST |
| `Others/` | Downstream source-layout verification, packaging, and smoke checks |
| `docs/` | Pretraining instructions and downstream source/validation records |

Downstream experiments use separately named `models/downstream_base_model.py` and
`utils/downstream_*` modules where required, preserving their existing implementations.
The original `LabelPredictionFinetune_main.py` remains available;
`LabelPredictionFinetune_CV_main.py` provides the clinical cross-validation workflow.

## Installation

Historical pretraining used **Python 3.10 and PyTorch 2.6.0 with CUDA 12.4**. Follow
[the pretraining installation instructions](docs/PRETRAINING.md#environment) and
`requirements-pretrain.txt` to use the recorded direct dependency versions.

For the broader workflows, install a PyTorch build appropriate for your CPU/CUDA
environment, then install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements-tested.txt` records the separate Python 3.12 CPU environment used for
the downstream checks. It is not the historical GPU pretraining environment.

## Data, checkpoints, and configuration

Participant datasets and pretrained checkpoints are **not distributed in this
repository**. Pretraining can start from random initialization with your own data.
Evaluation and workflows that use pretrained representations require a compatible
checkpoint that you train or obtain separately.

The supplied configurations retain experiment settings and absolute server paths.
Adapt data locations, checkpoint locations, output directories, and device options
to your machine before running them. For pretraining, update `data.csv_dir`,
`data.hf_dataset_dir`, and `training.save_dir`; the builder and trainer must refer to
the same Hugging Face dataset directory. See the
[configuration notes](docs/PRETRAINING.md#configuration).

## Clinical downstream tasks and event fine-tuning

Existing downstream workflows cover clinical classification and regression, frozen
CGMfoundation and GluFormer representations, GMI/IGLU and clinical-label baselines,
event-conditioned fine-tuning, and event autoregressive evaluation. Their JSON
configurations define the required cohort inputs and checkpoint paths.

```bash
python LabelPredictionFinetune_CV_main.py --help
python EventFinetune_main.py --help
python LabelPredictionFinetune_CV_main.py --config configs/lab/hpp_new_lab_cgmfoundation_finetune.json
python EventFinetune_main.py --config configs/event_finetune_tasks/CGMacros.json
python GluFormer_ridge_main.py --config configs/gluformer_lab_ridge/hpp_ridge.json
```

The recorded downstream checks used Windows, Python 3.12.10, and CPU: source-integrity
verification, 17 entry-point help checks, and a synthetic smoke run passed. These
checks cover execution with synthetic inputs and small random checkpoints; they do
not validate GPU/multi-GPU training or manuscript performance. To run the checks:

```bash
python Others/verify_downstream_layout.py
python Others/smoke_downstream.py
```

The smoke check creates its inputs and checkpoints in an isolated source copy.
See [downstream organization and usage](docs/DOWNSTREAM.md) and
[the downstream validation record](docs/DOWNSTREAM_VALIDATION.md) for the scope
and provenance of these checks.
