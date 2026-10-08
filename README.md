# CGMfoundation

Research code for continuous glucose monitoring foundation models.

## Clinical downstream tasks and event fine-tuning

The downstream experiment code follows the existing repository layout: executable entry
points in the root, task modules in their corresponding directories, shared models
in `models/`, and shared utilities in `utils/`.

```text
LabelPredictionFinetune_CV_main.py     Clinical classification and regression
CGMfoundation_ridge_main.py           Frozen CGMfoundation representation baseline
GluFormer_ridge_main.py               Frozen GluFormer representation baseline
GMI_IGLU_main.py                      GMI / IGLU baseline
LabelFeature_ridge_main.py            Clinical-label feature baseline
LabelPredictionFinetune_scheduler.py  Clinical task scheduler
GluFormer_ridge_scheduler.py          GluFormer task scheduler
EventFinetune_main.py                 Event-conditioned fine-tuning
EventFinetune_scheduler.py            Event task scheduler
EventFinetune_AR_main.py              Event autoregressive evaluation
LabelPredictionFinetune/              Clinical runner and result exports
EventFinetune/                        Event model, data, loss and result exports
GlucosePredictionBaseline/           PatchTST and existing upstream baselines
models/                              Original upstream and downstream model versions
utils/                               Original upstream and downstream utilities
configs/                             Original experiment configurations
Others/                              Source-layout verification and smoke checks
```

The existing `LabelPredictionFinetune_main.py` remains the upstream implementation;
`LabelPredictionFinetune_CV_main.py` is the submitted clinical cross-validation workflow.
The event entry point and its three task modules use the submitted working versions.
`models/base_model.py`, `utils/data_utils.py` and `utils/train_utils.py` remain upstream
versions; downstream experiments import their separately named `downstream_*` versions.

## Install and run

Use Python 3.12 for the locally checked environment. Install a PyTorch build appropriate
for your CPU/CUDA environment and the dependencies below:

```bash
python -m pip install -r requirements.txt
python LabelPredictionFinetune_CV_main.py --help
python EventFinetune_main.py --help
```

Run experiment commands from the repository root:

```bash
python LabelPredictionFinetune_CV_main.py --config configs/lab/hpp_new_lab_cgmfoundation_finetune.json
python EventFinetune_main.py --config configs/event_finetune_tasks/CGMacros.json
python GluFormer_ridge_main.py --config configs/gluformer_lab_ridge/hpp_ridge.json
```

These commands require the original cohort inputs and compatible trained checkpoints.
The configs retain original training parameters and server data/checkpoint paths. On
another machine set paths/device options in local config copies or supported CLI
arguments. Only references to relocated runner scripts were adjusted in the submitted
configs; no training settings were changed.

```bash
python Others/verify_downstream_layout.py
python Others/smoke_downstream.py
```

The smoke check creates synthetic inputs and small random checkpoints in an isolated
copy of the submitted source. It exercises the original training implementations, not
manuscript performance. See [the source mapping and usage notes](docs/DOWNSTREAM.md)
and [the validation record](docs/DOWNSTREAM_VALIDATION.md).
