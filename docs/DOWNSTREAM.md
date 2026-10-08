# Downstream source organization

Target repository: https://github.com/YurunLu/CGMfoundation.git
Inspected base: `572cea1911826872a7fb8db500c31a54dca6f146`.

## Mapping to the existing format

| Original working file | Submitted location |
| --- | --- |
| `code/CGMfoundation_lab.py` | `LabelPredictionFinetune_CV_main.py` |
| `code/lab_cv_runner.py` | `LabelPredictionFinetune/lab_cv_runner.py` |
| `code/CGMfoundation_lab_ridge.py` | `CGMfoundation_ridge_main.py` |
| `code/GluFormer_lab_ridge.py` | `GluFormer_ridge_main.py` |
| `code/GMI_IGLU_lab_ridge.py` | `GMI_IGLU_main.py` |
| `code/LabelFeature_lab_ridge.py` | `LabelFeature_ridge_main.py` |
| `code/CGMfoundation_lab_scheduler.py` | `LabelPredictionFinetune_scheduler.py` |
| `configs/lab/GluFormer_lab_ridge_scheduler.py` | `GluFormer_ridge_scheduler.py` |
| `code/EventFinetune_main.py` | `EventFinetune_main.py` |
| `code/event_finetune_scheduler.py` | `EventFinetune_scheduler.py` |
| `code/Eval_Event_AR_24h_FullContext.py` | `EventFinetune_AR_main.py` |
| `code/export_event_*.py` | `EventFinetune/export_event_*.py` |
| `code/export_lab_*.py` | `LabelPredictionFinetune/export_lab_*.py` |
| `code/train_patchtst_baseline.py` | `GlucosePredictionBaseline/PatchTST_main.py` |
| `models/base_model.py` | `models/downstream_base_model.py` |
| `utils/data_utils.py` | `utils/downstream_data_utils.py` |
| `utils/train_utils.py` | `utils/downstream_train_utils.py` |

The three event model/data/loss files retain their `EventFinetune/` locations and use
the working versions. Existing upstream shared modules are preserved so that older
pretraining and representation scripts continue to import their own versions.

`DOWNSTREAM_LAYOUT.json` records every source path, destination, original hash,
submitted hash and exact import/path replacement. No architecture, loss, optimizer,
preprocessing, data-split, normalization, training loop, metric or checkpoint-selection
logic was modified. Files requiring no import/path adjustment remain byte-identical.

```bash
python Others/verify_downstream_layout.py --original-root D:/last
```

## Original data and configuration conventions

Clinical data locations are defined in `LabelPredictionFinetune/lab_cv_runner.py`.
The original `data_lab/` hierarchy, numeric time-step column names and label tables
are retained. GluFormer/GMI read prepared folds at `data_lab/lab_target_cv/`; configure
other runners to use the same experiment's fold location when comparing them.

Event input files and channels follow the selected JSON. The event table retains the
original CSV index convention (`index_col=0`), `device_id`, `TimeIndex` and event columns.
No input adapter was added. Keep the original units, recording order and cohort labels.

Config directory depth is unchanged, preserving existing relative paths. Runner paths
such as `../../code/CGMfoundation_lab.py` now point to
`../../LabelPredictionFinetune_CV_main.py`. Absolute input/checkpoint paths are retained
as experiment references and need local equivalents on a different machine.

One source file, `configs/cgmfoundation_lab_ridge/hpp_cgmanchored_incident_180d_hba1c_available.json`,
contains a shell command appended after its JSON object. It is retained as supplied;
it is not a directly loadable JSON config. No experiment content was repaired during
this organization task.

## Branch integration

The changes are integrated on a working branch based on the upstream repository.
Clinical cross-validation is added through `LabelPredictionFinetune_CV_main.py`
and modules in `LabelPredictionFinetune/`. The event entry point and three existing
`EventFinetune/` modules are updated to the original working versions. Downstream
model and utility versions use separate filenames to preserve existing upstream
imports. Configuration files are integrated under `configs/`.

Review the branch diff and open a pull request into the intended target branch.
The source ZIP uses paths relative to the upstream
root and contains only this contribution, not a complete standalone upstream clone.
To regenerate it:

```bash
python Others/package_downstream.py --output cgm_downstream-source.zip
```

Participant data, split tables, checkpoints, logs, outputs and manuscript materials
are excluded from the package. `requirements-tested.txt` describes only the local CPU
validation environment; it does not replace the original experiment environment.
