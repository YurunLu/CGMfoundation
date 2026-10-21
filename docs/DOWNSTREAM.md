# Clinical and event-conditioned prediction

CGMfoundation supports clinical classification and regression through fine-tuning
or prediction from frozen representations. Event-conditioned models incorporate
annotated meals and other events. Run commands from the repository root after
installing `requirements.txt` and a compatible PyTorch build.

## Entry points

| Entry point | Purpose |
| --- | --- |
| `LabelPredictionFinetune_CV_main.py` | Clinical fine-tuning with cross-validation |
| `CGMfoundation_ridge_main.py` | Prediction from frozen CGMfoundation representations |
| `GluFormer_ridge_main.py` | Prediction from frozen GluFormer representations |
| `GMI_IGLU_main.py` | Prediction from GMI and IGLU glucose summaries |
| `LabelFeature_ridge_main.py` | Prediction from clinical features |
| `EventFinetune_main.py` | Event-conditioned fine-tuning |
| `EventFinetune_AR_main.py` | Autoregressive event-conditioned evaluation |

Clinical model components and fold preparation are in `LabelPredictionFinetune/`.
Event data processing, model components, losses, and prediction exports are in
`EventFinetune/`.

## Clinical inputs

Provide CGM tables and clinical labels under `data_lab/<cohort>/`. Dataset file
locations and label-table combinations are defined in
`LabelPredictionFinetune/lab_cv_runner.py`. The supplied cohorts include Zhao,
Nicheng, HPP, and Colas. Select the cohort and variant through the configuration.
Keep glucose units, numeric sample-column ordering, recording IDs, and label
encodings consistent with the selected dataset definition.

Target specifications use `cls:<column>` for classification and `reg:<column>`
for regression. A target object can also specify included label values and an
explicit label mapping. The `targets` list selects the analyses to run.

Prepared folds are stored under `data_lab/lab_target_cv/` by default. Set
`split.split_root`, `split.n_splits`, and `split.seed` for the intended protocol.
Reuse the same prepared folds across model comparisons. For repeated recordings,
check that fold assignments respect the participant grouping required by the
evaluation protocol.

## Clinical fine-tuning

Start with a configuration in `configs/lab/`. Set `pretrain.ckpt_path`,
`training.out_root`, and the device settings before running:

```bash
python LabelPredictionFinetune_CV_main.py --config configs/lab/hpp_new_lab_cgmfoundation_finetune.json
```

The configuration controls target selection, fold preparation, optimization,
encoder freezing, unfreezing, early stopping, and prediction exports. Use model
settings compatible with the checkpoint. The wrapper also provides target listing
and fold-preparation options; inspect them with:

```bash
python LabelPredictionFinetune_CV_main.py --help
```

## Frozen representations and feature baselines

Choose a task configuration and provide the checkpoint and input files it requires:

```bash
python GluFormer_ridge_main.py --config configs/gluformer_lab_ridge/hpp_ridge.json
python CGMfoundation_ridge_main.py --help
python GMI_IGLU_main.py --help
python LabelFeature_ridge_main.py --help
```

GluFormer and GMI/IGLU workflows consume prepared task folds. Point all compared
methods to the same experiment's split directory. Clinical-feature baselines need
the predictor columns requested by their configuration. Inspect each entry point
for its checkpoint, output, and feature-selection options.

## Event-conditioned inputs and training

Configurations in `configs/event_finetune_tasks/` specify the following inputs:

| Key | Input or setting |
| --- | --- |
| `data_csv` | CGM and event table |
| `onehot_csv` | Aligned encoded event channels |
| `pretrain_ckpt` | Compatible pretrained encoder checkpoint |
| `out_dir` | Checkpoints, metrics, and prediction exports |
| `event_cols` | Event variables used by the model |
| `glucose_scale` | Input glucose scaling |
| `device` | Training device |

The event reader uses the first CSV column as its index and expects the dataset's
`device_id`, `TimeIndex`, and configured event columns. Keep recording order and
event-channel alignment consistent between the input tables. Set the file paths
and device in the selected configuration, then run:

```bash
python EventFinetune_main.py --config configs/event_finetune_tasks/CGMacros.json
```

Prediction export is controlled by `export_predictions`, `prediction_split`, and
the checkpoint-saving options. Additional event and clinical export utilities are
provided in `EventFinetune/` and `LabelPredictionFinetune/`.

## Outputs and reproducibility

Output locations are configured for each task. Preserve the configuration,
checkpoint identity, fold assignments, task definitions, metrics, and any exported
predictions together. Participant data and trained checkpoints are not included
in this repository. Input paths in the example configurations must be set to
the files available in the user's environment.

See [workflow checks](DOWNSTREAM_VALIDATION.md) for source-file and
synthetic execution commands, and the [pretraining guide](PRETRAINING.md) for
encoder training and forecast evaluation.
