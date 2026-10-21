# Downstream workflow checks

These commands check source-file integrity and execution of clinical and
event-conditioned workflows. Install the repository dependencies and run from
the repository root.

## Source integrity

```bash
python Others/verify_downstream_layout.py
```

The verifier compares the 65 downstream files with the SHA-256 checksums in
[`DOWNSTREAM_LAYOUT.json`](DOWNSTREAM_LAYOUT.json). The manifest identifies
repository files and their expected contents.

## Command-line interfaces

```bash
python LabelPredictionFinetune_CV_main.py --help
python CGMfoundation_ridge_main.py --help
python GluFormer_ridge_main.py --help
python GMI_IGLU_main.py --help
python LabelFeature_ridge_main.py --help
python EventFinetune_main.py --help
python EventFinetune_AR_main.py --help
```

Use these interfaces to check accepted arguments before supplying cohort inputs
and checkpoints. Dataset and task conventions are described in the
[downstream guide](DOWNSTREAM.md).

## Synthetic execution

```bash
python Others/smoke_downstream.py
```

The check creates synthetic CGM and labels, small random model checkpoints, and
temporary configurations. It exercises clinical binary and multiclass
classification, regression, feature baselines, event-conditioned training, and
prediction export on the CPU. Its temporary data are separate from cohort inputs.

Synthetic execution checks software behavior. Scientific performance estimates
require the study data, trained checkpoints, and specified evaluation protocol.
The optional `requirements-tested.txt` file lists pinned workflow-check
dependencies; the pretraining environment is specified separately in
`requirements-pretrain.txt`.
