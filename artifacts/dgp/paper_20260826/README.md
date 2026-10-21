# Manuscript DGP reference parameters

This directory contains the frozen parameters for `TOB`, `HAD`, `DynVar`,
`DynHBGI`, `TPLE`, `SLI`, and DRS. Use
[`DGP_paper.py`](../../../DGP_paper.py) with the accompanying export configuration.
The [DGP guide](../../../docs/PAPER_DGP.md) provides the complete workflow and
metric definitions.

| File | Contents |
| --- | --- |
| [glycemic_basin_reference.npz](glycemic_basin_reference.npz) | Embedding normalization, ten PCA axes, healthy basin, and glycemic direction for the 256-dimensional encoder |
| [calibration.json](calibration.json) | Frozen NGT normalization and metric-definition settings |
| [export_config.json](export_config.json) | Matching model, sample-column, and validity-mask settings |
| [reference_manifest.json](reference_manifest.json) | Checkpoint identity, required export settings, and file checksums |
| [validation.json](validation.json) | Summary of parameter and manuscript-calculation checks |

The required checkpoint is
`best_forecast_20260420_105114_dim256_patch144_stride3.pt`. Obtain the weights
separately and verify their SHA-256 against the manifest before exporting
trajectories. The frozen reference and calibration use the Nicheng reference
cohort, including 200 NGT recordings for healthy-reference statistics.

TPLE measures growth of separation between matched pulse-response trajectories.
SLI uses log-transformed nighttime relaxation time. HAD normalization for DRS uses
the frozen NGT median and robust scale. Use the same reference and calibration for
all evaluation recordings; evaluation labels are not required.

These files contain aggregate parameters. Participant CGM recordings and clinical
labels are not distributed in this repository.
