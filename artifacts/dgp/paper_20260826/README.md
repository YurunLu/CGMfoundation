# Frozen manuscript DGP parameters

These parameters reproduce the DGP definitions in the 2026-08-26 manuscript
result package: `TOB`, `HAD`, `DynVar`, `DynHBGI`, `TPLE`, `SLI`, and the
associated DRS. Use [`DGP_paper.py`](../../../DGP_paper.py), as described in the
[manuscript DGP guide](../../../docs/PAPER_DGP.md).

- [glycemic_basin_reference.npz](glycemic_basin_reference.npz): original frozen
  reference, with ten PCA dimensions and a 256-dimensional encoder input.
- [calibration.json](calibration.json): frozen metric normalization and definition.
- [export_config.json](export_config.json): matching trajectory-export settings.
- [reference_manifest.json](reference_manifest.json): checkpoint identity,
  artifact hashes, and provenance.
- [validation.json](validation.json): aggregate comparisons with manuscript results.

The required checkpoint is
`best_forecast_20260420_105114_dim256_patch144_stride3.pt`. Verify its SHA-256
against the manifest before exporting trajectories. The weights must be obtained
separately. The NPZ and JSON contain aggregate reference parameters; no individual
CGM recordings or clinical labels are distributed here.

The original NPZ is preserved byte for byte. The JSON consolidates the original
NGT statistics, including the HAD median/robust-scale normalization actually used
in the source computations. TPLE uses matched pulse-response trajectories; SLI
uses log-transformed nighttime relaxation time. Earlier radial PLE and
expansion/speed SLI calibrations are incompatible with these definitions.
