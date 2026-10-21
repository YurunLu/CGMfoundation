# Manuscript DGP: frozen 2026-08-26 definitions

The manuscript workflow uses `TOB`, `HAD`, `DynVar`, `DynHBGI`, `TPLE`, and
`SLI`. Run it with `DGP_paper.py` and the files in
[`artifacts/dgp/paper_20260826/`](../artifacts/dgp/paper_20260826/README.md).
The earlier `DGP_main.py` workflow has different PLE, SLI, and DRS definitions.
Its calibration is not interchangeable with this release.

## Published files

| File | Purpose |
| --- | --- |
| `glycemic_basin_reference.npz` | Original frozen embedding normalization, PCA, healthy basin, and glycemic direction |
| `calibration.json` | Frozen NGT normalization and the manuscript metric definition |
| `export_config.json` | Encoder and patch settings for trajectory extraction |
| `reference_manifest.json` | Reference, calibration, configuration, and checkpoint SHA-256 checksums |
| `validation.json` | Aggregate numerical checks against manuscript source results |

The reference is the original NPZ, copied byte for byte. The calibration JSON
consolidates the original reference-cohort statistics used by the manuscript
scripts. The reference was fitted from Nicheng latent states; its healthy region
and metric normalization use 200 NGT recordings. Calibration does not use labels
from the cohort being evaluated. Participant-level data are not included.

## Run the manuscript workflow

Install the repository dependencies and obtain the exact checkpoint named in
the artifact manifest. Run from the repository root:

```bash
python Others/verify_paper_dgp_artifacts.py --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt
python ExportTrajectory.py --csv-path data/cohort.csv --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt --config artifacts/dgp/paper_20260826/export_config.json --output-dir outputs/patches/cohort
python DGP_paper.py --patch-dir outputs/patches/cohort --reference artifacts/dgp/paper_20260826/glycemic_basin_reference.npz --calibration artifacts/dgp/paper_20260826/calibration.json --output-dir outputs/dgp/cohort
```

Use a new or empty output directory. The CGM input requires recording IDs and
sample columns `0` through `3455`, in mmol/L at five-minute intervals. The frozen
configuration uses 144-sample history patches, stride 3, and a 12-sample forecast
reservation. The first patch endpoint is at noon for a midnight recording
origin. See the [trajectory format](TRAJECTORY_DGP.md#export) for observation
masks, interpolation, and recording selection.

The frozen export configuration sets `export.sample_column_start` to `0`,
matching the original 3,456-sample input and 1,101 patch slots. The legacy default
starts at column `1`; shifting the sample origin changes the exported states.
It also fixes `export.validity_policy` to `any-observed`, matching the manuscript
arrays: a patch is valid when it contains at least one originally observed sample.
Such a patch may extend beyond the final observation. Model inputs use linear
interpolation and edge extension. The public exporter retains original NaNs in
`patch_raw.npy` and writes observation masks; the archived manuscript raw-patch
arrays stored the already imputed model inputs. This storage difference does not
change the exported encoder states.

`DGP_paper.py` checks the checkpoint hash and sample-column origin in the trajectory metadata against
the frozen calibration. It also checks the reference hash and export settings.
Equal tensor dimensions or a matching filename do not establish checkpoint
compatibility. Legacy trajectories without an encoder hash must be exported
again with the verified checkpoint before running the public command.

Outputs are `dgp_metrics.csv`, a copy of `calibration.json`, and
`run_manifest.json`. The table retains raw components, valid-patch counts,
TPLE pair counts, SLI correlation and transition counts, and completeness flags.
The run manifest records the checkpoint, reference, and calibration hashes.

## Numerical definitions

For latent state `e_t`, define

```text
PC_t = ((e_t - mean) / sd) @ components
V_t = (PC_t - mu)^T inv_cov (PC_t - mu)
s_t = (PC_t - mu)^T u_high
a_t = (s_t - high_mean) / high_sd
```

| Metric | Definition |
| --- | --- |
| TOB | `mean(V_t > threshold)` |
| HAD | `mean(a_t)` |
| DynVar | NGT mean/sample-SD normalization of `sd(a_t, ddof=1)` |
| DynHBGI | NGT mean/sample-SD normalization of `mean(max(a_t, 0)^2) + 1.5*mean(s_t > high_q95)` |
| TPLE | Slope of the mean log separation between matched pulse-response trajectories over one hour, in `h^-1` |
| SLI | `log(1 + tau)`, with `tau = -1/log(rho)` in 15-minute steps |

TPLE detects outward latent pulses relative to the individual nighttime median
attractor, with a whole-recording fallback when fewer than 12 nighttime patches
are available. Each pulse peak is matched to its nearest eligible pulse peak in
five PCA dimensions, using the healthy precision submatrix. Both responses need
an intact four-step follow-up. The Theiler exclusion is greater than 12 patches
(three hours); the clock tolerance is 96 patches. One matched pair is sufficient
to define an estimate, and pair counts remain available for sensitivity analysis.
TPLE is not centered by an NGT quantile and is not an asymptotic maximal Lyapunov
exponent.

SLI uses the Mahalanobis radius. Consecutive nighttime transitions run from
00:00 through 06:00, including a transition ending at 06:00. Left and right
radius values are centered separately within each night before their pooled
lag-one Pearson correlation `rho` is computed. At least 24 valid transitions
and `0 < rho < 1` are required. Missing transitions are not compressed; absent
TPLE pairs or invalid relaxation estimates produce missing values.

The manuscript risk score is

```text
DRS = (HAD - HAD_NGT_median) / HAD_NGT_robust_scale
      + 0.25 * [(1 h) * TPLE + SLI]
```

The fixed HAD center is `-0.1031305349141052`, and the robust scale is
`0.4772495026293811`, as read by the original computation. The original Figure 05 description
called this normalization mean/SD; the source code and numerical results use
median/robust scale. This release follows the numerical results. The score does
not normalize against the evaluated cohort. The formula and follow-up window
were selected exploratorily and require independent prospective validation.

## Reuse with another encoder

The published NPZ and calibration are tied to the manuscript checkpoint. For a
new encoder, export a designated reference cohort, fit a new reference, and
calibrate the manuscript formulas on that cohort:

```bash
python DGP_reference.py --patch-dir outputs/patches/reference --labels data/reference_labels.csv --output-dir outputs/new_reference
python DGP_paper.py --patch-dir outputs/patches/reference --reference outputs/new_reference/glycemic_basin_reference.npz --calibrate-labels data/reference_labels.csv --output-dir outputs/new_calibration
python DGP_paper.py --patch-dir outputs/patches/test --reference outputs/new_reference/glycemic_basin_reference.npz --calibration outputs/new_calibration/calibration.json --output-dir outputs/new_test
```

Labels need unique `sample_id` (or `id`) and `Type` values `NGT`, `PreDM`, or
`DM`. Keep the reference and calibration fixed for subsequent evaluation. This
creates an encoder-specific analysis; it does not reproduce the frozen
manuscript coordinate system or validate clinical transportability.

## Verification

```bash
python Others/verify_paper_dgp_artifacts.py
python Others/verify_paper_dgp_artifacts.py --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt --patch-dir outputs/patches/cohort
python Others/smoke_paper_dgp.py
```

The first command validates bundled file hashes, reference geometry, calibration,
and copied numerical kernels. Optional checkpoint verification checks the file
SHA-256 and strict model loading. The validation report contains only aggregate
errors and source-file hashes. It covers the 821 three-group Nicheng records in
the manuscript figure and the 208-record Colas DRS source table. Archived
reference data and checkpoint inference checks are recorded separately in the
artifact manifest. Three full-trajectory CPU comparisons retained maximum
embedding differences of approximately `0.00172` against the archived arrays;
they did not pass a strict `rtol=1e-4, atol=1e-5` comparison. Checkpoint identity,
hyperparameters, imputed inputs, and validity masks were verified separately.
A synthetic-input check independently found CPU/CUDA differences with this
checkpoint; it does not establish the cause of every archived-trajectory
difference. Exact figure reproduction uses the archived trajectories. Retain
the device and runtime when exporting new states.
