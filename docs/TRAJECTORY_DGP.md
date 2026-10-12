# Patch trajectory export and current DGP

Run commands from the repository root after installing `requirements.txt` and a
compatible PyTorch build. These entry points synchronize the current local
numerical definitions into a portable workflow. They do not run the historical
candidate search, clinical model selection, or manuscript plotting pipelines.

## Export

```bash
python ExportTrajectory.py --csv-path data/cohort.csv --checkpoint-path checkpoints/forecast.pt --config configs/config_12h_3_12_256_12_8_1024_2_1.2_1all.yaml --output-dir outputs/patches/cohort --device cpu --batch-size 16
```

The CSV requires an `id` column (configurable with `--id-column`) and numeric
sample columns `1` through `max_seq_len - 1`, in mmol/L. IDs identify recordings;
repeated visits need distinct recording IDs. IDs are read as strings, including
leading zeroes. More than `max_seq_len * 0.1` observed samples are retained by
default, following `GetMeanPatchEmb.py`. Empty/insufficient recordings are counted
in metadata. No shuffle or mean pooling is applied.

The config needs the training `model` section, including `patch_size`, `stride`,
`forecast_steps`, `embed_dim`, `max_seq_len`, depth, heads, and embedding flags.
YAML and JSON are supported. Model loading is strict; a mismatched architecture
or checkpoint fails instead of producing partially initialized trajectories.
Checkpoint wrappers `model_state_dict`, `state_dict`, and `model` are supported.
Checkpoint files must be from a trusted training source and loadable with
PyTorch's `weights_only=True` loader.

| File | Shape/content |
| --- | --- |
| `patch_embeddings.npy` | float32 `[recording, patch, embedding_dimension]` |
| `patch_raw.npy` | float32 `[recording, patch, patch_size]`, original values and NaNs |
| `patch_valid_mask.npy` | boolean `[recording, patch]`, existing dataset validity policy |
| `patch_observed_mask.npy` | boolean `[recording, patch, patch_size]`, raw observed samples |
| `sample_ids.csv` | one `sample_id` per retained recording, same row order |
| `meta.json` | model config, input/checkpoint/config hashes, timing, filtering and mask policies |

The existing dataset fills internal NaNs by linear interpolation and extends edge
values. Its patch mask checks that the patch ends before the last observed sample;
internal missing values do not invalidate the whole patch. Observation masks are
exported separately. This is an offline representation workflow: the inherited
interpolation can use later observations and bridge internal missing intervals.
Do not assume this preprocessing is a causal online warning pipeline.

As in the existing exporter, patch starts reserve `forecast_steps` samples after
each history patch, even though only embeddings are exported. Arrays are written
in batches using memory maps. Use a new/empty output directory; existing exports
are not overwritten. `meta.json` is written last, after successful array export.

## Reference and calibration

```bash
python DGP_reference.py --patch-dir outputs/patches/train --labels data/train_labels.csv --output-dir outputs/reference
python DGP_main.py --patch-dir outputs/patches/train --reference outputs/reference/glycemic_basin_reference.npz --calibrate-labels data/train_labels.csv --output-dir outputs/dgp/train
python DGP_main.py --patch-dir outputs/patches/test --reference outputs/reference/glycemic_basin_reference.npz --calibration outputs/dgp/train/calibration.json --output-dir outputs/dgp/test
```

The label CSV needs unique `sample_id` (or `id`) and `Type` values `NGT`, `PreDM`,
`DM`, covering every calibration recording. Reference fitting uses all designated
training patches to standardize embeddings and fit PCA (10 dimensions by default),
then uses NGT patches for the healthy center, covariance, and 90th percentile
basin threshold. The high-glycemia direction is defined by raw glucose risk weights
in this training cohort, preserving the local reference formula. It requires raw
patches in mmol/L. Reference fitting materializes valid training embeddings and
standardized embeddings; allow approximately two full valid-embedding matrices
plus PCA/covariance working memory.

The NPZ contains `mean`, `sd`, `components`, `mu`, `inv_cov`, `threshold`, `u_high`,
`high_mean`, `high_sd`, and `high_q95`. A compatible existing local
`glycemic_basin_reference.npz` may also be supplied. Reuse the same encoder,
reference and calibration for evaluation; do not refit using test labels.
The calibration records the reference hash, metric version, encoder hash when
available, NGT means/SDs and robust SLI scales, PLE centering, and training DRS
scales. Legacy array exports lack encoder hashes, so their provenance must be
verified separately. Newly exported incompatible encoder hashes are rejected.

The current numerical kernels assume patch intervals of 15 minutes and the first
patch endpoint at 12:00. The exporter assumes midnight sample origin by default;
`--start-minute` records a different origin. DGP rejects incompatible timing.
Legacy metadata with only patch size and stride uses the documented local
5-minute, midnight-origin convention. It does not infer recording clock time from
sample IDs. Nighttime is 00:00–06:00. Transitions and pulse horizons require
consecutive valid patch indices; missing patches are never compressed into
adjacent transitions.

## Numerical definitions

Let `PC_t = ((embedding_t - mean) / sd) @ components`,
`V_t = (PC_t - mu)^T inv_cov (PC_t - mu)`,
`s_t = (PC_t - mu)^T u_high`, and
`a_t = (s_t - high_mean) / high_sd`.

| Index | Current implementation |
| --- | --- |
| TOB | `1 - mean(V_t <= threshold)`; retains HBO_v1 internally |
| HAD | `mean(a_t)` |
| DynVar | NGT mean/SD standardized `sd(a_t, ddof=1)` |
| DynHBGI | NGT mean/SD standardized `mean(max(a_t,0)^2) + 1.5*mean(s_t > high_q95)` |
| PLE | 75th percentile pulse radial one-hour log ratio, minus fixed calibration NGT median |
| SLI | `robust_z(q90(ell)) + robust_z(mean(ell>0)) + 1.5*robust_z(q90(speed))`, each component clipped to `[-8,8]` |

For SLI, `r_t = sqrt(V_t)`, `ell = 4*log((r_next+1e-8)/(r_t+1e-8))`
per hour, and `speed = norm(PC_next-PC_t)/(r_t+1e-8)`. Robust component
normalization uses NGT median and `1.4826*MAD`, with IQR/SD fallbacks. The signed
SLI scale is preserved.

PLE uses Euclidean PCA displacement from the individual's nighttime median
attractor (whole-recording median if fewer than 12 nighttime patches). Sparse
outward jumps must exceed the original radial and flow thresholds. Peaks are
selected within eight patches, separated by eight patches, and require an intact
eight-patch follow-up. The scored radial ratio uses four patches (one hour):
`log((r_peak+4 + 1e-8)/(r_peak + 1e-8)) / 1 hour`. PLE requires at least 80
valid patches and an eligible event; no-event estimates remain NaN. It is a
finite-time radial response proxy, not an asymptotic maximal Lyapunov exponent.
The original Nicheng NGT centering constant was approximately
`-0.256838948 h^-1`; new training calibration computes its own reference median.

DRS is `z(HAD) + 0.5*max(z(PLE),0)*max(z(SLI),0)`. The default `z` uses the
current export cohort's mean and population SD (`ddof=0`), matching the local
current release. This score depends on cohort composition. Use
`--drs-reference training` for frozen training normalization and compare only
scores on the same recorded scale. Zero/undefined cohort SD yields NaN. Raw
components, valid patch counts, pulse event counts and completeness are retained
in `dgp_metrics.csv` alongside the six indices and DRS.

## Provenance and checks

[TRAJECTORY_DGP_SOURCES.json](TRAJECTORY_DGP_SOURCES.json) records the source
paths and hashes of copied numerical functions. The adapters add argument
parsing, portable I/O and validation; they preserve the current formulas.

```bash
python Others/verify_trajectory_dgp_sources.py
python Others/verify_trajectory_dgp_sources.py --original-root /path/to/CGMfoundation_support
python Others/verify_local_dgp_parity.py --original-root /path/to/CGMfoundation_support
python Others/smoke_trajectory_dgp.py
python ExportTrajectory.py --help
python DGP_reference.py --help
python DGP_main.py --help
```

The smoke check uses synthetic CGM, a random CPU checkpoint and artificial pulse
trajectories in a temporary directory. It checks encoder parity, row/mask alignment,
NaN preservation, overwrite refusal, reference fitting, calibration/application,
both DRS normalization modes, missing transitions, and short traces.
See [the validation record](TRAJECTORY_DGP_VALIDATION.md) for the separate local
846-recording parity check. Numerical parity verifies synchronization of this
release; it does not establish clinical validity or unbiased performance.
