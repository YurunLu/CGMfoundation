# Latent CGM trajectory export

`ExportTrajectory.py` extracts ordered patch embeddings from a trained
CGMfoundation encoder. For manuscript DGP analysis, use the frozen export
configuration and parameters described in the [DGP guide](PAPER_DGP.md).

Run commands from the repository root after installing `requirements.txt` and a
compatible PyTorch build.

## Export

Verify the manuscript checkpoint and export a cohort:

```bash
python Others/verify_paper_dgp_artifacts.py --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt
python ExportTrajectory.py --csv-path data/cohort.csv --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt --config artifacts/dgp/paper_20260826/export_config.json --output-dir outputs/patches/cohort --device cpu --batch-size 16
```

The CSV requires an `id` column, configurable with `--id-column`. For the
manuscript configuration, provide sample columns `0` through `3455`, in mmol/L at
five-minute intervals. Each row represents a recording. Repeated visits need
distinct recording IDs; string IDs, including leading zeroes, are preserved.

The default recording filter retains more than `max_seq_len * 0.1` observed
samples. Excluded recordings are counted in the metadata. The exporter preserves
recording order and writes the full sequence of patch embeddings.

## Checkpoint and configuration

YAML and JSON configurations are supported. The `model` section specifies patch
size, stride, forecast reservation, embedding dimension, sequence length,
Transformer depth and heads, and positional and temporal embedding settings.

The manuscript configuration uses 144-sample patches, stride 3, a 12-sample
forecast reservation, and a 256-dimensional encoder. It produces 1,101 patch
slots from 3,456 samples. The `export` section sets:

```json
{
  "sample_column_start": 0,
  "validity_policy": "any-observed"
}
```

For other applications, `--sample-column-start` selects zero- or one-based
sample columns. `--validity-policy` selects `any-observed` or `dataset` validity.
Explicit CLI options override the corresponding export configuration settings.
The chosen settings are recorded in `meta.json`.

Model parameters are loaded strictly. When checkpoint `model_hparams` are present,
the exporter also checks settings such as stride and embedding flags that do not
change tensor shapes. Checkpoint wrappers `model_state_dict`, `state_dict`, and
`model` are supported. The loader uses `weights_only=True` with a restricted set
of numerical scalar types for checkpoint training statistics.

## Output files

| File | Shape or contents |
| --- | --- |
| `patch_embeddings.npy` | float32 `[recording, patch, embedding_dimension]` |
| `patch_raw.npy` | float32 `[recording, patch, patch_size]`, original glucose values and NaNs |
| `patch_valid_mask.npy` | boolean `[recording, patch]` |
| `patch_observed_mask.npy` | boolean `[recording, patch, patch_size]`, original observation positions |
| `sample_ids.csv` | One `sample_id` per retained recording, in array row order |
| `meta.json` | Model and export settings, input/checkpoint/configuration hashes, timing, and filtering rules |

Arrays are written in batches using memory maps. Supply a new or empty output
directory. Metadata are written after the array export completes.

## Preprocessing and masks

Model inputs use linear interpolation across missing values and extend the
nearest available value at the sequence edges. Original values and observation
positions are exported separately from the imputed model inputs.

Under the manuscript `any-observed` policy, a patch is valid when it contains at
least one original observation. Such a patch may extend beyond the final
observation. Under `dataset`, validity requires the patch endpoint to precede or
coincide with the last observed sample. Internal missing values are imputed under
both policies.

Patch starts reserve `forecast_steps` samples after each history window. This
reservation is also applied when exporting embeddings. The preprocessing is
intended for offline analysis; interpolation can use later observations and span
internal gaps.

## Timing

The default sample origin is midnight. The manuscript uses a 15-minute patch
interval and a first patch endpoint at noon. `--sample-minutes` and
`--start-minute` record the sampling interval and clock origin. The DGP command
checks the timing against the frozen reference requirements. Transitions require
consecutive valid patch indices; missing patches are not compressed into adjacent
transitions.

## Calculate DGP

```bash
python DGP_paper.py --patch-dir outputs/patches/cohort --reference artifacts/dgp/paper_20260826/glycemic_basin_reference.npz --calibration artifacts/dgp/paper_20260826/calibration.json --output-dir outputs/dgp/cohort
```

The [DGP guide](PAPER_DGP.md) defines the six manuscript indices and DRS,
describes checkpoint verification, and explains reference fitting for a new
encoder. See [reproducibility checks](TRAJECTORY_DGP_VALIDATION.md) for the
automated checks available with the source release.
