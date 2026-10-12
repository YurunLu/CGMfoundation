# Trajectory and DGP validation

Checked on 2026-10-12 against upstream commit
`cee21194ff08410166b7a90f7dfcd4c295f907cc`.

Environment: Windows, Python 3.12.14, PyTorch 2.6.0+cpu, NumPy 2.3.5,
pandas 2.3.3, PyYAML 6.0.3. Dependencies were installed into an isolated local
validation directory; the repository's existing environment files are preserved.

## Source and execution checks

- All nine copied numerical functions match their recorded SHA256 hashes and the
  original local source functions.
- All three new entry-point `--help` commands passed.
- Synthetic export matched direct inference from the same random encoder.
- Exported masks matched `CGMForecastDataset`; string IDs and raw NaNs were retained.
- Empty recordings were excluded and existing nonempty outputs were rejected.
- Synthetic reference fitting, NGT calibration and frozen application passed.
- Both dataset and training DRS normalization ran successfully.
- Invalid transitions across missing patches were excluded; short traces retained
  missing estimates.
- The existing downstream source verifier passed for all 65 recorded files.

Reproduce the public checks:

```bash
python Others/verify_trajectory_dgp_sources.py
python Others/smoke_trajectory_dgp.py
python Others/verify_downstream_layout.py
```

## Read-only local numerical comparison

The new calculator was applied to all 846 local Nicheng patch trajectories using
the existing glycemic-basin reference, original NGT feature calibration and the
current local PLE centering. Recording IDs were joined one-to-one to
`ple_sli_current_results/tables/nicheng_current_dgp.csv`.
All six indices and DRS matched at `rtol=1e-5, atol=1e-5`, including missingness.

| Metric | Maximum absolute difference |
| --- | ---: |
| TOB | 2.22e-16 |
| HAD | 7.61e-08 |
| DynVar | 3.12e-07 |
| DynHBGI | 1.23e-06 |
| PLE | 1.73e-16 |
| SLI | 1.58e-14 |
| DRS | 2.67e-08 |

Each compared metric had 846 nonmissing values in both versions. The parity check
is available when the private local inputs are present:

```bash
python Others/verify_local_dgp_parity.py --original-root /path/to/CGMfoundation_support --report outputs/local_dgp_parity.json
```

Participant arrays, identifiers, labels, checkpoints and trained reference files
are excluded from this contribution. GPU inference and multi-GPU execution were
not tested. Numerical parity is scoped to the current Nicheng release and does
not demonstrate forecast accuracy, clinical utility or performance superiority.
New training references follow the same fitting formula but are not the original
Nicheng reference; PCA signs, calibration and scores depend on the designated
training cohort and encoder.
