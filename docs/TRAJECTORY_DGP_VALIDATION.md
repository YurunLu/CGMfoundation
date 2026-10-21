# Trajectory and DGP reproducibility checks

Run the following checks from the repository root after installing the
dependencies. They cover frozen-parameter integrity, trajectory export, and
calculation of the manuscript DGP indices.

## Frozen manuscript parameters

```bash
python Others/verify_paper_dgp_artifacts.py
python Others/verify_paper_dgp_artifacts.py --checkpoint-path checkpoints/best_forecast_20260420_105114_dim256_patch144_stride3.pt --patch-dir outputs/patches/cohort
```

The verifier checks artifact hashes, reference dimensions and scales, calibration
definitions, and numerical-function checksums. With the optional arguments, it
also verifies the checkpoint file and trajectory metadata. The
[artifact manifest](../artifacts/dgp/paper_20260826/reference_manifest.json)
records the required checkpoint identity and export settings.

## Synthetic execution

```bash
python Others/smoke_trajectory_dgp.py
python Others/smoke_paper_dgp.py
```

The trajectory check uses synthetic CGM and a small random checkpoint. It checks
embedding export against direct model inference, sample-column origin, both
validity policies, observation masks, string IDs, missing-value retention,
reference fitting, calibration, and output protection.

The manuscript DGP check exercises a matched pulse-response pair with a known
growth rate, missing follow-up handling, nighttime transition boundaries,
within-night centering, and insufficient-data rules. It also checks rejection of
an incorrect checkpoint hash, sample-column origin, or validity policy before
output files are created.

## Numerical-function integrity

```bash
python Others/verify_trajectory_dgp_sources.py
```

Numerical functions are identified by module, function name, and SHA-256 in the
corresponding checksum manifests. The manuscript artifact verifier checks the
functions used by `DGP_paper.py`.

## Manuscript calculation checks

The [validation summary](../artifacts/dgp/paper_20260826/validation.json) covers
the six DGP indices for 821 three-group Nicheng recordings and TPLE, SLI, and DRS
for 208 Colas recordings. It also records public-command execution and reuse of
the fixed calibration on a recording subset. Definitions and input requirements
are in the [DGP guide](PAPER_DGP.md).

Software and numerical checks do not establish clinical utility or prospective
predictive performance. Evaluate scientific claims using the study protocol and
an appropriate independent cohort.
