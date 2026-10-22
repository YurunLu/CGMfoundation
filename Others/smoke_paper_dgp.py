"""Check manuscript DRS, TPLE gap handling, and checkpoint rejection."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DGP.paper import assemble, validate_calibration
from DGP.paper_kernels import pulse_neighbor_le, night_transition_blocks, lag_one_terms
from Others.verify_paper_dgp_artifacts import verify


def main():
    artifact = ROOT / 'artifacts/dgp/paper_20260826'
    verify(artifact)
    calibration = json.loads((artifact / 'calibration.json').read_text())
    manifest = json.loads((artifact / 'reference_manifest.json').read_text())
    # Known values distinguish the positive-part interaction from a sum.
    center, scale = calibration['stats']['HAD']
    raw = pd.DataFrame(dict(
        HAD=[center + 2 * scale] * 6, TOB=[0.0] * 6,
        dynvar_raw=[0.0] * 6, dynhbgi_raw=[0.0] * 6,
        TPLE=[2.0, -2.0, 0.0, np.nan, 2.0, -2.0],
        SLI=[3.0, 3.0, 3.0, 3.0, np.nan, np.nan]))
    scores = assemble(raw, calibration)
    np.testing.assert_allclose(scores.DRS, [3.5, 2.0, 2.0, np.nan, np.nan, np.nan],
                               rtol=1e-12, atol=1e-12, equal_nan=True)
    subset = assemble(raw.iloc[[0, 1]], calibration)
    np.testing.assert_allclose(subset.DRS, scores.DRS.iloc[[0, 1]])
    # One pair with a known exponential separation defines the one-hour slope.
    index = np.r_[np.arange(5), np.arange(20, 25)]
    pc = np.zeros((10, 5))
    pc[5:, 0] = np.exp(0.4 * np.arange(5) * 0.25)
    result = pulse_neighbor_le(index, pc, [0, 5], 5, 4, 96, np.eye(5))
    assert result['pairs'] == 1
    np.testing.assert_allclose(result['slope'], 0.4, rtol=1e-7, atol=1e-7)
    # Removing one follow-up state must not turn the gap into an adjacent step.
    keep = np.arange(10) != 7
    broken = pulse_neighbor_le(index[keep], pc[keep], [0, 5], 5, 4, 96, np.eye(5))
    assert broken['pairs'] == 0 and np.isnan(broken['slope'])
    starts, days = night_transition_blocks(np.arange(190))
    assert 71 in starts and 72 not in starts  # 05:45->06:00 is retained.
    rng = np.random.default_rng(20260826)
    radius = rng.normal(size=190)
    terms = lag_one_terms(radius, starts, days, 'r')
    shifted = radius.copy()
    for day in np.unique(days):
        selected = starts[days == day]
        shifted[np.unique(np.r_[selected, selected+1])] += 20 * (day + 1)
    after = lag_one_terms(shifted, starts, days, 'r')
    np.testing.assert_allclose(terms['r_ac1_nightfixed'], after['r_ac1_nightfixed'], atol=1e-12)
    short = lag_one_terms(radius, starts[:23], days[:23], 'r')
    assert np.isnan(short['r_ac1_nightfixed'])
    reference = artifact / 'glycemic_basin_reference.npz'
    valid_meta = dict(manifest['export_requirements'], checkpoint_sha256=calibration['checkpoint_sha256'])
    validate_calibration(calibration, reference, valid_meta)
    for stale in [dict(calibration, version=calibration['version'].replace('_v2', '_v1')),
                  dict(calibration, drs_formula='z_NGT(HAD) + 0.25 * [(1 h) * TPLE + SLI]')]:
        try:
            validate_calibration(stale, reference, valid_meta)
        except ValueError:
            pass
        else:
            raise AssertionError('Incompatible DRS calibration accepted')
    for bad_hash in [None, '0' * 64]:
        meta = dict(valid_meta, checkpoint_sha256=bad_hash)
        try:
            validate_calibration(calibration, reference, meta)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing or incorrect encoder identity accepted')
    for key, value in [('sample_column_start', 1), ('validity_policy', 'dataset')]:
        try:
            validate_calibration(calibration, reference, dict(valid_meta, **{key:value}))
        except ValueError:
            pass
        else:
            raise AssertionError(f'Incorrect manuscript export setting accepted: {key}')
    with tempfile.TemporaryDirectory(prefix='paper_dgp_') as folder:
        directory = Path(folder)
        (directory / 'meta.json').write_text(json.dumps(dict(valid_meta, checkpoint_sha256='0'*64)))
        output = directory / 'output'
        process = subprocess.run([sys.executable, str(ROOT/'DGP_paper.py'), '--patch-dir', str(directory),
                                  '--reference', str(reference), '--calibration', str(artifact/'calibration.json'),
                                  '--output-dir', str(output)], capture_output=True, text=True)
        assert process.returncode != 0 and 'checkpoint SHA-256' in process.stderr
        assert not output.exists()
    print('PASS: positive-part DRS interaction, missing scores, fixed calibration, one-pair TPLE, gap handling, nighttime rules, and incompatible-input rejection')


if __name__ == '__main__':
    main()
