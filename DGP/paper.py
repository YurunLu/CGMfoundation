"""DGP and DRS definitions used in the 2026-08-26 manuscript result package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DGP.compute import load_inputs
from DGP.paper_kernels import detect_pulse_peaks, pulse_neighbor_le, night_transition_blocks, lag_one_terms, robust_location_scale
from ExportTrajectory import sha256

VERSION = 'paper_20260826_trajectory_paired_tple_nightfixed_sli_v1'
METRICS = ['TOB', 'HAD', 'DynVar', 'DynHBGI', 'TPLE', 'SLI']
DRS_FORMULA = 'z_NGT(HAD) + 0.25 * [(1 h) * TPLE + SLI]'


def raw_metrics(directory, reference_path):
    embeddings, valid, ids, ref, meta = load_inputs(directory, reference_path)
    # Preserve compute_ple_sli_v3's affine float32 healthy-coordinate projection.
    precision = np.asarray(ref['inv_cov'], dtype=np.float64)
    values, vectors = np.linalg.eigh(precision)
    whitener = vectors @ np.diag(np.sqrt(np.maximum(values, 1e-10)))
    mean, sd = np.asarray(ref['mean'], float), np.asarray(ref['sd'], float)
    components, mu = np.asarray(ref['components'], float), np.asarray(ref['mu'], float)
    matrix = ((components @ whitener) / sd[:, None]).astype(np.float32)
    offset = (-(mean @ ((components @ whitener) / sd[:, None])) - mu @ whitener).astype(np.float32)
    if components.shape[1] < 5:
        raise ValueError('Paper TPLE requires at least five reference PCA dimensions')
    rows = []
    for i, sample_id in enumerate(ids):
        index = np.flatnonzero(valid[i])
        row = dict(sample_index=i, sample_id=sample_id, valid_patch_count=len(index),
                   TPLE_pair_count=0, TPLE_peak_count=0, SLI_step_count=0)
        row.update({key: np.nan for key in ['TOB', 'HAD', 'dynvar_raw', 'dynhbgi_raw',
                                           'TPLE', 'SLI', 'SLI_raw', 'SLI_rho']})
        if len(index) >= 10:
            states = np.asarray(embeddings[i, index], dtype=np.float32)
            if not np.isfinite(states).all():
                raise ValueError(f'Nonfinite valid embeddings for recording index {i}')
            pc = ((states - ref['mean']) / ref['sd']) @ ref['components']
            displacement = pc - ref['mu']
            potential = np.sum(displacement @ ref['inv_cov'] * displacement, axis=1)
            projection = displacement @ ref['u_high']
            high = (projection - ref['high_mean']) / ref['high_sd']
            row.update(TOB=float(np.mean(potential > ref['threshold'])), HAD=float(np.mean(high)),
                       dynvar_raw=float(np.std(high, ddof=1)),
                       dynhbgi_raw=float(np.mean(np.maximum(high, 0)**2) + 1.5 * np.mean(projection > ref['high_q95'])))
            if len(index) >= 80:
                peaks = detect_pulse_peaks(index, pc)
                tple = pulse_neighbor_le(index, pc, peaks, dimensions=5, horizon=4,
                                         tolerance=96, inv_cov=ref['inv_cov'])
                row.update(TPLE=float(tple['slope']), TPLE_pair_count=int(tple['pairs']),
                           TPLE_peak_count=len(peaks))
                healthy = states @ matrix + offset
                radius = np.linalg.norm(healthy, axis=1)
                starts, days = night_transition_blocks(index)
                terms = lag_one_terms(radius, starts, days, 'sli_radius')
                rho = float(terms['sli_radius_ac1_nightfixed'])
                row.update(SLI_rho=rho, SLI_step_count=int(terms['sli_radius_step_count']))
                if np.isfinite(rho) and 0 < rho < 1:
                    row['SLI_raw'] = float(-1 / np.log(rho))
                    row['SLI'] = float(np.log1p(row['SLI_raw']))
        rows.append(row)
        if (i + 1) % 100 == 0:
            print(f'Computed {i + 1}/{len(ids)} recordings', flush=True)
    return pd.DataFrame(rows), meta


def validate_calibration(calibration, reference_path, meta):
    if calibration['version'] != VERSION:
        raise ValueError('Calibration is not for the paper_20260826 definition')
    if calibration['reference_sha256'] != sha256(reference_path):
        raise ValueError('Calibration reference hash differs from the supplied NPZ')
    expected = calibration.get('checkpoint_sha256')
    actual = meta.get('checkpoint_sha256')
    if not expected or not actual or expected != actual:
        raise ValueError('Paper DGP requires matching checkpoint SHA-256 in calibration and trajectory meta.json; re-export legacy arrays with the verified checkpoint')
    for key in ['dynvar_raw', 'dynhbgi_raw', 'HAD']:
        center, scale = calibration['stats'][key]
        if not np.isfinite([center, scale]).all() or scale <= 0:
            raise ValueError(f'Invalid calibration for {key}')
    if calibration.get('drs_formula') != DRS_FORMULA:
        raise ValueError('Paper DRS formula does not match the frozen definition')
    manifest_path = reference_path.parent / 'reference_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        for key, value in manifest.get('export_requirements', {}).items():
            if key == 'model_config':
                actual_config = meta.get(key, {})
                if any(actual_config.get(k) != v for k, v in value.items()):
                    raise ValueError('Trajectory model_config differs from the paper export requirements')
            elif meta.get(key) != value:
                raise ValueError(f'Trajectory {key} differs from the paper export requirements')


def assemble(raw, calibration):
    frame = raw.copy()
    for metric, source in [('DynVar', 'dynvar_raw'), ('DynHBGI', 'dynhbgi_raw')]:
        center, scale = calibration['stats'][source]
        frame[metric] = (frame[source] - center) / scale
    center, scale = calibration['stats']['HAD']
    frame['zNGT_HAD'] = (frame.HAD - center) / scale
    frame['DRS'] = frame.zNGT_HAD + 0.25 * (frame.TPLE + frame.SLI)
    frame['definition_version'] = VERSION
    frame['TPLE_unit'] = 'h^-1'
    frame['SLI_unit'] = 'log1p(15-minute steps)'
    frame['DRS_formula'] = DRS_FORMULA
    frame['DRS_z_reference'] = 'frozen_Nicheng_NGT' if calibration.get('release') == 'paper_20260826' else 'frozen_reference_NGT'
    frame['complete_six_dgp'] = frame[METRICS].notna().all(axis=1)
    return frame


def calibrate(raw, labels_path, reference_path, meta):
    table = pd.read_csv(labels_path, dtype={'sample_id': str, 'id': str})
    if 'sample_id' not in table:
        table = table.rename(columns={'id': 'sample_id'})
    if 'sample_id' not in table or 'Type' not in table or table.sample_id.duplicated().any():
        raise ValueError('Labels require unique sample_id (or id) and Type')
    joined = raw.merge(table[['sample_id', 'Type']], on='sample_id', how='left', validate='one_to_one')
    if not joined.Type.isin(['NGT', 'PreDM', 'DM']).all():
        raise ValueError('Every reference recording requires NGT, PreDM, or DM')
    healthy = joined.loc[joined.Type.eq('NGT')]
    stats = {}
    for column in ['dynvar_raw', 'dynhbgi_raw']:
        values = healthy[column].dropna()
        if len(values) < 2 or not np.isfinite(values.std(ddof=1)) or values.std(ddof=1) <= 0:
            raise ValueError(f'Insufficient nondegenerate NGT values for {column}')
        stats[column] = [float(values.mean()), float(values.std(ddof=1))]
    if healthy.HAD.notna().sum() < 2 or not meta.get('checkpoint_sha256'):
        raise ValueError('Calibration requires NGT HAD values and an encoder SHA-256')
    stats['HAD'] = list(robust_location_scale(healthy.HAD))
    return dict(version=VERSION, stats=stats, drs_formula=DRS_FORMULA,
                reference_sha256=sha256(reference_path), checkpoint_sha256=meta['checkpoint_sha256'],
                calibration_recordings=len(joined), ngt_recordings=len(healthy),
                labels_sha256=sha256(labels_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--calibration', type=Path)
    choice.add_argument('--calibrate-labels', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {args.output_dir}')
    meta = json.loads((args.patch_dir / 'meta.json').read_text(encoding='utf-8'))
    calibration = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
        validate_calibration(calibration, args.reference, meta)
    raw, meta = raw_metrics(args.patch_dir, args.reference)
    if calibration is None:
        calibration = calibrate(raw, args.calibrate_labels, args.reference, meta)
        validate_calibration(calibration, args.reference, meta)
    output = assemble(raw, calibration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / 'dgp_metrics.csv', index=False)
    (args.output_dir / 'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    manifest = dict(version=VERSION, sample_count=len(output),
                    complete_six_dgp=int(output.complete_six_dgp.sum()),
                    nonmissing_drs=int(output.DRS.notna().sum()), reference_sha256=sha256(args.reference),
                    checkpoint_sha256=meta['checkpoint_sha256'],
                    calibration_sha256=sha256(args.calibration) if args.calibration else sha256(args.output_dir / 'calibration.json'),
                    trajectory_meta=meta)
    (args.output_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(output)} paper_20260826 profiles; {manifest["complete_six_dgp"]} complete six-index rows')


if __name__ == '__main__':
    main()
