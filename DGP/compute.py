"""Calculate radial PLE DGP indices with a frozen reference and calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from DGP.metric_kernels import (
    individual_attractor_pulse_events, finite_time_propagator_terms,
    mahalanobis_radius_terms, robust_location_scale, z_from_stats,
)
from ExportTrajectory import sha256

VERSION = 'current_ngt_centered_radial_ple_restored_sli_v1'
SLI_COLS = ['sli_maha_ell_q90_raw', 'sli_maha_ell_pos_frac_raw', 'sli_maha_speed_q90_raw']
METRICS = ['TOB', 'HAD', 'DynVar', 'DynHBGI', 'PLE', 'SLI']


def load_inputs(directory, reference_path):
    meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
    # Existing exports recorded patch_size/stride only; their documented clock
    # is 5-minute CGM, sample 1 at midnight, patch endpoint at noon.
    interval = meta.get('patch_interval_minutes', meta['stride'] * 5.0)
    first = meta.get('first_patch_minute', meta['patch_size'] * 5.0)
    if not np.isclose(interval, 15) or not np.isclose(first % 1440, 720):
        raise ValueError('Current DGP kernels require 15-minute patch spacing and first patch endpoint at noon')
    embeddings = np.load(directory / 'patch_embeddings.npy', mmap_mode='r')
    valid = np.load(directory / 'patch_valid_mask.npy', mmap_mode='r')
    ids = pd.read_csv(directory / 'sample_ids.csv', dtype=str).iloc[:, 0]
    if embeddings.ndim != 3 or valid.shape != embeddings.shape[:2] or len(ids) != len(embeddings):
        raise ValueError('Expected embeddings [sample,patch,dim], validity [sample,patch], and aligned IDs')
    if valid.dtype != np.bool_ or ids.isna().any() or ids.duplicated().any():
        raise ValueError('Require a boolean patch validity mask and unique recording IDs')
    with np.load(reference_path, allow_pickle=False) as archive:
        keys = ['mean', 'sd', 'components', 'mu', 'inv_cov', 'threshold', 'u_high', 'high_mean', 'high_sd', 'high_q95']
        ref = {key: archive[key] for key in keys}
    provenance_path = reference_path.parent / 'reference_manifest.json'
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
        if provenance['reference_sha256'] != sha256(reference_path):
            raise ValueError('Reference file differs from its recorded provenance')
        if provenance['trajectory_meta'].get('checkpoint_sha256') != meta.get('checkpoint_sha256'):
            raise ValueError('Reference and trajectory encoder hashes differ')
    dim, pcs = ref['components'].shape
    if dim != embeddings.shape[2] or ref['mean'].shape != (dim,) or ref['sd'].shape != (dim,):
        raise ValueError('Reference dimensions do not match embeddings')
    if ref['mu'].shape != (pcs,) or ref['inv_cov'].shape != (pcs, pcs) or ref['u_high'].shape != (pcs,):
        raise ValueError('Reference PCA dimensions are inconsistent')
    if any(not np.isfinite(value).all() for value in ref.values()) or np.any(ref['sd'] <= 0) or ref['high_sd'] <= 0:
        raise ValueError('Reference must be finite with positive scales')
    return embeddings, valid, ids, ref, meta


def raw_metrics(directory, reference_path):
    embeddings, valid, ids, ref, meta = load_inputs(directory, reference_path)
    rows = []
    # Match compute_ple_sli_v3's float32 affine projection for radial PLE.
    matrix = (ref['components'].astype(float) / ref['sd'].astype(float)[:, None]).astype(np.float32)
    offset = (-(ref['mean'].astype(float) @ (ref['components'].astype(float) / ref['sd'].astype(float)[:, None]))).astype(np.float32)
    for i, sid in enumerate(ids):
        index = np.flatnonzero(valid[i])
        row = dict(sample_index=i, sample_id=sid, valid_patch_count=len(index), ple_individual_pulse_event_count=0)
        row.update({key: np.nan for key in ['HBO_v1', 'HAD', 'patch_high_axis_sd', 'l_hbgi_tail_raw', 'PLE_raw', *SLI_COLS]})
        if len(index) >= 10:
            states = np.asarray(embeddings[i, index], dtype=np.float32)
            if not np.isfinite(states).all():
                raise ValueError(f'Nonfinite valid embeddings for sample index {i}')
            pc = ((states - ref['mean']) / ref['sd']) @ ref['components']
            displacement = pc - ref['mu']
            potential = np.sum(displacement @ ref['inv_cov'] * displacement, axis=1)
            high_projection = displacement @ ref['u_high']
            high = (high_projection - ref['high_mean']) / ref['high_sd']
            row.update(HBO_v1=float(np.mean(potential <= ref['threshold'])), HAD=float(np.mean(high)),
                       patch_high_axis_sd=float(np.std(high, ddof=1)),
                       l_hbgi_tail_raw=float(np.mean(np.maximum(high, 0)**2) + 1.5 * np.mean(high_projection > ref['high_q95'])))
            adjacent = np.flatnonzero(np.diff(index) == 1)
            # Match optimize_sli_mahalanobis.to_pc's float32 PCA then float64.
            sli_pc = pc.astype(np.float64)
            row.update(mahalanobis_radius_terms(sli_pc[adjacent], sli_pc[adjacent + 1], ref))
            if len(index) >= 80:
                ple_pc = states @ matrix + offset
                centered, events = individual_attractor_pulse_events(index, ple_pc)
                terms = finite_time_propagator_terms(centered, events, [], 4, 'ple')
                row['ple_individual_pulse_event_count'] = len(events)
                row['PLE_raw'] = terms['ple_radial_q75_per_hour']
        rows.append(row)
    return pd.DataFrame(rows), meta


def calibrate(raw, labels, reference_path, meta):
    table = pd.read_csv(labels, dtype={'sample_id': str, 'id': str})
    if 'sample_id' not in table:
        table = table.rename(columns={'id': 'sample_id'})
    if 'Type' not in table or 'sample_id' not in table or table['sample_id'].duplicated().any():
        raise ValueError('Labels need unique sample_id (or id) and Type columns')
    joined = raw.merge(table[['sample_id', 'Type']], on='sample_id', how='left', validate='one_to_one')
    if joined['Type'].isna().any() or not joined['Type'].isin(['NGT', 'PreDM', 'DM']).all():
        raise ValueError('Every calibration recording needs Type NGT, PreDM, or DM')
    ngt = joined.loc[joined.Type.eq('NGT')]
    if len(ngt) < 2:
        raise ValueError('At least two NGT calibration recordings are required')
    stats = {}
    for col in ['patch_high_axis_sd', 'l_hbgi_tail_raw']:
        values = ngt[col].dropna()
        if len(values) < 2:
            raise ValueError(f'Insufficient NGT values for {col}')
        sd = float(values.std(ddof=1))
        stats[col] = [float(values.mean()), sd if np.isfinite(sd) and sd >= 1e-8 else 1.0]
    for col in SLI_COLS:
        if ngt[col].notna().sum() < 2:
            raise ValueError(f'Insufficient NGT values for {col}')
        stats[col] = list(robust_location_scale(ngt[col]))
    ple_center = float(ngt.PLE_raw.median())
    if not np.isfinite(ple_center):
        raise ValueError('No eligible NGT pulse events for PLE centering')
    return dict(version=VERSION, reference_sha256=sha256(reference_path), stats=stats,
                ple_healthy_reference=ple_center, checkpoint_sha256=meta.get('checkpoint_sha256'))


def assemble(raw, calibration, drs_reference='dataset'):
    frame = raw.copy()
    stats = calibration['stats']
    frame['TOB'] = 1 - frame.HBO_v1
    for metric, col in [('DynVar', 'patch_high_axis_sd'), ('DynHBGI', 'l_hbgi_tail_raw')]:
        center, scale = stats[col]
        frame[metric] = (frame[col] - center) / scale
    frame['PLE'] = frame.PLE_raw - calibration['ple_healthy_reference']
    frame['SLI'] = sum(weight * z_from_stats(frame[col], *stats[col]) for weight, col in zip([1, 1, 1.5], SLI_COLS))
    if drs_reference == 'dataset':
        z = {}
        for col in ['HAD', 'PLE', 'SLI']:
            values = frame[col].replace([np.inf, -np.inf], np.nan)
            sd = values.std(ddof=0)
            z[col] = (values - values.mean()) / sd if np.isfinite(sd) and sd > 0 else values * np.nan
    else:
        z = {col: (frame[col] - center) / scale for col, (center, scale) in calibration['drs_stats'].items()}
    frame['DRS'] = z['HAD'] + 0.5 * z['PLE'].clip(lower=0) * z['SLI'].clip(lower=0)
    frame['PLE_healthy_reference_per_hour'] = calibration['ple_healthy_reference']
    frame['definition_version'] = VERSION
    frame['DRS_z_reference'] = drs_reference
    frame['complete_six_dgp'] = frame[METRICS].notna().all(axis=1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-dir', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path, help='Frozen glycemic_basin_reference.npz')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--calibration', type=Path, help='Previously frozen calibration JSON')
    choice.add_argument('--calibrate-labels', type=Path, help='Training/reference recording labels; creates calibration JSON')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--drs-reference', choices=['dataset', 'training'], default='dataset')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {args.output_dir}')
    raw, meta = raw_metrics(args.patch_dir, args.reference)
    if args.calibrate_labels:
        calibration = calibrate(raw, args.calibrate_labels, args.reference, meta)
        training = assemble(raw, calibration)
        calibration['drs_stats'] = {}
        for col in ['HAD', 'PLE', 'SLI']:
            sd = float(training[col].std(ddof=0))
            if not np.isfinite(sd) or sd <= 0:
                raise ValueError(f'Cannot fit training DRS scale for {col}')
            calibration['drs_stats'][col] = [float(training[col].mean()), sd]
    else:
        calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    if calibration['version'] != VERSION or calibration['reference_sha256'] != sha256(args.reference):
        raise ValueError('Calibration definition/reference does not match this run')
    if calibration.get('checkpoint_sha256') != meta.get('checkpoint_sha256'):
        raise ValueError('Calibration and trajectory checkpoint hashes differ')
    for center, scale in calibration['stats'].values():
        if not np.isfinite([center, scale]).all() or scale <= 0:
            raise ValueError('Calibration scales must be finite and positive')
    if args.drs_reference == 'training' and 'drs_stats' not in calibration:
        raise ValueError('Training DRS normalization requires drs_stats in calibration')
    if args.drs_reference == 'training':
        if set(calibration['drs_stats']) != {'HAD', 'PLE', 'SLI'}:
            raise ValueError('Training DRS normalization needs HAD, PLE, SLI scales')
        for center, scale in calibration['drs_stats'].values():
            if not np.isfinite([center, scale]).all() or scale <= 0:
                raise ValueError('Training DRS scales must be finite and positive')
    output = assemble(raw, calibration, args.drs_reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / 'dgp_metrics.csv', index=False)
    (args.output_dir / 'calibration.json').write_text(json.dumps(calibration, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    summary = dict(version=VERSION, sample_count=len(output), complete_six_dgp=int(output.complete_six_dgp.sum()),
                   nonmissing_drs=int(output.DRS.notna().sum()), drs_z_reference=args.drs_reference,
                   reference_sha256=sha256(args.reference), trajectory_meta=meta)
    (args.output_dir / 'run_manifest.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(output)} recordings; {summary["complete_six_dgp"]} complete six-index rows')


if __name__ == '__main__':
    main()
