"""Fit the local glycemic-basin reference formula on a designated training cohort."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from DGP.metric_kernels import patch_risk
from ExportTrajectory import sha256


def fit(directory, labels, output, dimensions=10):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {output}')
    embeddings = np.load(directory / 'patch_embeddings.npy', mmap_mode='r')
    raw = np.load(directory / 'patch_raw.npy', mmap_mode='r')
    valid = np.load(directory / 'patch_valid_mask.npy', mmap_mode='r')
    ids = pd.read_csv(directory / 'sample_ids.csv', dtype=str).iloc[:, 0]
    if ids.isna().any() or ids.duplicated().any() or ids.empty:
        raise ValueError('Training recording IDs must be nonempty and unique')
    meta = json.loads((directory / 'meta.json').read_text(encoding='utf-8'))
    interval = meta.get('patch_interval_minutes', meta['stride'] * 5.0)
    first = meta.get('first_patch_minute', meta['patch_size'] * 5.0)
    if not np.isclose(interval, 15) or not np.isclose(first % 1440, 720):
        raise ValueError('Current reference requires 15-minute patch spacing and first endpoint at noon')
    table = pd.read_csv(labels, dtype={'sample_id': str, 'id': str})
    if 'sample_id' not in table:
        table = table.rename(columns={'id': 'sample_id'})
    if 'sample_id' not in table or 'Type' not in table or table.sample_id.duplicated().any():
        raise ValueError('Labels need unique sample_id (or id) and Type columns')
    types = ids.map(table.set_index('sample_id').Type)
    if types.isna().any() or not types.isin(['NGT', 'PreDM', 'DM']).all():
        raise ValueError('Every training recording needs NGT, PreDM, or DM labels')
    if embeddings.ndim != 3 or valid.shape != embeddings.shape[:2] or raw.shape[:2] != valid.shape or len(ids) != len(embeddings):
        raise ValueError('Trajectory array shapes and IDs do not align')
    if valid.dtype != np.bool_ or not 2 <= dimensions <= embeddings.shape[2]:
        raise ValueError('Require boolean validity and 2 <= dimensions <= embedding dimension')
    sample, patch = np.where(valid)
    x = np.asarray(embeddings[sample, patch], dtype=np.float32)
    ngt = types.to_numpy()[sample] == 'NGT'
    if ngt.sum() < dimensions + 1 or not np.isfinite(x).all():
        raise ValueError('Insufficient NGT patches or nonfinite valid embeddings')
    mean, sd = x.mean(0), x.std(0)
    sd[sd < 1e-8] = 1
    xs = (x - mean) / sd
    covariance = xs.T @ xs / max(len(xs) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    components = vectors[:, np.argsort(values)[::-1][:dimensions]]
    pc = xs @ components
    mu = pc[ngt].mean(0)
    inv_cov = np.linalg.inv(np.cov(pc[ngt], rowvar=False) + np.eye(dimensions) * 1e-5)
    diff = pc[ngt] - mu
    threshold = np.percentile(np.sum(diff @ inv_cov * diff, axis=1), 90)
    weights = []
    for begin in range(0, len(ids), 32):
        end = min(begin + 32, len(ids))
        mean_g, hbgi, _, tar, _ = patch_risk(np.asarray(raw[begin:end]))
        weight = hbgi + 12.0 * tar + np.maximum(mean_g - 7.8, 0.0)**2
        weights.append(weight[valid[begin:end]])
    weights = np.concatenate(weights)
    gate = weights >= np.quantile(weights, 0.90)
    mu_high = np.average(pc[gate], axis=0, weights=np.maximum(weights[gate], 1e-6))
    u_high = mu_high - mu
    norm = np.linalg.norm(u_high)
    if norm <= 1e-8:
        raise ValueError('Training data do not define a high-glycemia direction')
    u_high /= norm
    s_high = (pc - mu) @ u_high
    high_sd = float(np.std(s_high[ngt], ddof=1))
    if not np.isfinite(high_sd) or high_sd <= 0:
        raise ValueError('Training high-axis scale is degenerate')
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / 'glycemic_basin_reference.npz', mean=mean, sd=sd, components=components,
             mu=mu, inv_cov=inv_cov, threshold=threshold, u_high=u_high,
             high_mean=float(np.mean(s_high[ngt])), high_sd=high_sd,
             high_q95=float(np.quantile(s_high[ngt], 0.95)))
    manifest = dict(formula_source='build_dgp_results_section.fit_reference + optimize_glycemic_basin_compact_dgp.fit_glycemic_reference',
                    dimensions=dimensions, training_recordings=len(ids), valid_training_patches=len(x),
                    labels_sha256=sha256(labels), trajectory_meta=meta,
                    reference_sha256=sha256(output / 'glycemic_basin_reference.npz'))
    (output / 'reference_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Fitted reference from {len(ids)} training recordings and {len(x)} valid patches')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patch-dir', required=True, type=Path)
    parser.add_argument('--labels', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--dimensions', default=10, type=int)
    args = parser.parse_args()
    fit(args.patch_dir, args.labels, args.output_dir, args.dimensions)
