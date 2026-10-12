"""CPU synthetic checks; no participant data or pretrained weights needed."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.base_model import CGMBaseModel
from DGP.compute import raw_metrics, assemble, calibrate
from utils.data_utils import CGMForecastDataset


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def check(directory):
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    length = 1441
    config = {'model': dict(patch_size=144, stride=3, forecast_steps=12, embed_dim=8,
                           max_seq_len=length, depth=1, num_heads=2, hidden_dim=16,
                           dropout=0.0, use_cnn=True, use_transformer=True,
                           use_temporal=False, use_positional=True)}
    model_args = dict(config['model'])
    stride = model_args.pop('stride')
    model_args['forecast_size'] = model_args.pop('forecast_steps')
    model = CGMBaseModel(**model_args).eval()
    torch.save({'model_state_dict': model.state_dict()}, directory / 'checkpoint.pt')
    (directory / 'config.json').write_text(json.dumps(config))
    data = rng.normal(6, 1, (6, length - 1)).astype(np.float32)
    data[0, 250:260] = np.nan
    data[1, 1100:] = np.nan
    data[-1] = np.nan
    ids = ['001', '002', '003', '004', '005', 'empty']
    frame = pd.DataFrame(data, columns=map(str, range(1, length)))
    frame.insert(0, 'id', ids)
    frame.to_csv(directory / 'cgm.csv', index=False)
    export = directory / 'patches'
    command = [ROOT / 'ExportTrajectory.py', '--csv-path', directory / 'cgm.csv',
               '--checkpoint-path', directory / 'checkpoint.pt', '--config', directory / 'config.json',
               '--output-dir', export, '--batch-size', '2']
    run(*command)
    z = np.load(export / 'patch_embeddings.npy')
    mask = np.load(export / 'patch_valid_mask.npy')
    raw = np.load(export / 'patch_raw.npy')
    assert pd.read_csv(export / 'sample_ids.csv', dtype=str).sample_id.tolist() == ids[:-1]
    dataset = CGMForecastDataset(data[:-1], patch_size=144, stride=stride, forecast_steps=12)
    with torch.inference_mode():
        direct = model(torch.stack([dataset[i][0] for i in range(5)]), stride=stride, return_embedding=True).numpy()
    np.testing.assert_allclose(z, direct, rtol=1e-5, atol=1e-5)
    assert np.array_equal(mask, np.stack([dataset[i][1].numpy() for i in range(5)]))
    assert np.isnan(raw[0]).any() and not mask[1, -1]
    duplicate = subprocess.run([sys.executable, *map(str, command)], cwd=ROOT, capture_output=True)
    assert duplicate.returncode != 0 and b'FileExistsError' in duplicate.stderr

    # Use known pulsatile trajectories to exercise PLE, gaps and calibration.
    patch_count = z.shape[1]
    time = np.arange(patch_count)
    fixture = rng.normal(0, 0.12, z.shape).astype(np.float32)
    for sample in range(5):
        amplitude = np.zeros(patch_count)
        for pulse in range(20, patch_count - 20, 35):
            amplitude[pulse:pulse + 20] += (2 + sample * 0.25) * np.exp(-np.arange(20) / 6)
        fixture[sample, :, 0] += amplitude
    np.save(export / 'patch_embeddings.npy', fixture)
    reference = directory / 'reference.npz'
    np.savez(reference, mean=np.zeros(8, dtype=np.float32), sd=np.ones(8, dtype=np.float32),
             components=np.eye(8, dtype=np.float32), mu=np.zeros(8), inv_cov=np.eye(8),
             threshold=3., u_high=np.eye(8)[0], high_mean=0., high_sd=1., high_q95=1.5)
    labels = directory / 'labels.csv'
    pd.DataFrame({'sample_id': ids[:-1], 'Type': ['NGT', 'NGT', 'NGT', 'PreDM', 'DM']}).to_csv(labels, index=False)
    run(ROOT / 'DGP_reference.py', '--patch-dir', export, '--labels', labels,
        '--output-dir', directory / 'fitted_reference', '--dimensions', '6')
    run(ROOT / 'DGP_main.py', '--patch-dir', export,
        '--reference', directory / 'fitted_reference' / 'glycemic_basin_reference.npz',
        '--calibrate-labels', labels, '--output-dir', directory / 'fitted_reference_dgp')
    raw_dgp, meta = raw_metrics(export, reference)
    cal = calibrate(raw_dgp, labels, reference, meta)
    result = assemble(raw_dgp, cal)
    assert result.complete_six_dgp.all() and result.DRS.notna().all()
    assert result.ple_individual_pulse_event_count.min() > 0
    fit_dir = directory / 'calibrated'
    run(ROOT / 'DGP_main.py', '--patch-dir', export, '--reference', reference,
        '--calibrate-labels', labels, '--output-dir', fit_dir)
    apply_dir = directory / 'applied'
    run(ROOT / 'DGP_main.py', '--patch-dir', export, '--reference', reference,
        '--calibration', fit_dir / 'calibration.json', '--output-dir', apply_dir)
    pd.testing.assert_frame_equal(pd.read_csv(fit_dir / 'dgp_metrics.csv'), pd.read_csv(apply_dir / 'dgp_metrics.csv'))
    run(ROOT / 'DGP_main.py', '--patch-dir', export, '--reference', reference,
        '--calibration', fit_dir / 'calibration.json', '--output-dir', directory / 'training_drs', '--drs-reference', 'training')
    # There is no valid transition across omitted patches, even if endpoints exist.
    gapped = mask.copy()
    gapped[:] = False
    gapped[:, ::2] = True
    np.save(export / 'patch_valid_mask.npy', gapped)
    gap_raw, _ = raw_metrics(export, reference)
    assert gap_raw.PLE_raw.isna().all() and gap_raw.sli_step_count.eq(0).all()
    assert gap_raw.sli_maha_ell_q90_raw.isna().all()
    # Short traces retain coverage but cannot invent pulse-response estimates.
    gapped[:] = False
    gapped[:, :5] = True
    np.save(export / 'patch_valid_mask.npy', gapped)
    short_raw, _ = raw_metrics(export, reference)
    assert short_raw.PLE_raw.isna().all() and short_raw.HAD.isna().all()
    print('PASS: encoder parity, masks, IDs, raw NaNs, overwrite refusal, DGP calibration/application, DRS modes, gaps')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path)
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=False)
        check(args.work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix='cgm_trajectory_dgp_') as path:
            check(Path(path))
