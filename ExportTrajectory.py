"""Export ordered patch embeddings using the existing forecast encoder."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-path', type=Path, required=True)
    parser.add_argument('--checkpoint-path', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='Training YAML or JSON; model section required')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--id-column', default='id')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--sample-minutes', type=float, default=5.0)
    parser.add_argument('--start-minute', type=float, default=0.0, help='Clock time of sample column 1, minutes after midnight')
    parser.add_argument('--min-observed-fraction', type=float, default=0.1)
    return parser.parse_args()


def export(args):
    import torch
    import yaml
    from models.base_model import CGMBaseModel
    from utils.data_utils import CGMForecastDataset

    if args.batch_size < 1 or args.sample_minutes <= 0 or not 0 <= args.min_observed_fraction < 1:
        raise ValueError('Invalid batch size, sampling interval, or observed fraction')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f'Use a new or empty output directory: {args.output_dir}')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    model_config = dict(config['model'])
    stride = int(model_config.pop('stride'))
    forecast = int(model_config.pop('forecast_steps'))
    if stride <= 0 or forecast <= 0:
        raise ValueError('stride and forecast_steps must be positive')
    model_config['forecast_size'] = forecast
    model = CGMBaseModel(**model_config)
    # Checkpoints should come from a trusted training run. Strict loading prevents
    # silently exporting trajectories with missing/random encoder parameters.
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=True)
    if isinstance(checkpoint, dict):
        # Flags and stride do not change parameter tensor shapes. Strict state
        # loading alone cannot detect a configuration that changes inference.
        effective = {name: parameter.default for name, parameter in inspect.signature(CGMBaseModel).parameters.items()
                     if parameter.default is not inspect.Parameter.empty}
        effective.update(model_config)
        effective.update(stride=stride, forecast_steps=forecast)
        for name, expected in checkpoint.get('model_hparams', {}).items():
            if expected is not None and name in effective and effective[name] != expected:
                raise ValueError(f'Checkpoint model_hparams mismatch for {name}: config={effective[name]!r}, checkpoint={expected!r}')
        for key in ('model_state_dict', 'state_dict', 'model'):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(args.device)

    frame = pd.read_csv(args.csv_path, dtype={args.id_column: str})
    if args.id_column not in frame:
        raise ValueError(f'Missing ID column: {args.id_column}')
    length = int(model_config['max_seq_len'])
    columns = [str(i) for i in range(1, length)]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Expected sample columns 1..{length - 1}; missing {len(missing)} columns')
    raw = frame[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if np.isinf(raw).any():
        raise ValueError('CGM values must be finite or NaN')
    keep = np.isfinite(raw).sum(axis=1) > length * args.min_observed_fraction
    ids = frame.loc[keep, args.id_column]
    if ids.isna().any() or ids.duplicated().any() or ids.empty:
        raise ValueError('Retained sample IDs must be nonempty and unique; use recording IDs for repeated visits')
    raw = raw[keep]
    dataset = CGMForecastDataset(raw, patch_size=model.patch_size, stride=stride, forecast_steps=forecast)
    if dataset.num_patches == 0:
        raise ValueError('Sequence is shorter than patch_size + forecast_steps')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n, p, d = len(ids), dataset.num_patches, model.embed_dim
    arrays = {
        'patch_embeddings': np.lib.format.open_memmap(args.output_dir / 'patch_embeddings.npy', mode='w+', dtype='float32', shape=(n, p, d)),
        'patch_raw': np.lib.format.open_memmap(args.output_dir / 'patch_raw.npy', mode='w+', dtype='float32', shape=(n, p, model.patch_size)),
        'patch_valid_mask': np.lib.format.open_memmap(args.output_dir / 'patch_valid_mask.npy', mode='w+', dtype='bool', shape=(n, p)),
        'patch_observed_mask': np.lib.format.open_memmap(args.output_dir / 'patch_observed_mask.npy', mode='w+', dtype='bool', shape=(n, p, model.patch_size)),
    }
    with torch.inference_mode():
        for begin in range(0, n, args.batch_size):
            end = min(begin + args.batch_size, n)
            items = [dataset[i] for i in range(begin, end)]
            patches = torch.stack([item[0] for item in items]).to(args.device)
            embeddings = model(patches, stride=stride, return_embedding=True).cpu().numpy()
            if not np.isfinite(embeddings).all():
                raise ValueError('Encoder produced nonfinite embeddings')
            arrays['patch_embeddings'][begin:end] = embeddings
            original = np.stack([raw[begin:end, start:start + model.patch_size] for start in dataset.starts], axis=1)
            arrays['patch_raw'][begin:end] = original
            arrays['patch_observed_mask'][begin:end] = np.isfinite(original)
            arrays['patch_valid_mask'][begin:end] = torch.stack([item[1] for item in items]).numpy()
            print(f'Exported {end}/{n}', flush=True)
    for array in arrays.values():
        array.flush()
    ids.rename('sample_id').to_csv(args.output_dir / 'sample_ids.csv', index=False)
    meta = {
        'format_version': 'cgm_patch_trajectory_v1',
        'csv_path': str(args.csv_path.resolve()), 'csv_sha256': sha256(args.csv_path),
        'checkpoint_path': str(args.checkpoint_path.resolve()), 'checkpoint_sha256': sha256(args.checkpoint_path),
        'config_sha256': sha256(args.config), 'model_config': model_config,
        'num_samples': n, 'num_patches': p, 'embed_dim': d,
        'patch_size': model.patch_size, 'stride': stride, 'forecast_steps': forecast,
        'sample_minutes': args.sample_minutes, 'start_minute': args.start_minute,
        'first_patch_minute': args.start_minute + model.patch_size * args.sample_minutes,
        'patch_interval_minutes': stride * args.sample_minutes,
        'units': 'mmol/L', 'mask_policy': 'existing CGMForecastDataset: patch ends before last observed sample',
        'imputation': 'existing fill_nan_lin: linear interpolation; edge extension',
        'raw_policy': 'original values before interpolation',
        'excluded_sample_count': int((~keep).sum()),
        'min_observed_fraction': args.min_observed_fraction,
    }
    (args.output_dir / 'meta.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    export(parse_args())
