"""Validate the frozen manuscript artifacts and optional local checkpoint/export."""
from __future__ import annotations
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from DGP.paper import VERSION, DRS_FORMULA, validate_calibration
from ExportTrajectory import sha256, load_checkpoint


def verify(directory, checkpoint=None, patch_dir=None, original_root=None):
    manifest = json.loads((directory / 'reference_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'verified' or manifest.get('release') != 'paper_20260826':
        raise ValueError('Artifact bundle is not a verified manuscript release')
    for name, digest in manifest['files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Manifest entries must name files in the artifact directory')
        if sha256(directory / name) != digest:
            raise ValueError(f'Published artifact checksum mismatch: {name}')
    reference = directory / 'glycemic_basin_reference.npz'
    if sha256(reference) != manifest['reference_sha256']:
        raise ValueError('Reference checksum mismatch')
    calibration = json.loads((directory / 'calibration.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'export_config.json').read_text(encoding='utf-8'))
    if calibration['version'] != VERSION or calibration['drs_formula'] != DRS_FORMULA:
        raise ValueError('Unexpected manuscript metric definition')
    if calibration['checkpoint_sha256'] != manifest['checkpoint']['sha256']:
        raise ValueError('Checkpoint identities differ between manifest and calibration')
    metadata = dict(manifest['export_requirements'], checkpoint_sha256=manifest['checkpoint']['sha256'])
    validate_calibration(calibration, reference, metadata)
    with np.load(reference, allow_pickle=False) as archive:
        if archive['components'].shape != (256, 10):
            raise ValueError('Unexpected manuscript PCA dimensions')
        for name in ['mean', 'sd', 'components', 'mu', 'inv_cov', 'u_high']:
            if not np.isfinite(archive[name]).all():
                raise ValueError(f'Nonfinite reference parameter: {name}')
        if np.any(archive['sd'] <= 0) or float(archive['high_sd']) <= 0:
            raise ValueError('Nonpositive reference scales')
        if np.min(np.linalg.eigvalsh(archive['inv_cov'])) <= 0:
            raise ValueError('Healthy precision must be positive definite')
    sources = json.loads((ROOT / 'docs/PAPER_DGP_SOURCES.json').read_text(encoding='utf-8'))
    code = (ROOT / 'DGP/paper_kernels.py').read_text(encoding='utf-8')
    functions = {n.name: ast.get_source_segment(code, n) for n in ast.parse(code).body if isinstance(n, ast.FunctionDef)}
    for entry in sources['kernels']:
        body = functions[entry['function']]
        if hashlib.sha256(body.encode()).hexdigest() != entry['sha256']:
            raise ValueError(f'Paper kernel checksum mismatch: {entry["function"]}')
        if original_root:
            original = (original_root / entry['source']).read_text(encoding='utf-8-sig')
            original_functions = {n.name: ast.get_source_segment(original, n) for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
            if body != original_functions[entry['function']]:
                raise ValueError(f'Paper kernel differs from original: {entry["function"]}')
    if checkpoint:
        if sha256(checkpoint) != manifest['checkpoint']['sha256']:
            raise ValueError('Checkpoint SHA-256 does not match the manuscript encoder')
        import torch
        from models.base_model import CGMBaseModel
        settings = dict(config['model'])
        stride = settings.pop('stride')
        forecast = settings.pop('forecast_steps')
        settings['forecast_size'] = forecast
        payload = load_checkpoint(checkpoint)
        effective = {k: p.default for k, p in inspect.signature(CGMBaseModel).parameters.items()
                     if p.default is not inspect.Parameter.empty}
        effective.update(settings)
        effective.update(stride=stride, forecast_steps=forecast)
        for key, expected in payload.get('model_hparams', {}).items():
            if expected is not None and key in effective and effective[key] != expected:
                raise ValueError(f'Checkpoint/config mismatch: {key}')
        state = payload
        for key in ['model_state_dict', 'state_dict', 'model']:
            if key in payload:
                state = payload[key]
                break
        CGMBaseModel(**settings).load_state_dict(state, strict=True)
    if patch_dir:
        meta = json.loads((patch_dir / 'meta.json').read_text(encoding='utf-8'))
        validate_calibration(calibration, reference, meta)
    return dict(release='paper_20260826', artifact_files=len(manifest['files_sha256']),
                copied_kernels=len(sources['kernels']), local_checkpoint_file_checked=bool(checkpoint),
                trajectory_metadata_verified=bool(patch_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT / 'artifacts/dgp/paper_20260826')
    parser.add_argument('--checkpoint-path', type=Path)
    parser.add_argument('--patch-dir', type=Path)
    parser.add_argument('--original-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact_dir, args.checkpoint_path, args.patch_dir, args.original_root), indent=2))
