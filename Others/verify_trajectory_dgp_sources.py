"""Verify numerical-function checksums for radial PLE DGP."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def functions(path):
    source = path.read_text(encoding='utf-8-sig')
    return {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    manifest = json.loads((ROOT / 'docs/TRAJECTORY_DGP_SOURCES.json').read_text())
    kernels = functions(ROOT / manifest['module'])
    entries = manifest['kernels']
    for entry in entries:
        name = entry['function']
        assert hashlib.sha256(kernels[name].encode()).hexdigest() == entry['sha256'], name
    print(f'PASS: {len(entries)} numerical functions match their release checksums')
