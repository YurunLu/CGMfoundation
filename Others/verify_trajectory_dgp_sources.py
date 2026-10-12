"""Verify copied DGP kernels against recorded hashes and optionally source files."""
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
    parser.add_argument('--original-root', type=Path)
    args = parser.parse_args()
    kernels = functions(ROOT / 'DGP/metric_kernels.py')
    entries = json.loads((ROOT / 'docs/TRAJECTORY_DGP_SOURCES.json').read_text())['kernels']
    for entry in entries:
        name = entry['function']
        assert hashlib.sha256(kernels[name].encode()).hexdigest() == entry['sha256'], name
        if args.original_root:
            assert kernels[name] == functions(args.original_root / entry['source'])[name], name
    print(f'PASS: {len(entries)} numerical kernels match recorded source hashes')
