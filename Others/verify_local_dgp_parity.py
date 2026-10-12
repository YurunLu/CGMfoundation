"""Read-only numerical comparison with the current local Nicheng release."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DGP.compute import raw_metrics, assemble, VERSION
from ExportTrajectory import sha256


def verify(root):
    reference = root / 'Results/DGP/GlycemicBasinCompactDGP/glycemic_basin_reference.npz'
    raw, _ = raw_metrics(root / 'Results/PatchEmb/Nicheng', reference)
    current = pd.read_csv(root / 'Results/DGP/DGP_LyapunovIntegrated/ple_sli_current_results/tables/nicheng_current_dgp.csv', dtype={'sample_id': str})
    stats = {}
    for row in pd.read_csv(root / 'Results/DGP/FinalDGP/tables/patch_lvar_ngt_standardization.csv').itertuples():
        stats[row.feature] = [row.ngt_mean, row.ngt_sd]
    for row in pd.read_csv(root / 'Results/DGP/DGP_LyapunovIntegrated/tables/sli_v2_ngt_calibration_stats.csv').itertuples():
        stats[row.component] = [row.ngt_median, row.ngt_scale]
    ngt = current.loc[current.Type.eq('NGT')]
    tail = ngt.latent_high_burden + 1.5 * ngt.latent_high_extreme_occupancy
    stats['l_hbgi_tail_raw'] = [float(tail.mean()), float(tail.std(ddof=1))]
    calibration = dict(version=VERSION, stats=stats,
                       ple_healthy_reference=float(current.PLE_healthy_reference_per_hour.iloc[0]),
                       reference_sha256=sha256(reference))
    output = assemble(raw, calibration)
    joined = output.merge(current, on='sample_id', suffixes=('_new', '_current'), validate='one_to_one')
    if len(joined) != len(output) or len(joined) != len(current):
        raise AssertionError('Output and source recording sets differ')
    summary = {'recordings': len(joined), 'rtol': 1e-5, 'atol': 1e-5, 'metrics': {}}
    for col in ['TOB', 'HAD', 'DynVar', 'DynHBGI', 'PLE', 'SLI', 'DRS']:
        a = joined[col + '_new'].to_numpy(float)
        b = joined[col + '_current'].to_numpy(float)
        summary['metrics'][col] = dict(maximum_absolute_error=float(np.nanmax(abs(a - b))),
                                       matches=bool(np.allclose(a, b, rtol=1e-5, atol=1e-5, equal_nan=True)),
                                       missing_new=int(np.isnan(a).sum()), missing_current=int(np.isnan(b).sum()))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    summary = verify(args.original_root)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n', encoding='utf-8')
    if not all(row['matches'] for row in summary['metrics'].values()):
        raise SystemExit(1)
