"""Estimate false-positive rates and power under simulated nulls."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import outcome_tests as ot

ap = argparse.ArgumentParser()
ap.add_argument('--n-sess', type=int, default=1000)
ap.add_argument('--B', type=int, default=499)
ap.add_argument('--blocks', action='store_true', help='also sweep the block length of the local test')
ap.add_argument('--redo', action='store_true')
a = ap.parse_args()

out = ROOT / 'results' / 'calibration.npz'
if out.exists() and not a.redo:
    z = np.load(out)
    res = {}
    for k in z.files:
        sc, kind, m = k.split('|')
        res.setdefault(sc, {}).setdefault(kind, {})[m] = z[k]
else:
    res = ot.calibrate(n_sess=a.n_sess, B=a.B)
    np.savez_compressed(out, **{f'{sc}|{kind}|{m}': p for sc, d in res.items() for kind, dd in d.items() for m, p in dd.items()})

import matplotlib
matplotlib.use('Agg')
ot.plot_calibration(res, ROOT / 'docs' / 'figures' / 'calibration.png')

rows = ['| simulated neuron | test | prev outcome | next outcome | prev > next |', '|---|---|---|---|---|']
for sc, d in res.items():
    for m in d['prev']:
        f = lambda kind: f"{100 * np.mean(d[kind][m] <= .05):.1f}" if m in d[kind] else ''
        rows.append(f"| {sc} | {m} | {f('prev')} | {f('next')} | {f('dir')} |")
(ROOT / 'results' / 'calibration_table.md').write_text('\n'.join(rows) + '\n')

if a.blocks:
    import json
    sw = ot.sweep_block()
    (ROOT / 'results' / 'calibration_blocks.json').write_text(json.dumps(sw, indent=1))
    for sc, d in sw.items():
        print(sc, {b: round(100 * v, 1) for b, v in d.items()})
