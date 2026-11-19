"""Generate synthetic sessions with known behavior and spike labels."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm.simulate import simulate_session

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=110)
ap.add_argument('--seed', type=int, default=7)
a = ap.parse_args()

out = ROOT / 'data' / 'raw'
out.mkdir(parents=True, exist_ok=True)
for f in out.glob('*.npz'):
    f.unlink()

seeds = np.random.SeedSequence(a.seed).generate_state(a.n)
n_b = round(a.n * 0.73)           # two animals, unequal numbers of cells like the real sample
size = 0
for i in range(a.n):
    monkey = 'B' if i < n_b else 'S'
    name = f'{monkey}{i + 1:03d}'
    raw = simulate_session(int(seeds[i]), name, monkey)
    tr = raw.pop('trials'); truth = raw.pop('truth')
    path = out / f'{name}.npz'
    np.savez(path, truth=json.dumps(truth), **raw, **{'tr_' + k: v for k, v in tr.items()})
    size += path.stat().st_size
print(f'{a.n} sessions -> {out}  ({size / 1e6:.0f} MB)')
