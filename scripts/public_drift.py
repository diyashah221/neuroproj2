"""Estimate trial-to-trial rate drift in the public recordings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm.align import window_count
from pcvm.io import load_all

rows = []
for s in load_all(ROOT)['public']:
    c = window_count(s.sym, s.t_sym, -1400, 0)[s.learn].astype(float)
    m, v = c.mean(), c.var(ddof=1)
    x = c - m
    ac = [np.mean(x[:-k] * x[k:]) / x.var() for k in (1, 2, 8)]
    rows.append(dict(session=s.name, rate=m / 1.4, fano=v / m, cv_rate=np.sqrt(max(v - m, 0)) / m,
                     ac1=ac[0], ac2=ac[1], ac8=ac[2]))
df = pd.DataFrame(rows)
df.to_csv(ROOT / 'results' / 'public_drift.csv', index=False)
q = lambda c: '%.2f (%.2f-%.2f)' % tuple(np.percentile(df[c], [50, 25, 75]))
print(f'{len(df)} cells, median (IQR)')
print('  rate /s            ', q('rate'))
print('  Fano factor        ', q('fano'))
print('  CV of rate, trials ', q('cv_rate'))
print('  autocorr lag 1     ', q('ac1'))
print('  autocorr lag 2     ', q('ac2'))
print('  autocorr lag 8     ', q('ac8'))
