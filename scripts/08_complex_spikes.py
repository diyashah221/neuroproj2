"""Measure complex-spike responses in the synthetic recordings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm import behavior as bh, glm
from pcvm.align import window_count
from pcvm.io import load_all

sessions = load_all(ROOT).get('sim', [])
rows = []
psth = {('sym', st): [] for st in ('OT', 'early', 'late')}
psth.update({('mov', st): [] for st in ('OT', 'early', 'late')})
for s in sessions:
    bh.learning_state(s)
    st = bh.stage(s.p_hat, s.trials.block.values)
    for stg in ('OT', 'early', 'late'):
        j = st == stg
        if j.sum() >= 10:
            psth[('sym', stg)].append(s.cs_sym[j].mean(0) * 1000)
            psth[('mov', stg)].append(s.cs_mov[j].mean(0) * 1000)
    p_fail = 1 - s.p_hat
    prev_wrong = 1 - s.prev_correct()
    cur_wrong = 1 - s.trials.correct.values
    X = np.c_[p_fail - p_fail.mean(), prev_wrong - prev_wrong.mean(), cur_wrong - cur_wrong.mean()]
    for name, y in (('symbol', window_count(s.cs_sym, s.t_sym, 50, 150)), ('movement', window_count(s.cs_mov, s.t_mov, 50, 150)),
                    ('fixation cue', window_count(s.cs_sym, s.t_sym, -750, -650))):
        b0, b = glm.fit(X, y.astype(float), lam=0.5)
        rows.append(dict(session=s.name, window=name, p_resp=float((y > 0).mean()), b_pfail=b[0], b_prev=b[1], b_cur=b[2],
                         true_gain=s.meta['truth']['cs_pfail']))

df = pd.DataFrame(rows)
df.to_csv(ROOT / 'results' / 'complex_spikes.csv', index=False)
for w, d in df.groupby('window', sort=False):
    line = f'{w:13s} trials with a complex spike {d.p_resp.mean():.0%};'
    for c, lab in (('b_pfail', 'p(error)'), ('b_prev', 'previous error'), ('b_cur', 'current error')):
        p = stats.wilcoxon(d[c]).pvalue
        line += f'  {lab}: median {d[c].median():+.2f} (p = {p:.1g})'
    print(line)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from pcvm import plotting as P
P.style()
fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
for a_, al, t, lim in ((ax[0], 'sym', sessions[0].t_sym, (-1000, 400)), (ax[1], 'mov', sessions[0].t_mov, (-300, 500))):
    for stg, col in (('OT', P.GREY), ('early', P.BLUE), ('late', P.AQUA)):
        m = gaussian_filter1d(np.mean(psth[(al, stg)], 0), 12)
        a_.plot(t, m, color=col, label=stg if stg == 'OT' else stg + ' learning')
    P.events(a_, (0,) if al == 'mov' else (-800, 0))
    a_.set(xlim=lim, xlabel=f'ms from {"symbol" if al == "sym" else "movement"}', ylabel='complex spikes / s',
           title='Aligned to symbol (fixation cue at -800)' if al == 'sym' else 'Aligned to movement')
ax[0].legend()
a_ = ax[2]
for i, (w, col) in enumerate((('symbol', P.BLUE), ('movement', P.VIOLET))):
    d = df[df.window == w]
    for j, c in enumerate(('b_pfail', 'b_prev', 'b_cur')):
        q = np.percentile(d[c], [25, 50, 75])
        a_.plot([j + (i - 0.5) * 0.3] * 2, q[[0, 2]], color=col, lw=2)
        a_.plot(j + (i - 0.5) * 0.3, q[1], 'o', color=col, ms=6, mec='white', label=w + ' window' if j == 0 else None)
a_.axhline(0, color=P.LIGHT, lw=0.8)
a_.set_xticks(range(3)); a_.set_xticklabels(['p(error)', 'previous\nerror', 'current\nerror'])
a_.set(ylabel='log rate coefficient (median, IQR)', title='What the complex-spike response follows')
a_.legend()
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'complex_spikes.png')
