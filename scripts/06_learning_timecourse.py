"""Estimate outcome effects across learning without reusing selection trials."""
import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', 'Mean of empty slice')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm import behavior as bh
from pcvm.align import trial_windows
from pcvm.io import load_all
from tools import crossfit_timecourse as cf

EDGES = np.array([0.0, 0.6, 0.7, 0.8, 0.9, 1.01])
data = load_all(ROOT)
rows, store = [], {}
for ds, sessions in data.items():
    for s in sessions:
        bh.learning_state(s)
        L = s.learn
        counts, lab, centre = trial_windows(s)
        tc = cf.timecourse(counts[L], s.prev_correct()[L])
        runs = cf.run_labels(len(L))
        p_run = np.array([s.p_hat[L][runs == b].mean() for b in range(runs.max() + 1)])
        nv, cv = cf.summary(tc)
        row = dict(dataset=ds, session=s.name, naive=nv, crossfit=cv, window=f'{lab[tc["window"]]}{centre[tc["window"]]:+.0f}',
                   win_align=lab[tc['window']], win_t=centre[tc['window']], sign=tc['sign'])
        grp = 'public cells'
        if ds == 'sim':
            tt = s.meta['truth']
            row.update(coding=tt['coding'], ep_c=tt['ep_c'], ep_amp=tt['ep_amp'], ep_sign=tt['ep_sign'])
            grp = 'simulated, outcome signal' if tt['coding'] else 'simulated, no outcome signal'
        rows.append(row)
        store.setdefault(grp, []).append((tc, p_run))

df = pd.DataFrame(rows)
df.to_csv(ROOT / 'results' / 'timecourse.csv', index=False)
curves = {}
for g, v in store.items():
    tcs, covs = zip(*v)
    curves[g] = {k: cf.binned(list(tcs), list(covs), EDGES, k) for k in ('naive', 'cf')}
    m, lo, hi, cell = curves[g]['cf']
    mn = curves[g]['naive'][0]
    print(f'\n{g} (n = {len(tcs)})')
    print('  p(correct) bin     ' + '  '.join(f'{EDGES[j]:.1f}-{min(EDGES[j + 1], 1):.1f}' for j in range(len(m))))
    print('  window on all data ' + '  '.join(f'{x:7.2f}' for x in mn))
    print('  cross-fitted       ' + '  '.join(f'{x:7.2f}' for x in m))
    print('  95% CI low         ' + '  '.join(f'{x:7.2f}' for x in lo))
    print('  95% CI high        ' + '  '.join(f'{x:7.2f}' for x in hi))

if 'sim' in data:
    c = df[(df.dataset == 'sim') & df.coding.fillna(False).astype(bool)]
    print(f'\nsimulated cells with a signal: preferred direction right in {np.mean(c.sign == c.ep_sign):.0%}')

# figure ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcvm import plotting as P
P.style()
names = list(curves)
fig, ax = plt.subplots(1, len(names) + 1, figsize=(3.6 * (len(names) + 1), 3.7))
x = np.arange(len(EDGES) - 1)
xl = ['<.6', '.6-.7', '.7-.8', '.8-.9', '>.9']
for a_, g in zip(ax, names):
    for key, col, lab in (('naive', P.ORANGE, 'window chosen on all trials'), ('cf', P.BLUE, 'cross-fitted')):
        m, lo, hi, _ = curves[g][key]
        a_.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        a_.plot(x, m, 'o-', color=col, ms=4, label=lab)
    a_.axhline(0, color=P.LIGHT, lw=0.8)
    a_.set_xticks(x); a_.set_xticklabels(xl)
    a_.set(xlabel='estimated p(correct)', title=f'{g} (n={len(store[g])})')
ax[0].set_ylabel('after error minus after reward\n(sd units, preferred direction)')
ax[0].legend()
a_ = ax[-1]
if 'sim' in data:
    sim = df[df.dataset == 'sim']
    for coding, col, lab in ((True, P.BLUE, 'outcome signal'), (False, P.GREY, 'no outcome signal')):
        d = sim[sim.coding.astype(bool) == coding]
        a_.scatter(d.naive, d.crossfit, s=13, color=col, edgecolor='white', lw=0.3, label=lab)
    lim = [min(sim.crossfit.min(), 0) - 0.1, sim.naive.max() + 0.1]
    a_.plot(lim, lim, color=P.LIGHT, lw=0.8)
    a_.axhline(0, color=P.LIGHT, lw=0.8)
    a_.set(xlabel='window chosen on all trials', ylabel='cross-fitted', title='Per cell, simulated (sd units)')
    a_.legend()
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'learning_timecourse.png')
