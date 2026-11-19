"""Compare outcome-coding tests against trial drift and negative controls."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm.align import trial_windows
from pcvm.io import load_all
from tools import outcome_tests as ot

ap = argparse.ArgumentParser()
ap.add_argument('--B', type=int, default=999)
a = ap.parse_args()


def bh_fdr(p, q=0.05):
    p = np.asarray(p); o = np.argsort(p); m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    out = np.zeros(m, bool)
    if ok.any():
        out[o[:np.where(ok)[0].max() + 1]] = True
    return out


data = load_all(ROOT)
rows = []
for ds, sessions in data.items():
    outs = {s.name: s.trials.correct.values[s.learn] for s in sessions}
    for i, s in enumerate(sessions):
        L = s.learn
        tr = s.trials.iloc[L]
        counts, lab, centre = trial_windows(s)
        S = ot.make_surrogates(tr.correct.values, tr.stim.values, tr.choice.values, tr.target.values,
                               donors=[v for k, v in outs.items() if k != s.name], B=a.B, free_start=True, seed=i)
        edge = float(s.trials.correct.values[L[0] - 1]) if L[0] > 0 else 1.0
        row = dict(dataset=ds, session=s.name, n=len(L), n_err=int((tr.correct == 0).sum()))
        for name, r in (('prev', ot.test_all(counts[L], S, 1, edge=edge)), ('next', ot.test_all(counts[L], S, -1)),
                        ('dir', ot.test_direction(counts[L], S, edge=edge))):
            row.update({f'{name}|{k}': v for k, v in r.items()})
        # where and which way, by the local statistic
        r = ot.partial_corr(np.sqrt(counts[L]), ot.lagged(tr.correct.values, 1, edge), ot.block_basis(len(L)))[0]
        k = int(np.argmax(np.abs(r)))
        row.update(best_win=f'{lab[k]}{centre[k]:+.0f}', best_align=lab[k], best_t=centre[k], best_r=r[k])
        if ds == 'sim':
            tt = s.meta['truth']
            row.update(coding=tt['coding'], ep_c=tt['ep_c'], ep_amp=tt['ep_amp'], ep_sign=tt['ep_sign'])
        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(ROOT / 'results' / 'outcome_tests.csv', index=False)
methods = [c[5:] for c in df if c.startswith('prev|')]
frac = lambda d, c: np.mean(d[c].dropna() <= 0.05) if c in d and d[c].notna().any() else np.nan

groups = {}
if 'sim' in data:
    sim = df[df.dataset == 'sim']
    groups['simulated, no outcome signal'] = sim[~sim.coding.astype(bool)]
    groups['simulated, outcome signal'] = sim[sim.coding.astype(bool)]
if 'public' in data:
    groups['public cells'] = df[df.dataset == 'public']
lines = []
for g, d in groups.items():
    lines.append(f'\n{g} (n = {len(d)})\n  {"":18s}  prev   next   prev>next')
    for m in methods:
        lines.append(f'  {m:18s} {100 * frac(d, "prev|" + m):5.0f}% {100 * frac(d, "next|" + m):5.0f}% '
                     + (f'{100 * frac(d, "dir|" + m):5.0f}%' if 'dir|' + m in d else ''))
    lines.append(f'  local test, FDR 5% across cells: {bh_fdr(d["prev|local"].values).sum()} of {len(d)} for the previous outcome, '
                 f'{bh_fdr(d["next|local"].values).sum()} for the next')
print('\n'.join(lines))
(ROOT / 'results' / 'outcome_tests_summary.txt').write_text('\n'.join(lines) + '\n')

# figure ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcvm import plotting as P
P.style()
fig, ax = plt.subplots(1, len(groups), figsize=(3.9 * len(groups), 4.2), sharey=True)
ax = np.atleast_1d(ax)
y = np.arange(len(methods))[::-1]
for a_, (g, d) in zip(ax, groups.items()):
    pv = [100 * frac(d, 'prev|' + m) for m in methods]
    nx = [100 * frac(d, 'next|' + m) for m in methods]
    a_.barh(y + 0.19, pv, height=0.34, color=P.BLUE, label='previous outcome')
    a_.barh(y - 0.19, nx, height=0.34, color=P.LIGHT, label='next outcome (control)')
    a_.axvline(5, color=P.GREY, lw=0.8, ls='--')
    a_.set(xlim=(0, 100), xlabel='cells significant at .05 (%)', title=f'{g} (n={len(d)})')
    a_.set_yticks(y); a_.set_yticklabels(methods)
ax[-1].legend(loc='lower right')
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'outcome_tests.png')
