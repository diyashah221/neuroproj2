"""Analyze behavioral learning and choice-history effects."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm import behavior as bh
from pcvm.io import load_all
from tools import learning_curve as lc

data = load_all(ROOT)
rows, checks = [], {}
for ds, sessions in data.items():
    for s in sessions:
        bh.learning_state(s)
        L = s.learn
        tr = s.trials.iloc[L]
        cmp_ = bh.compare_models(tr.stim.values, tr.choice.values, tr.target.values)
        n = len(L)
        resid = tr.correct.values - s.lc['p']           # what is left of the outcomes after the learning curve
        row = dict(dataset=ds, session=s.name, n_learn=n, acc=tr.correct.mean(),
                   resid_ac1=np.corrcoef(resid[:-1], resid[1:])[0, 1],
                   learning_trial=s.learning_trial, criterion_8of10=lc.criterion_trial(tr.correct.values),
                   **bh.behaviour_stats(tr.stim.values, tr.choice.values, tr.correct.values))
        for name, (nll, k, th) in cmp_.items():
            row['bic ' + name] = 2 * nll + k * np.log(n)
        rows.append(row)
        ag = bh.Agent(cmp_['two rates + stickiness'][2])
        checks[(ds, s.name)] = bh.predictive_check(ag, tr.stim.values, tr.choice.values, tr.correct.values, tr.target.values)

df = pd.DataFrame(rows)
df.to_csv(ROOT / 'results' / 'behavior.csv', index=False)
bic_cols = [c for c in df if c.startswith('bic ')]
for ds, d in df.groupby('dataset'):
    lt = d.learning_trial.dropna()
    print(f'\n{ds}: {len(d)} sessions, median {d.n_learn.median():.0f} learning trials, '
          f'learning trial found in {len(lt)} (median {lt.median() + 1:.0f}), 8/10 criterion median {d.criterion_8of10.median() + 1:.0f}')
    print('  accuracy by thirds %.2f %.2f %.2f, win-stay %.2f, lose-shift %.2f' %
          tuple(d[c].median() for c in ('first', 'mid', 'last', 'win_stay', 'lose_shift')))
    print(f'  lag-1 autocorrelation of outcomes around the learning curve: median {d.resid_ac1.median():+.2f}')
    print('  summed BIC relative to best:', {c[4:]: round(d[c].sum() - d[bic_cols].sum().min(), 1) for c in bic_cols})
    print('  sessions won by each model:', d[bic_cols].idxmin(axis=1).str[4:].value_counts().to_dict())
    # predictive check: where does the observed statistic fall in the simulated distribution
    for stat in ('first', 'last', 'lose_shift', 'max_error_run'):
        q = [np.mean(sim[stat] <= obs[stat]) for (dd, _), (obs, sim) in checks.items() if dd == ds and np.isfinite(obs[stat])]
        print(f'  predictive quantile of {stat}: median {np.median(q):.2f}, outside 2.5-97.5%: {np.mean((np.array(q) < .025) | (np.array(q) > .975)):.2f}')

# figure ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcvm import plotting as P
P.style()
fig = plt.figure(figsize=(11, 6.6))
gs = fig.add_gridspec(2, 4)
show = [(ds, s) for ds in data for s in data[ds] if s.learning_trial is not None][:2]
if 'public' in data:
    show = show[:1] + [('public', s) for s in data['public'] if s.learning_trial is not None][:3]
for i, (ds, s) in enumerate(show[:4]):
    ax = fig.add_subplot(gs[0, i])
    y = s.trials.correct.values[s.learn]; t = np.arange(1, len(y) + 1); r = s.lc
    ax.fill_between(t, r['lo'], r['hi'], color=P.BLUE, alpha=0.18, lw=0)
    ax.plot(t, r['p'], color=P.BLUE)
    ax.plot(t[y == 1], np.full(y.sum(), 1.05), '|', color=P.GREY, ms=3)
    ax.plot(t[y == 0], np.full((1 - y).sum(), -0.05), '|', color=P.ORANGE, ms=3)
    ax.axhline(0.5, color=P.LIGHT, lw=0.8)
    ax.axvline(s.learning_trial + 1, color=P.BLUE, lw=0.9, ls='--')
    ax.set(title=f'{s.name} ({"simulated" if ds == "sim" else "monkey, public"})', xlabel='trial in learning block', ylim=(-0.1, 1.1))
    if i == 0:
        ax.set_ylabel('p(correct)')
ax = fig.add_subplot(gs[1, 0])
for ds, col in (('sim', P.BLUE), ('public', P.ORANGE)):
    if ds in data:
        v = df[df.dataset == ds].learning_trial.dropna() + 1
        ax.hist(v, bins=np.arange(0, 130, 10), histtype='step', color=col, lw=1.6, density=True,
                label=f'{"simulated" if ds == "sim" else "public"} (n={len(v)})')
ax.set(xlabel='learning trial', ylabel='density', title='When learning is detected')
ax.legend()
ax = fig.add_subplot(gs[1, 1])
ax.scatter(df.criterion_8of10 + 1, df.learning_trial + 1, s=12, c=[P.BLUE if d == 'sim' else P.ORANGE for d in df.dataset],
           edgecolor='white', lw=0.3)
ax.plot([0, 120], [0, 120], color=P.LIGHT, lw=0.8)
ax.set(xlabel='first 8-of-10 window ends', ylabel='state-space learning trial', title='Two definitions of "learned"')
ax = fig.add_subplot(gs[1, 2])
names = [c[4:] for c in bic_cols]
w = 0.38
for j, (ds, col) in enumerate((('sim', P.BLUE), ('public', P.ORANGE))):
    if ds in data:
        d = df[df.dataset == ds]
        rel = (d[bic_cols].sum() - d[bic_cols].sum().min()) / len(d)
        ax.barh(np.arange(len(names)) + (j - 0.5) * w, rel.values, height=w - 0.04, color=col)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
ax.set(xlabel='BIC per session above the best model', title='Learning models')
ax = fig.add_subplot(gs[1, 3])
stats_ = ('first', 'last', 'lose_shift', 'max_error_run')
for j, (ds, col) in enumerate((('sim', P.BLUE), ('public', P.ORANGE))):
    for i, st in enumerate(stats_):
        q = [np.mean(sim[st] <= obs[st]) for (dd, _), (obs, sim) in checks.items() if dd == ds and np.isfinite(obs[st])]
        if q:
            ax.plot(np.full(len(q), i) + (j - 0.5) * 0.3 + np.random.default_rng(0).uniform(-0.08, 0.08, len(q)), q,
                    'o', ms=2.5, color=col, alpha=0.6)
ax.axhspan(0.025, 0.975, color=P.GRID, alpha=0.5, lw=0)
ax.set_xticks(range(len(stats_))); ax.set_xticklabels(['first\nthird', 'last\nthird', 'lose-\nshift', 'error\nrun'])
ax.set(ylabel='quantile of data among model simulations', title='Fitted model vs data')
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'behavior.png')
