"""Fit neural encoding models and compare held-out performance."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm import behavior as bh, glm
from pcvm.io import load_all

data = load_all(ROOT)
rows, kernels = [], {}
for ds, sessions in data.items():
    for s in sessions:
        bh.learning_state(s)
        idx = s.learn
        r = glm.analyse_cell(s, idx)

        # flexible benchmark on the same folds: bin, trial number, previous outcome, hand, stimulus, learning state
        nb = glm.bin_raster(s.sym[idx]).shape[1]
        tr = r['trial']
        F = np.c_[np.tile(np.arange(nb), len(idx)), tr, s.prev_correct()[idx][tr], s.trials.choice.values[idx][tr],
                  s.trials.stim.values[idx][tr], s.p_hat[idx][tr]]
        dm = dn = 0.0
        for f in np.unique(r['fold']):
            a, b = r['fold'] != f, r['fold'] == f
            gb = HistGradientBoostingRegressor(loss='poisson', max_iter=150, learning_rate=0.08, max_leaf_nodes=12,
                                               min_samples_leaf=40, l2_regularization=1.0, random_state=0)
            gb.fit(F[a], r['y'][a])
            dm += glm.deviance(r['y'][b], gb.predict(F[b]))
            dn += glm.deviance(r['y'][b], np.full(b.sum(), r['y'][a].mean()))
        row = dict(dataset=ds, session=s.name, n_trials=len(idx), rate=s.sym[idx].mean() * 1000,
                   r2_glm=r['r2_full'], r2_gbm=1 - dm / dn, d_prev=r['d_prev'], d_hand=r['d_hand'], d_slow=r['d_slow'],
                   lam=r['lam'], slow_every=r['slow_every'])
        k = r['prev_kernel']
        t_bins = s.t_sym[0] + glm.BIN * (np.arange(len(k)) + 0.5)
        row['kernel_peak_ms'] = t_bins[np.argmax(np.abs(k))]
        row['kernel_peak'] = k[np.argmax(np.abs(k))]
        if ds == 'sim':
            tt = s.meta['truth']
            row.update(coding=tt['coding'], ep_c=tt['ep_c'], ep_amp=tt['ep_amp'], ep_sign=tt['ep_sign'], hand_amp=tt['hand_amp'])
        rows.append(row)
        kernels[s.name] = k

df = pd.DataFrame(rows)
df.to_csv(ROOT / 'results' / 'encoding.csv', index=False)
np.savez(ROOT / 'results' / 'prev_kernels.npz', **kernels)

for ds, d in df.groupby('dataset'):
    print(f'\n{ds}: {len(d)} cells')
    print(f'  held-out deviance explained, median: GLM {d.r2_glm.median():.3f}, boosted trees {d.r2_gbm.median():.3f}; '
          f'GLM >= trees in {np.mean(d.r2_glm >= d.r2_gbm):.0%} of cells')
    print('  median loss when removed: previous outcome %.4f, hand %.4f, slow terms %.4f' %
          (d.d_prev.median(), d.d_hand.median(), d.d_slow.median()))
    print(f'  cells where previous-outcome terms help on held-out trials: {np.mean(d.d_prev > 0):.0%}')
    if ds == 'sim':
        cod = d.coding.astype(bool)
        c = d[cod]; n = d[~cod]
        print(f'  ... among cells built with an outcome signal {np.mean(c.d_prev > 0):.0%}, without {np.mean(n.d_prev > 0):.0%}')
        ok = c[(c.d_prev > 0) & (c.ep_c > -1350) & (c.ep_c < 450)]
        print(f'  epoch time recovered (r = {np.corrcoef(ok.ep_c, ok.kernel_peak_ms)[0, 1]:.2f}, '
              f'median abs error {np.median(np.abs(ok.ep_c - ok.kernel_peak_ms)):.0f} ms, n = {len(ok)}); '
              f'sign right in {np.mean(np.sign(ok.kernel_peak) == ok.ep_sign):.0%}')

# figure ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcvm import plotting as P
from pcvm.align import sdf
P.style()
fig, ax = plt.subplots(2, 3, figsize=(11, 6.4))
sim = df[df.dataset == 'sim'].copy()
sim['coding'] = sim.coding.astype(bool)
# example cell: strongest simulated coding cell with its epoch inside the window
ex = sim[sim.coding & (sim.ep_c > -1200) & (sim.ep_c < 300)].sort_values('d_prev').iloc[-1]
s = next(x for x in data['sim'] if x.name == ex.session)
bh.learning_state(s)
L = s.learn; early = L[s.p_hat[L] < 0.75]; prev = s.prev_correct()
for m, col, lab in ((prev[early] == 1, P.C_CORRECT, 'after reward'), (prev[early] == 0, P.C_WRONG, 'after error')):
    ax[0, 0].plot(s.t_sym, sdf(s.sym[early][m], 25), color=col, label=f'{lab} (n={m.sum()})')
P.events(ax[0, 0])
ax[0, 0].axvspan(ex.ep_c - 110, ex.ep_c + 110, color=P.GRID, alpha=0.6, lw=0)
ax[0, 0].set(xlabel='ms from symbol', ylabel='simple spikes / s', title=f'{s.name}, early learning (shaded: true epoch)')
ax[0, 0].legend()
k = kernels[s.name]; tb = s.t_sym[0] + glm.BIN * (np.arange(len(k)) + 0.5)
ax[0, 1].plot(tb, k, color=P.INK)
ax[0, 1].axhline(0, color=P.LIGHT, lw=0.8); P.events(ax[0, 1])
ax[0, 1].axvspan(ex.ep_c - 110, ex.ep_c + 110, color=P.GRID, alpha=0.6, lw=0)
ax[0, 1].set(xlabel='ms from symbol', ylabel='log rate, error minus reward', title='Fitted previous-outcome kernel')
ok = sim[sim.coding & (sim.d_prev > 0)]
ax[0, 2].scatter(ok.ep_c, ok.kernel_peak_ms, s=14, color=P.BLUE, edgecolor='white', lw=0.3)
ax[0, 2].plot([-1400, 500], [-1400, 500], color=P.LIGHT, lw=0.8)
ax[0, 2].set(xlabel='true epoch centre (ms)', ylabel='peak of fitted kernel (ms)', title='Epoch timing recovered')
bins = np.linspace(-0.004, 0.03, 35)
ax[1, 0].hist(sim[~sim.coding].d_prev.clip(bins[0], bins[-1]), bins, color=P.LIGHT, label='no outcome signal')
ax[1, 0].hist(sim[sim.coding].d_prev.clip(bins[0], bins[-1]), bins, histtype='step', color=P.BLUE, lw=1.6, label='outcome signal')
if 'public' in data:
    ax[1, 0].hist(df[df.dataset == 'public'].d_prev.clip(bins[0], bins[-1]), bins, histtype='step', color=P.ORANGE, lw=1.6, label='public cells')
ax[1, 0].axvline(0, color=P.GREY, lw=0.8)
ax[1, 0].set(xlabel='held-out deviance explained lost without\nprevious-outcome terms', ylabel='cells', title='Unique contribution of previous outcome')
ax[1, 0].legend()
for ds, col in (('sim', P.BLUE), ('public', P.ORANGE)):
    d = df[df.dataset == ds]
    ax[1, 1].scatter(d.r2_gbm.clip(-0.1), d.r2_glm.clip(-0.1), s=14, color=col, edgecolor='white', lw=0.3,
                     label='simulated' if ds == 'sim' else 'public')
lim = [-0.12, max(df.r2_gbm.max(), df.r2_glm.max()) * 1.05]
ax[1, 1].plot(lim, lim, color=P.LIGHT, lw=0.8)
ax[1, 1].set(xlabel='boosted trees (values below -0.1 drawn at -0.1)', ylabel='GLM', title='Held-out deviance explained')
ax[1, 1].legend()
for j, (c, lab) in enumerate((('d_prev', 'previous\noutcome'), ('d_hand', 'hand'), ('d_slow', 'slow terms'))):
    for i, (ds, col) in enumerate((('sim', P.BLUE), ('public', P.ORANGE))):
        v = df[df.dataset == ds][c]
        if len(v):
            q = np.percentile(v, [25, 50, 75])
            ax[1, 2].plot([j + (i - 0.5) * 0.3] * 2, q[[0, 2]], color=col, lw=2)
            ax[1, 2].plot(j + (i - 0.5) * 0.3, q[1], 'o', color=col, ms=6, mec='white')
ax[1, 2].set_xticks(range(3)); ax[1, 2].set_xticklabels(['previous\noutcome', 'hand', 'slow terms'])
ax[1, 2].set_yscale('symlog', linthresh=0.002)
ax[1, 2].axhline(0, color=P.LIGHT, lw=0.8)
ax[1, 2].set(ylabel='deviance explained lost (median, IQR)', title='What each group of terms buys')
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'encoding.png')
