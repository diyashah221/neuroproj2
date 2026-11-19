"""Analyze pseudo-population decoding and temporal generalization."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm import behavior as bh, decoding as dc
from pcvm.align import trial_windows
from pcvm.io import load_all

MIN = 5            # trials per class a cell needs in a stage to be included (as in Ipata et al.)
data = load_all(ROOT)


def gather(sessions, target, stg, centre=True, keep=None):
    """Per-cell (features, labels) for one decoding problem.  Features: sliding-window counts."""
    cells, centres, labs = [], None, None
    for s in sessions:
        if keep is not None and not keep(s):
            continue
        counts, lab, t = trial_windows(s, 200, 100)
        x = np.sqrt(counts.astype(float))
        L = s.learn
        st = bh.stage(s.p_hat, s.trials.block.values)
        if target == 'prev':
            y = (s.prev_correct() == 0).astype(int)          # 1 = after an error
        else:
            y = s.trials.choice.values.astype(int)
        if centre:
            x[L] = dc.local_centre(x[L])
            ot_ = s.block('OT')
            if len(ot_):
                x[ot_] = dc.local_centre(x[ot_])
        j = np.where(st == stg)[0]
        if min(np.sum(y[j] == 0), np.sum(y[j] == 1)) >= MIN:
            cells.append((x[j], y[j]))
        centres, labs = t, lab
    return cells, labs, centres


res = {}
store = {}
for ds, sessions in data.items():
    for s in sessions:
        bh.learning_state(s)
    res[ds] = {}
    for target, stages in (('prev', ('early', 'late')), ('hand', ('OT', 'early', 'late'))):
        for stg in stages:
            cells, labs, t = gather(sessions, target, stg)
            if len(cells) < 8:
                print(f'{ds} {target} {stg}: only {len(cells)} cells with >= {MIN} trials per class, skipped')
                continue
            gen = target == 'prev' and stg == 'early'
            out = dc.decode(cells, n_rep=20, generalise=gen)
            acc, G = out if gen else (out, None)
            null = dc.null_trial_level(cells, n_null=100, n_rep=2, local=(target == 'prev'))
            thr = np.quantile(null.max(1), 0.95)                  # familywise over windows
            k = int(np.argmax(acc))
            res[ds][f'{target}|{stg}'] = dict(n_cells=len(cells), peak=float(acc[k]), peak_window=f'{labs[k]}{t[k]:+.0f}',
                                              null_95=float(np.quantile(null, 0.95)), null_max_95=float(thr),
                                              n_windows_sig=int(np.sum(acc > thr)))
            store[(ds, target, stg)] = dict(acc=acc, null=null, G=G, labs=labs, t=t, cells=cells, thr=thr)
            print(f'{ds:6s} {target:4s} {stg:5s}: {len(cells):3d} cells, peak accuracy {acc[k]:.2f} at {labs[k]}{t[k]:+.0f} ms, '
                  f'95% of trial-level null maxima {thr:.2f}, {np.sum(acc > thr)} of {len(acc)} windows above it')

# which classifier, at the best window for previous outcome in early learning
for ds in data:
    key = (ds, 'prev', 'early')
    if key in store:
        w = int(np.argmax(store[key]['acc']))
        cmp_ = dc.compare_classifiers(store[key]['cells'], w)
        res[ds]['classifiers'] = cmp_
        print(f'{ds}: classifiers at the best window:', {k: round(v, 3) for k, v in cmp_.items()})

# what counts as chance: the two nulls at one window
key = ('sim', 'prev', 'early') if ('sim', 'prev', 'early') in store else next(iter(store))
w = int(np.argmax(store[key]['acc']))
n_trial = store[key]['null'][:, w]
n_pseudo = dc.null_pseudo_level(store[key]['cells'], w, n_null=300)
res['null_sd'] = dict(trial_level=float(n_trial.std()), pseudo_level=float(n_pseudo.std()))
print(f'sd of chance accuracy: relabelling recorded trials {n_trial.std():.3f}, shuffling pseudo-trial labels {n_pseudo.std():.3f}')

# can a decoder read "previous outcome" from cells that have none?  (simulation only)
if 'sim' in data:
    nulls = lambda s: not s.meta['truth']['coding']
    for centre in (False, True):
        cells, labs, t = gather(data['sim'], 'prev', 'early', centre=centre, keep=nulls)
        acc = dc.decode(cells, n_rep=20)
        null = dc.null_trial_level(cells, n_null=100, n_rep=2, local=centre)
        thr = np.quantile(null.max(1), 0.95)
        store[('nullcells', centre)] = dict(acc=acc, null=null, t=t, labs=labs, thr=thr)
        res[f'null_cells_centred_{centre}'] = dict(n_cells=len(cells), peak=float(acc.max()), mean=float(acc.mean()),
                                                   null_max_95=float(thr), n_windows_sig=int(np.sum(acc > thr)))
        print(f'cells with no outcome signal, {"centred within runs" if centre else "raw counts, plain shuffle"}: '
              f'{len(cells)} cells, mean accuracy {acc.mean():.3f}, peak {acc.max():.2f}, '
              f'{np.sum(acc > thr)} of {len(acc)} windows above the 95% null maximum')

(ROOT / 'results' / 'decoding.json').write_text(json.dumps(res, indent=1))

# figure ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcvm import plotting as P
P.style()


def xaxis(labs, t):
    """Windows are symbol-aligned then movement-aligned; put them on one axis with a gap."""
    x = np.arange(len(t), dtype=float)
    x[labs == 'mov'] += 1
    return x


def ticks(ax, labs, t, x):
    sel = [i for i in range(len(t)) if (labs[i] == 'sym' and t[i] in (-1200, -800, -400, 0)) or (labs[i] == 'mov' and t[i] in (0, 400))]
    ax.set_xticks(x[sel])
    ax.set_xticklabels([f'{"S" if labs[i] == "sym" else "M"}{t[i]:+.0f}' for i in sel])
    ax.set_xlabel('window centre, ms from symbol (S) or movement (M)')


fig, ax = plt.subplots(2, 3, figsize=(12, 6.6))
for a_, ds, title in ((ax[0, 0], 'sim', 'Previous outcome, simulated cells'), (ax[0, 1], 'public', 'Previous outcome, public cells')):
    for stg, col in (('early', P.BLUE), ('late', P.AQUA)):
        k = (ds, 'prev', stg)
        if k in store:
            d = store[k]; x = xaxis(d['labs'], d['t'])
            a_.plot(x, d['acc'], color=col, label=f'{stg} learning ({len(d["cells"])} cells)')
            a_.axhline(d['thr'], color=col, lw=0.8, ls=':')
            ticks(a_, d['labs'], d['t'], x)
    a_.axhline(0.5, color=P.LIGHT, lw=0.8)
    a_.set(ylabel='decoding accuracy', title=title, ylim=(0.3, 1.22))
    a_.set_yticks([0.4, 0.6, 0.8, 1.0])
    a_.legend(title='dotted: 95% of chance maxima', title_fontsize=7.5, loc='upper right')
d = store.get(('sim', 'prev', 'early'))
if d is not None and d['G'] is not None:
    im = ax[0, 2].imshow(d['G'], origin='lower', cmap='Blues', vmin=0.5, vmax=max(0.75, d['G'].max()), aspect='auto')
    ax[0, 2].set(xlabel='test window', ylabel='train window', title='Train on one time, test on another')
    plt.colorbar(im, ax=ax[0, 2], fraction=0.046, label='accuracy')
a_ = ax[1, 0]
for centre, col, lab in ((False, P.ORANGE, 'raw counts'), (True, P.BLUE, 'centred within runs of 8 trials')):
    d = store.get(('nullcells', centre))
    if d:
        x = xaxis(d['labs'], d['t'])
        a_.plot(x, d['acc'], color=col, label=lab)
        a_.axhline(d['thr'], color=col, lw=0.8, ls=':')
        ticks(a_, d['labs'], d['t'], x)
a_.axhline(0.5, color=P.LIGHT, lw=0.8)
a_.set(ylabel='decoding accuracy', title='Cells built without an outcome signal', ylim=(0.25, 1.05))
a_.legend(loc='upper right')
a_ = ax[1, 1]
bins = np.linspace(0.3, 0.7, 41)
a_.hist(n_trial, bins, color=P.BLUE, alpha=0.75, label=f'relabel recorded trials (sd {n_trial.std():.3f})', density=True)
a_.hist(n_pseudo, bins, color=P.ORANGE, alpha=0.75, label=f'shuffle pseudo-trial labels (sd {n_pseudo.std():.3f})', density=True)
a_.set(xlabel='accuracy with no information', ylabel='density', title='Two ideas of chance')
a_.legend()
a_ = ax[1, 2]
for ds, ls in (('sim', '-'), ('public', '--')):
    for stg, col in (('OT', P.GREY), ('early', P.BLUE), ('late', P.AQUA)):
        k = (ds, 'hand', stg)
        if k in store:
            d = store[k]; x = xaxis(d['labs'], d['t'])
            a_.plot(x, d['acc'], color=col, ls=ls, label=f'{stg}, {"simulated" if ds == "sim" else "public"}')
            ticks(a_, d['labs'], d['t'], x)
a_.axhline(0.5, color=P.LIGHT, lw=0.8)
a_.set(ylabel='decoding accuracy', title='Hand used', ylim=(0.3, 1.3))
a_.set_yticks([0.4, 0.6, 0.8, 1.0])
a_.legend(fontsize=7, ncol=2, loc='upper left')
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'population.png')
