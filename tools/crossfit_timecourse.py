"""Estimate learning-stage signals using held-out trials for selection."""
import argparse
import numpy as np


def run_labels(n, run=8):
    lab = np.arange(n) // run
    if n % run and n % run < run / 2 and lab.max() > 0:
        lab[lab == lab.max()] -= 1
    return lab


def local_contrasts(counts, x, run=8):
    """
    counts : (n, W) spike counts,  x : (n,) 1 where the previous trial was rewarded.
    For each run of trials: d = mean(sqrt count | x=0) - mean(sqrt count | x=1) per
    window, its weight n0*n1/(n0+n1), the within-run sum of squares, and dof.
    Runs with only one kind of trial get weight 0.
    """
    y = np.sqrt(np.asarray(counts, float))
    if y.ndim == 1:
        y = y[:, None]
    x = np.asarray(x) > 0.5
    lab = run_labels(len(x), run)
    nb = lab.max() + 1
    d = np.zeros((nb, y.shape[1])); w = np.zeros(nb); ss = np.zeros((nb, y.shape[1])); dof = np.zeros(nb)
    centre = np.zeros(nb)
    for b in range(nb):
        j = lab == b
        centre[b] = np.where(j)[0].mean()
        a, c = y[j & ~x], y[j & x]
        if len(a) and len(c):
            d[b] = a.mean(0) - c.mean(0)
            w[b] = len(a) * len(c) / (len(a) + len(c))
            ss[b] = ((a - a.mean(0)) ** 2).sum(0) + ((c - c.mean(0)) ** 2).sum(0)
            dof[b] = len(a) + len(c) - 2
    return d, w, ss, dof, centre


def _pick(d, w, ss, dof):
    """Window and sign with the largest pooled contrast; also the pooled sd in that window."""
    sd = np.sqrt(ss.sum(0) / max(dof.sum(), 1))
    sd[sd == 0] = np.inf
    t = (w[:, None] * d).sum(0) / (sd * np.sqrt(max(w.sum(), 1e-9)))
    k = int(np.argmax(np.abs(t)))
    return k, np.sign(t[k]) or 1.0, sd[k]


def timecourse(counts, x, run=8):
    """
    Returns dict with, per run of trials:
      centre  mean trial index of the run
      weight  information in the run (0 if all trials followed the same outcome)
      naive   signed contrast in the window/sign chosen from all runs
      cf      signed contrast in the window/sign chosen from the other runs
    plus the naively chosen window, its sign, and the cross-fitted windows.
    Contrasts are in within-run sd units; positive means "in the direction
    the rest of the data said this cell prefers".
    """
    d, w, ss, dof, centre = local_contrasts(counts, x, run)
    k, sgn, sd = _pick(d, w, ss, dof)
    naive = sgn * d[:, k] / sd
    cf = np.zeros(len(w)); wins = np.full(len(w), -1)
    for b in np.where(w > 0)[0]:
        keep = np.arange(len(w)) != b
        if w[keep].sum() == 0:
            continue
        kb, sb, sdb = _pick(d[keep], w[keep], ss[keep], dof[keep])
        cf[b] = sb * d[b, kb] / sdb
        wins[b] = kb
    return dict(centre=centre, weight=w, naive=naive, cf=cf, window=k, sign=sgn, cf_windows=wins)


def summary(tc):
    """One number per cell: weighted mean contrast, naive and cross-fitted."""
    w = tc['weight']
    if w.sum() == 0:
        return np.nan, np.nan
    return float((w * tc['naive']).sum() / w.sum()), float((w * tc['cf']).sum() / w.sum())


def binned(tcs, covs, edges, key='cf', n_boot=1000, seed=0):
    """
    Average over cells within bins of a per-run covariate (trial number, p(correct)...).
    tcs : list of timecourse() outputs, covs : list of (n_runs,) covariate arrays.
    Each cell contributes its weighted mean in the bin; CI by bootstrap over cells.
    """
    rng = np.random.default_rng(seed)
    nb = len(edges) - 1
    cell = np.full((len(tcs), nb), np.nan)
    for i, (tc, c) in enumerate(zip(tcs, covs)):
        for j in range(nb):
            m = (c >= edges[j]) & (c < edges[j + 1]) & (tc['weight'] > 0)
            if m.any():
                cell[i, j] = (tc['weight'][m] * tc[key][m]).sum() / tc['weight'][m].sum()
    mean = np.nanmean(cell, 0)
    boot = np.array([np.nanmean(cell[rng.integers(0, len(cell), len(cell))], 0) for _ in range(n_boot)])
    return mean, np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0), cell


# ---------------------------------------------------------------- demo

def _sim_cell(rng, effect, n=None, W=30, base=8.0):
    n = n or int(rng.integers(80, 180))
    mid, slope = rng.uniform(20, 60), rng.uniform(4, 12)
    p = 0.5 + 0.46 / (1 + np.exp(-(np.arange(n) - mid) / slope))
    out = (rng.random(n) < p).astype(float)
    prev = np.r_[1.0, out[:-1]]
    drift = np.zeros(n)
    phi, sd = rng.uniform(0.6, 0.98), rng.uniform(0.15, 0.5)
    for t in range(1, n):
        drift[t] = phi * drift[t - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
    lr = np.tile(drift[:, None], (1, W)) + rng.normal(0, 0.3, W)[None]
    gate = np.clip((0.96 - p) / 0.46, 0, 1)
    if effect:
        c0 = rng.integers(0, W - 4)
        lr[:, c0:c0 + 4] += (rng.choice([-1, 1]) * effect * gate * (0.5 - prev))[:, None]
    counts = rng.poisson(base * np.exp(lr))
    # expected contrast on the sqrt scale in sd units (sd of sqrt of a Poisson count is ~0.5)
    truth = np.sqrt(base) * (np.exp(effect * gate / 4) - np.exp(-effect * gate / 4)) / 0.5
    return counts, prev, p, truth


def _demo(n_cells=300, seed=0, save=None, show=True):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    edges = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.97])
    mids = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for a, eff, title in ((ax[0], 0.0, 'cells with no outcome signal'), (ax[1], 0.6, 'cells whose signal fades with learning')):
        tcs, covs, tr = [], [], []
        for _ in range(n_cells):
            counts, prev, p, truth = _sim_cell(rng, eff)
            tc = timecourse(counts, prev)
            lab = run_labels(len(prev))
            tcs.append(tc)
            covs.append(np.array([p[lab == b].mean() for b in range(lab.max() + 1)]))
            tr.append(np.array([truth[lab == b].mean() for b in range(lab.max() + 1)]))
        for key, col, name in (('naive', '#eb6834', 'window chosen on all trials'), ('cf', '#2a78d6', 'cross-fitted')):
            m, lo, hi, _ = binned(tcs, covs, edges, key)
            a.fill_between(mids, lo, hi, color=col, alpha=0.18, lw=0)
            a.plot(mids, m, 'o-', color=col, lw=1.8, ms=4, label=name)
        tm = [np.mean(np.concatenate([t[(c >= edges[j]) & (c < edges[j + 1])] for t, c in zip(tr, covs)])) for j in range(len(mids))]
        a.plot(mids, tm, '--', color='#52514e', lw=1.2, label='true size in the right window')
        a.axhline(0, color='#c3c2b7', lw=0.8)
        a.set_title(title, fontsize=9.5)
        a.set_xlabel('p(correct) at that point in learning')
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel('after error minus after reward\n(sd units, preferred direction)')
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Cross-fitted time course of an outcome signal over learning.')
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--counts', help='.npy, trials x windows')
    ap.add_argument('--prev', help='.npy, 1 where the previous trial was rewarded')
    ap.add_argument('--run', type=int, default=8)
    ap.add_argument('--save-fig')
    a = ap.parse_args()
    if a.counts and a.prev:
        tc = timecourse(np.load(a.counts), np.load(a.prev), a.run)
        nv, cf = summary(tc)
        print(f'window {tc["window"]}, sign {tc["sign"]:+.0f}; mean contrast naive {nv:.3f}, cross-fitted {cf:.3f} (sd units)')
        for c, w, n_, f in zip(tc['centre'], tc['weight'], tc['naive'], tc['cf']):
            print(f'  trials ~{c:5.1f}  weight {w:4.1f}  naive {n_:+.2f}  cross-fitted {f:+.2f}')
    else:
        if a.save_fig:
            import matplotlib
            matplotlib.use('Agg')
        _demo(save=a.save_fig, show=a.save_fig is None)
