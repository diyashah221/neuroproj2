"""Estimate a behavioral learning curve from binary trial outcomes."""
import argparse
import numpy as np
from math import erf, exp, sqrt


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _filter(y, sig2, mu, x0, v0):
    K = len(y)
    xp = np.empty(K); vp = np.empty(K)      # one-step predictions
    xf = np.empty(K); vf = np.empty(K)      # filtered
    xprev, vprev = x0, v0
    for k in range(K):
        xp[k] = xprev
        vp[k] = vprev + sig2
        # posterior mode: x = xp + vp * (y - p(x)), Newton on that fixed point
        x = xp[k]
        for _ in range(25):
            p = 1.0 / (1.0 + exp(-(mu + x)))
            g = x - xp[k] - vp[k] * (y[k] - p)
            dx = g / (1.0 + vp[k] * p * (1 - p))
            x -= dx
            if abs(dx) < 1e-8:
                break
        p = 1.0 / (1.0 + exp(-(mu + x)))
        xf[k] = x
        vf[k] = 1.0 / (1.0 / vp[k] + p * (1 - p))
        xprev, vprev = xf[k], vf[k]
    return xp, vp, xf, vf


def _smooth(xp, vp, xf, vf):
    K = len(xf)
    xs = xf.copy(); vs = vf.copy()
    A = np.zeros(K)
    for k in range(K - 2, -1, -1):
        A[k] = vf[k] / vp[k + 1]
        xs[k] = xf[k] + A[k] * (xs[k + 1] - xp[k + 1])
        vs[k] = vf[k] + A[k] ** 2 * (vs[k + 1] - vp[k + 1])
    cov = A[:-1] * vs[1:]       # cov(x_k, x_k+1 | all data)
    return xs, vs, cov


def fit(y, chance=0.5, sig2_init=0.05, start_at_chance=True, max_iter=150, tol=1e-4):
    """
    y : 0/1 array in trial order.
    start_at_chance : pin the state before trial 1 at chance (right for a brand
        new stimulus pair; turn off for blocks where the animal may already
        know something, e.g. a reversal).

    Returns dict with p (median curve), lo / hi (5th / 95th pct), x, v, sig2,
    p_above (posterior prob. that p > chance on each trial) and learning_trial
    (0-based, None if never).
    """
    y = np.asarray(y, float)
    K = len(y)
    mu = np.log(chance / (1 - chance))
    sig2 = sig2_init
    x0 = 0.0
    for it in range(max_iter):
        v0 = 0.0 if start_at_chance else sig2
        xp, vp, xf, vf = _filter(y, sig2, mu, x0, v0)
        xs, vs, cov = _smooth(xp, vp, xf, vf)
        # E[(x_k - x_{k-1})^2] summed over trials, with x_0 as the pre-session state
        W = vs + xs ** 2
        Wc = cov + xs[:-1] * xs[1:]
        if start_at_chance:
            first = W[0] - 2 * xs[0] * x0 + x0 ** 2
        else:
            # x_0 unknown: take it equal to the smoothed first state (Smith's convention)
            x0 = xs[0]
            first = vs[0]
        new = (first + np.sum(W[1:] + W[:-1] - 2 * Wc)) / K
        new = max(new, 1e-4)
        done = abs(new - sig2) < tol * sig2
        sig2 = new
        if done:
            break

    sd = np.sqrt(vs)
    p = _logistic(mu + xs)
    lo = _logistic(mu + xs - 1.645 * sd)
    hi = _logistic(mu + xs + 1.645 * sd)
    p_above = np.array([0.5 * (1 + erf(m / (s * sqrt(2)))) for m, s in zip(xs, sd)])

    # learning trial: first k with p_above >= .95 from k to the end
    ok = p_above >= 0.95
    lt = None
    if ok[-1]:
        bad = np.where(~ok)[0]
        lt = 0 if len(bad) == 0 else int(bad[-1] + 1)
    return dict(p=p, lo=lo, hi=hi, x=xs, v=vs, sig2=sig2, p_above=p_above,
                learning_trial=lt, n_iter=it + 1)


def criterion_trial(y, n_correct=8, window=10):
    """Old-style criterion (first trial ending a window with >= n_correct hits), for comparison."""
    y = np.asarray(y, int)
    c = np.convolve(y, np.ones(window, int), mode='valid')
    hit = np.where(c >= n_correct)[0]
    return int(hit[0] + window - 1) if len(hit) else None


def _demo(seed=3, show=True, save=None):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    K = 110
    ptrue = 0.5 + 0.47 / (1 + np.exp(-(np.arange(K) - 38) / 7.0))
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for a in ax:
        y = (rng.random(K) < ptrue).astype(int)
        r = fit(y)
        t = np.arange(1, K + 1)
        a.fill_between(t, r['lo'], r['hi'], color='#2a78d6', alpha=0.2, lw=0)
        a.plot(t, r['p'], color='#2a78d6', lw=2, label='state-space estimate')
        a.plot(t, ptrue, color='#52514e', lw=1.2, ls='--', label='true p(correct)')
        a.plot(t[y == 1], np.full(y.sum(), 1.04), '|', color='#52514e', ms=4)
        a.plot(t[y == 0], np.full((1 - y).sum(), -0.04), '|', color='#e34948', ms=4)
        a.axhline(0.5, color='#c3c2b7', lw=0.8)
        lt, ct = r['learning_trial'], criterion_trial(y)
        if lt is not None:
            a.axvline(lt + 1, color='#2a78d6', lw=1)
        if ct is not None:
            a.axvline(ct + 1, color='#eb6834', lw=1, ls=':')
        a.set_title(f"learning trial {None if lt is None else lt + 1}   8/10 criterion {None if ct is None else ct + 1}",
                    fontsize=9)
        a.set_xlabel('trial')
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel('p(correct)')
    ax[0].legend(frameon=False, fontsize=8, loc='center right')
    fig.suptitle('same underlying learner, three simulated sessions', fontsize=10)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160)
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--demo', action='store_true')
    ap.add_argument('--csv')
    ap.add_argument('--col', default='correct')
    ap.add_argument('--chance', type=float, default=0.5)
    ap.add_argument('--free-start', action='store_true', help="don't pin the pre-session state at chance")
    ap.add_argument('--out')
    ap.add_argument('--save-fig')
    a = ap.parse_args()

    if a.demo or not a.csv:
        _demo(save=a.save_fig, show=a.save_fig is None)
    else:
        import csv
        with open(a.csv) as f:
            y = [int(float(row[a.col])) for row in csv.DictReader(f)]
        r = fit(y, chance=a.chance, start_at_chance=not a.free_start)
        lt = r['learning_trial']
        print(f"{len(y)} trials, {int(np.sum(y))} correct, sigma^2 = {r['sig2']:.4f}")
        print('learning trial:', 'none' if lt is None else lt + 1)
        if a.out:
            np.savetxt(a.out, np.c_[np.arange(1, len(y) + 1), y, r['p'], r['lo'], r['hi'], r['p_above']],
                       delimiter=',', header='trial,correct,p,lo,hi,p_above_chance', comments='', fmt='%.5g')
