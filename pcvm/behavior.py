"""Behavioral learning curves and trial-history model fits."""
import numpy as np
from scipy import optimize

from tools import learning_curve as lc
from tools.outcome_tests import Agent


def learning_state(sess):
    """
    Attach to the session: p_hat (state-space estimate of p(correct) on every
    trial; overtrained trials get the block mean), learning_trial (index within
    the learning block or None) and the fit itself.
    """
    L = sess.learn
    y = sess.trials.correct.values
    fit = lc.fit(y[L], start_at_chance=not sess.meta.get('reversal', False))
    p = np.full(len(y), y[sess.block('OT')].mean() if len(sess.block('OT')) else np.nan)
    p[L] = fit['p']
    sess.p_hat, sess.lc, sess.learning_trial = p, fit, fit['learning_trial']
    return sess


def stage(p_hat, block):
    """Learning stage per trial: OT, early (p < .75) or late, the split used by Ipata et al."""
    return np.where(block == 'OT', 'OT', np.where(p_hat < 0.75, 'early', 'late'))


# model variants: map a short parameter vector onto Agent's (a_win, a_lose, beta, bias, stick, v0)
MODELS = {
    'bias only':  (lambda th: [0.005, 0.005, 1.0, th[0], 0.0, 0.0], [(-3, 3)]),
    'win-stay lose-shift': (lambda th: [1.0, 1.0, th[0], th[1], 0.0, 0.0], [(0.1, 6), (-2, 2)]),
    'one rate':   (lambda th: [th[0], th[0], th[1], th[2], 0.0, th[3]], [(0.005, 1), (0.3, 12), (-2, 2), (-1, 1)]),
    'two rates + stickiness': (lambda th: [th[0], th[1], th[2], th[3], th[4], th[5]],
                               [(0.005, 1), (0.005, 1), (0.3, 12), (-2, 2), (-2, 2), (-1, 1)]),
}


def compare_models(stim, choice, target, seed=0):
    """Max-likelihood fit of each variant to one block.  Returns {name: (nll, k, theta_full)}."""
    stim = np.asarray(stim, int); choice = np.asarray(choice, int); target = np.asarray(target, float)
    ag = Agent(np.zeros(6))
    rng = np.random.default_rng(seed)
    out = {}
    for name, (expand, bounds) in MODELS.items():
        lo, hi = np.array(bounds).T
        best = None
        for _ in range(4):
            x0 = lo + (hi - lo) * rng.uniform(0.2, 0.8, len(lo))
            r = optimize.minimize(lambda th: ag.nll(stim, choice, target, expand(th)), x0, method='L-BFGS-B',
                                  bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        out[name] = (float(best.fun), len(bounds), np.array(expand(best.x), float))
    return out


def behaviour_stats(stim, choice, correct):
    """Summary numbers used to check simulated behaviour against the animals'."""
    y = np.asarray(correct)
    n = len(y)
    ws, ls, last = [], [], {}
    for s, c, o in zip(stim, choice, y):
        if s in last:
            (ws if last[s][1] else ls).append(c == last[s][0])
        last[s] = (c, o)
    runs = np.diff(np.where(np.diff(np.r_[1, y, 1]) != 0)[0])[::2] if (y == 0).any() else np.array([0])
    return dict(first=y[:n // 3].mean(), mid=y[n // 3:2 * n // 3].mean(), last=y[2 * n // 3:].mean(),
                win_stay=np.mean(ws) if ws else np.nan, lose_shift=1 - np.mean(ls) if ls else np.nan,
                max_error_run=float(runs.max()))


def predictive_check(agent, stim, choice, correct, target, n_sim=200, seed=0):
    """Same statistics for the data and for sessions simulated from the fitted agent."""
    rng = np.random.default_rng(seed)
    obs = behaviour_stats(stim, choice, correct)
    ch, out = agent.simulate(np.asarray(stim, int), np.asarray(target, float), n_sim, rng)
    sims = [behaviour_stats(stim, c.astype(int), o.astype(int)) for c, o in zip(ch, out)]
    return obs, {k: np.array([s[k] for s in sims]) for k in obs}
