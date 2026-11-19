"""Outcome-selectivity tests with drift-aware null distributions.

The same statistic is evaluated against several controls, including future-trial outcomes."""
import argparse
import numpy as np
from scipy import optimize, stats


# ---------------------------------------------------------------- statistic

def trend_basis(n, deg=3):
    """Intercept + Legendre polynomials in trial number (slow-trend nuisance)."""
    return np.polynomial.legendre.legvander(np.linspace(-1, 1, n), deg)


def block_basis(n, block=8):
    """One dummy per run of `block` consecutive trials (a short tail joins the last run)."""
    lab = block_labels(n, block)
    return (lab[:, None] == np.arange(lab.max() + 1)[None]).astype(float)


def block_labels(n, block=8):
    lab = np.arange(n) // block
    if n % block and n % block < block / 2 and lab.max() > 0:
        lab[lab == lab.max()] -= 1
    return lab


def partial_corr(y, X, Z=None):
    """
    y : (n,) or (n, W) responses,  X : (B, n) regressors,  Z : (n, k) nuisance
    (must contain the intercept; default intercept only).  Returns (B, W).
    """
    y = np.asarray(y, float)
    Y = y[None] if y.ndim == 1 else y.T
    X = np.atleast_2d(np.asarray(X, float))
    n = Y.shape[1]
    if Z is None:
        Z = np.ones((n, 1))
    Q = np.linalg.qr(Z)[0]
    Yr = Y - (Y @ Q) @ Q.T
    Xr = X - (X @ Q) @ Q.T
    ny = np.linalg.norm(Yr, axis=1)
    nx = np.linalg.norm(Xr, axis=1)
    ny[ny < 1e-9] = np.inf
    nx[nx < 1e-9] = np.inf
    return (Xr @ Yr.T) / np.outer(nx, ny)


def _stat(y, X, Z):
    return np.abs(partial_corr(y, X, Z)).max(axis=1)


def _vst(counts):
    return np.sqrt(np.asarray(counts, float))


# ---------------------------------------------------------------- surrogates

def surr_shuffle(x, B, rng):
    return rng.permuted(np.tile(np.asarray(x, np.float32), (B, 1)), axis=1)


def surr_circshift(x, B, rng, margin=5):
    n = len(x)
    s = rng.integers(margin, n - margin, B)
    return np.asarray(x, np.float32)[(np.arange(n)[None] - s[:, None]) % n]


def surr_local(x, B, rng, block=8):
    lab = block_labels(len(x), block)
    out = np.tile(np.asarray(x, np.float32), (B, 1))
    for b in range(lab.max() + 1):
        j = np.where(lab == b)[0]
        out[:, j] = rng.permuted(out[:, j], axis=1)
    return out


def lagged(out, lag=1, edge=1.0):
    """Outcome sequence(s) -> regressor.  lag=1: outcome of the previous trial, lag=-1: of the next."""
    out = np.atleast_2d(np.asarray(out, float))
    x = np.full_like(out, edge)
    if lag > 0:
        x[:, lag:] = out[:, :-lag]
    elif lag < 0:
        x[:, :lag] = out[:, -lag:]
    else:
        x = out.copy()
    return x


def surr_sessions(donors, n, rng, tail=20):
    """
    Outcome sequences from other sessions, cut to n trials.  Donors shorter
    than n are extended by resampling their last `tail` trials, i.e. assuming
    behaviour is stationary once the block is that old.
    """
    out = []
    for d in donors:
        d = np.asarray(d, float)
        if len(d) < n:
            d = np.r_[d, rng.choice(d[-tail:], n - len(d))]
        out.append(d[:n])
    return np.array(out, np.float32)


# ---------------------------------------------------------------- learning model

class Agent:
    """
    Two stimuli, two hands.  v[s] in [-1, 1] is how strongly stimulus s is
    tied to the right hand.  Feedback reveals the correct hand either way, so
    v[s] moves toward it at rate a_win after a reward and a_lose after an error.

        p(right | s) = logistic(beta * v[s] + bias + stick * last_choice)

    v0 scales the starting association: 0 for a new pair, negative if the block
    starts from the opposite mapping (reversal).
    """
    names = ('a_win', 'a_lose', 'beta', 'bias', 'stick', 'v0')
    lo = np.array([0.005, 0.005, 0.3, -2.0, -2.0, -1.0])
    hi = np.array([1.0, 1.0, 12.0, 2.0, 2.0, 1.0])

    def __init__(self, theta):
        self.theta = np.asarray(theta, float)

    def nll(self, stim, choice, target, theta=None):
        a_w, a_l, beta, bias, stick, v0 = self.theta if theta is None else theta
        v = v0 * np.array([target[stim == 0][0] if np.any(stim == 0) else 0.0,
                           target[stim == 1][0] if np.any(stim == 1) else 0.0])
        last = 0.0
        ll = 0.0
        for s, c, tg in zip(stim, choice, target):
            z = beta * v[s] + bias + stick * last
            sgn = 1.0 if c == 1 else -1.0
            ll += -np.logaddexp(0.0, -sgn * z)
            win = (sgn == tg)
            v[s] += (a_w if win else a_l) * (tg - v[s])
            last = sgn
        return -ll

    def simulate(self, stim, target, B, rng):
        """B synthetic sessions on the given stimulus sequence -> choices, outcomes (B, n)."""
        a_w, a_l, beta, bias, stick, v0 = self.theta
        n = len(stim)
        tg0 = target[stim == 0][0] if np.any(stim == 0) else 0.0
        tg1 = target[stim == 1][0] if np.any(stim == 1) else 0.0
        v = np.tile(v0 * np.array([tg0, tg1]), (B, 1))
        last = np.zeros(B)
        ch = np.empty((B, n)); out = np.empty((B, n))
        u = rng.random((B, n))
        for t in range(n):
            s, tg = stim[t], target[t]
            z = beta * v[:, s] + bias + stick * last
            sgn = np.where(u[:, t] < 1 / (1 + np.exp(-z)), 1.0, -1.0)
            win = sgn == tg
            v[:, s] += np.where(win, a_w, a_l) * (tg - v[:, s])
            last = sgn
            ch[:, t] = sgn > 0
            out[:, t] = win
        return ch, out


def fit_agent(stim, choice, target, free_start=False, n_starts=4, seed=0):
    """
    Maximum likelihood (lightly penalised) fit of Agent to one block.
    stim, choice in {0, 1};  target = +1 where the right hand is correct, -1 otherwise.
    """
    stim = np.asarray(stim, int); choice = np.asarray(choice, int)
    target = np.asarray(target, float)
    ag = Agent(np.zeros(6))
    lo, hi = Agent.lo.copy(), Agent.hi.copy()
    if not free_start:
        lo[5] = hi[5] = 0.0

    def loss(th):
        # weak pulls toward no bias / no stickiness / moderate beta; sessions are ~100 trials
        pen = 0.5 * (th[3] / 1.0) ** 2 + 0.5 * (th[4] / 1.0) ** 2 + 0.5 * ((np.log(th[2]) - np.log(3.0)) / 1.5) ** 2
        return ag.nll(stim, choice, target, th) + pen

    rng = np.random.default_rng(seed)
    best = None
    starts = [np.array([0.2, 0.2, 3.0, 0.0, 0.0, 0.0])]
    for _ in range(n_starts - 1):
        starts.append(np.array([rng.uniform(0.03, 0.6), rng.uniform(0.03, 0.6), rng.uniform(1, 8),
                                rng.normal(0, .3), rng.normal(0, .3), rng.uniform(-.8, .3) if free_start else 0.0]))
    for x0 in starts:
        r = optimize.minimize(loss, np.clip(x0, lo, hi), method='L-BFGS-B', bounds=list(zip(lo, hi)))
        if best is None or r.fun < best.fun:
            best = r
    out = Agent(best.x)
    out.nll_fit = ag.nll(stim, choice, target, best.x)
    return out


# ---------------------------------------------------------------- tests

def perm_p(y, x, X_surr, Z=None):
    """p-value of max_w |partial r| against surrogate regressors.  y is already transformed."""
    obs = _stat(y, x[None], Z)[0]
    null = _stat(y, X_surr, Z)
    return (1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1)


def naive_p(y, x, Z=None):
    """Parametric p of the partial correlation in each window."""
    r = partial_corr(y, x[None], Z)[0]
    df = len(x) - (1 if Z is None else Z.shape[1]) - 1
    t = r * np.sqrt(df / np.maximum(1 - r ** 2, 1e-12))
    return 2 * stats.t.sf(np.abs(t), df)


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def linshift_p(y, x, N=20):
    """
    Conservative linear shift (Harris 2020): keep the middle n-2N trials of x,
    slide y by -N..N against it, p = (number of shifts at least as good as no
    shift) / (N + 1).
    """
    n = len(x)
    if n - 2 * N < 20:
        N = max(3, (n - 20) // 2)
    xs = np.asarray(x, float)[N:n - N]
    st = np.array([_stat(y[N + s:n - N + s], xs[None], None)[0] for s in range(-N, N + 1)])
    return min(1.0, np.sum(st >= st[N] - 1e-12) / (N + 1))


def make_surrogates(outcome, stim=None, choice=None, target=None, donors=None, agent=None,
                    B=999, block=8, free_start=False, seed=0):
    """
    Surrogate outcome sequences of every kind for one session, (B, n) each.

    outcome : (n,) 0/1 outcomes of this block in trial order
    stim, choice, target : for the synthetic-session null (see fit_agent); pass
              `agent` to reuse a fit
    donors  : outcome sequences from other sessions, for session permutation
    """
    rng = np.random.default_rng(seed)
    o = np.asarray(outcome, float)
    S = dict(out=o, block=block)
    S['shuffle'] = surr_shuffle(o, B, rng)
    S['circshift'] = surr_circshift(o, B, rng)
    S['local'] = surr_local(o, B, rng, block)
    if donors is not None and len(donors) >= 19:
        S['session'] = surr_sessions(donors, len(o), rng)
    if stim is not None:
        if agent is None:
            agent = fit_agent(stim, choice, target, free_start=free_start)
        S['synthetic'] = agent.simulate(np.asarray(stim, int), np.asarray(target, float), B, rng)[1].astype(np.float32)
        S['agent'] = agent
    return S


def _designs(n, block):
    return trend_basis(n, 3), block_basis(n, block)


def test_all(counts, S, lag=1, alpha=0.05, run=3, edge=1.0):
    """
    counts : (n,) or (n, W) spike counts per trial (and time window)
    S      : output of make_surrogates for the same trials
    lag    : 1 tests the previous trial's outcome, -1 the next trial's.  The
             second is a negative control: an outcome that has not happened yet
             cannot drive the neuron, so anything "significant" there is drift
             or a slow state shared by behaviour and firing.

    Returns {method: p} for one neuron.  With several windows every p is for
    the neuron as a whole (max statistic over windows), except 'per-window t',
    which is the uncorrected rule "p < alpha in at least `run` consecutive
    windows", reported as 0 or 1.
    """
    y = _vst(counts)
    if y.ndim == 1:
        y = y[:, None]
    n, W = y.shape
    x = lagged(S['out'], lag, edge)[0]
    X = lambda k: lagged(S[k], lag, edge)
    Zt, Zb = _designs(n, S['block'])
    res = {}
    p = naive_p(y, x)
    res['per-window t'] = 0.0 if longest_run(p < alpha) >= min(run, W) else 1.0
    res['t, Bonferroni'] = min(1.0, p.min() * W)
    res['shuffle'] = perm_p(y, x, X('shuffle'))
    res['shuffle+trend'] = perm_p(y, x, X('shuffle'), Zt)
    res['circshift'] = perm_p(y, x, X('circshift'))
    res['linshift'] = linshift_p(y, x)
    if 'session' in S:
        res['session'] = perm_p(y, x, X('session'))
        res['session+trend'] = perm_p(y, x, X('session'), Zt)
    if 'synthetic' in S:
        res['synthetic'] = perm_p(y, x, X('synthetic'))
        res['synthetic+trend'] = perm_p(y, x, X('synthetic'), Zt)
    res['local'] = perm_p(y, x, X('local'), Zb)
    return res


def test_direction(counts, S, edge=1.0):
    """
    Is the neuron tied more tightly to the previous outcome than to the next one?
    Statistic: max_w |r(prev)| - max_w |r(next)|, one-sided, against the same
    difference computed from surrogate outcome sequences.  A slow state that
    changes both error rate and firing (arousal, engagement) loads on both lags
    and cancels; a response to feedback only loads on the past.
    """
    y = _vst(counts)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    Zt, Zb = _designs(n, S['block'])
    res = {}
    for name, key, Z in (('shuffle+trend', 'shuffle', Zt), ('session+trend', 'session', Zt),
                         ('synthetic+trend', 'synthetic', Zt), ('local', 'local', Zb)):
        if key not in S:
            continue
        d = lambda O: _stat(y, lagged(O, 1, edge), Z) - _stat(y, lagged(O, -1, edge), Z)
        obs = d(S['out'][None])[0]
        null = d(S[key])
        res[name] = (1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1)
    return res


# ---------------------------------------------------------------- simulation for calibration

def sim_behaviour(rng, n, alpha=None, beta=None, reversal=None, lapse_mean=-2.6, lapse_sd=1.0, rho=0.97):
    """
    Generating learner, deliberately not the same model as Agent: action values
    updated for the chosen hand only (slower after errors), plus lapses whose
    rate wanders slowly (attention drifting in and out).  A third of blocks
    start from the opposite mapping, like the reversal blocks in the real data.
    Ranges were set by eye so that accuracy by thirds and win-stay / lose-shift
    rates look like the 24 public sessions.

    Returns stim, choice, target, outcome, the lapse-free probability of a
    correct response on each trial, and the engagement state behind the lapses.
    """
    alpha = 10 ** rng.uniform(-1.9, -1.1) if alpha is None else alpha
    beta = rng.uniform(4, 10) if beta is None else beta
    reversal = rng.random() < 0.35 if reversal is None else reversal
    a_lose = alpha * rng.uniform(0.2, 0.8)
    bias = rng.normal(0, 0.3)
    stim = rng.integers(0, 2, n)
    right_is_correct = rng.permutation(2)
    Q = np.full((2, 2), 0.5)
    if reversal:
        for s in range(2):
            Q[s, right_is_correct[s]] = 0.25
            Q[s, 1 - right_is_correct[s]] = 0.75
    e = rng.normal(0, 1)
    choice = np.empty(n, int); out = np.empty(n, int); pc = np.empty(n); eng = np.empty(n)
    for t in range(n):
        s = stim[t]
        corr = right_is_correct[s]
        p_right = 1 / (1 + np.exp(-(beta * (Q[s, 1] - Q[s, 0]) + bias)))
        pc[t] = p_right if corr == 1 else 1 - p_right
        e = rho * e + np.sqrt(1 - rho ** 2) * rng.normal()
        eng[t] = e
        if rng.random() < 1 / (1 + np.exp(-(lapse_mean + lapse_sd * e))):
            c = rng.integers(0, 2)
        else:
            c = int(rng.random() < p_right)
        r = int(c == corr)
        Q[s, c] += (alpha if r else a_lose) * (r - Q[s, c])
        choice[t] = c; out[t] = r
    target = np.where(right_is_correct[stim] == 1, 1.0, -1.0)
    return stim, choice, target, out, pc, eng


def sim_abrupt(rng, n):
    """
    Step-like learner: performance sits near chance (or below, for a reversal)
    and jumps to a high plateau at a change point.  Individual learning curves
    often look like this (Gallistel et al. 2004, PNAS 101:13124), and several
    of the public sessions do.
    """
    stim = rng.integers(0, 2, n)
    right_is_correct = rng.permutation(2)
    cp = int(rng.uniform(12, max(14, 0.7 * n)))
    p0 = rng.choice([0.5, 0.5, 0.3])
    pc = np.where(np.arange(n) < cp, p0, rng.uniform(0.85, 0.98))
    out = (rng.random(n) < pc).astype(int)
    corr = right_is_correct[stim]
    choice = np.where(out == 1, corr, 1 - corr)
    target = np.where(corr == 1, 1.0, -1.0)
    return stim, choice, target, out, pc, np.zeros(n)


def ar1(rng, n, sd, phi):
    d = np.empty(n)
    d[0] = rng.normal(0, sd)
    for t in range(1, n):
        d[t] = phi * d[t - 1] + rng.normal(0, sd * np.sqrt(1 - phi ** 2))
    return d


SCENARIOS = {
    # drift_sd: sd of log rate across trials, phi: its lag-1 autocorrelation.
    # 'drift' spans what I measured in the public rasters (docs/methods_notes.md).
    # couple: rate follows the learning state itself, no outcome coding.
    # engage: rate follows the slow engagement state that also causes lapses.
    # effect: log rate difference after error vs after reward at chance performance.
    'stationary':        dict(drift_sd=(0, 0), phi=(0, 0)),
    'drift':             dict(drift_sd=(0.15, 0.6), phi=(0.6, 0.95)),
    'slow drift':        dict(drift_sd=(0.15, 0.6), phi=(0.97, 0.995)),
    'learning-coupled':  dict(drift_sd=(0.15, 0.6), phi=(0.6, 0.95), couple=1.0),
    'engagement-coupled': dict(drift_sd=(0.15, 0.6), phi=(0.6, 0.95), engage=0.3),
    # step: one abrupt change in rate at a random trial (electrode moved, cell
    # changed state), paired with step-like learners
    'rate step, abrupt learner': dict(drift_sd=(0.15, 0.6), phi=(0.6, 0.95), step=(0.3, 0.8), learner='abrupt'),
    'drift + coding':    dict(drift_sd=(0.15, 0.6), phi=(0.6, 0.95), effect=0.6),
}


def sim_counts(rng, prev, pc, eng, drift_sd, phi, couple=0, engage=0, effect=0, step=None,
               n_bins=38, bin_ms=50, win_bins=4, learner=None):
    """
    Counts (n_trials, W) in sliding windows (200 ms, 50 ms steps over 1.9 s).
    The outcome effect, if any, lives in one 250 ms epoch at a random time and
    fades as the association is learned.
    """
    n = len(prev)
    base = np.exp(rng.normal(np.log(35), 0.6)) * bin_ms / 1000.0
    tt = np.arange(n_bins)
    prof = sum(rng.uniform(-0.5, 0.5) * np.exp(-0.5 * ((tt - rng.uniform(0, n_bins)) / rng.uniform(2, 6)) ** 2)
               for _ in range(2))
    lr = np.zeros((n, n_bins)) + prof[None]
    sd = rng.uniform(*drift_sd)
    if sd > 0:
        lr += ar1(rng, n, sd, rng.uniform(*phi))[:, None]
    if couple:
        lr += (rng.choice([-1, 1]) * rng.uniform(0.4, 1.0) * couple * (1 - pc - 0.25))[:, None]
    if engage:
        lr += (rng.choice([-1, 1]) * engage * eng)[:, None]
    if step:
        lr[rng.integers(10, n - 10):] += rng.choice([-1, 1]) * rng.uniform(*step)
    if effect:
        gate = np.clip((0.97 - pc) / 0.47, 0, 1)
        c0 = rng.integers(0, n_bins - 5)
        lr[:, c0:c0 + 5] += (rng.choice([-1, 1]) * effect * gate * (0.5 - prev))[:, None]
    fine = rng.poisson(base * np.exp(lr))
    cs = np.cumsum(np.c_[np.zeros(n, int), fine], axis=1)
    return cs[:, win_bins:] - cs[:, :-win_bins]


def calibrate(n_sess=400, B=499, n_range=(60, 200), scenarios=None, seed=1, block=8, verbose=True,
              beh_kw=None):
    """
    Rejection rates at alpha = .05 for every method in every scenario.
    Returns res[scenario][kind][method] = array of p-values, kind in
    'prev' (the test), 'next' (negative control), 'dir' (prev vs next).
    """
    rng = np.random.default_rng(seed)
    scenarios = scenarios or list(SCENARIOS)
    kinds = sorted({SCENARIOS[sc].get('learner', 'gradual') for sc in scenarios})
    pools = {}
    for kind in kinds:
        pools[kind] = []
        for i in range(n_sess):
            n = int(rng.integers(*n_range))
            gen = sim_abrupt(rng, n) if kind == 'abrupt' else sim_behaviour(rng, n, **(beh_kw or {}))
            pools[kind].append(dict(zip(('stim', 'choice', 'target', 'out', 'pc', 'eng'), gen)))
    P = {sc: dict(prev={}, next={}, dir={}) for sc in scenarios}
    for kind in kinds:
        beh = pools[kind]
        for i, b in enumerate(beh):
            # surrogates are built once per session and reused across scenarios
            idx = rng.choice([j for j in range(n_sess) if j != i], min(200, n_sess - 1), replace=False)
            S = make_surrogates(b['out'], b['stim'], b['choice'], b['target'],
                                donors=[beh[j]['out'] for j in idx], B=B, block=block,
                                free_start=True, seed=int(rng.integers(1 << 30)))
            prev = lagged(b['out'], 1)[0]       # the simulated neuron only ever sees the past
            for sc in scenarios:
                if SCENARIOS[sc].get('learner', 'gradual') != kind:
                    continue
                counts = sim_counts(rng, prev, b['pc'], b['eng'], **SCENARIOS[sc])
                for what, r in (('prev', test_all(counts, S, 1)), ('next', test_all(counts, S, -1)),
                                ('dir', test_direction(counts, S))):
                    for k, v in r.items():
                        P[sc][what].setdefault(k, []).append(v)
            if verbose and (i + 1) % 100 == 0:
                print(f'  {kind} learners: {i + 1}/{n_sess}', flush=True)
    res = {sc: {what: {k: np.array(v) for k, v in d.items()} for what, d in P[sc].items()} for sc in scenarios}
    if verbose:
        for sc in scenarios:
            print(f'\n{sc}  ({n_sess} sessions)      prev    next    prev>next')
            for k in res[sc]['prev']:
                f = lambda what: (f"{100 * np.mean(res[sc][what][k] <= 0.05):6.1f}%" if k in res[sc][what] else '      -')
                print(f'  {k:16s}      {f("prev")} {f("next")} {f("dir")}', flush=True)
    return res


def sweep_block(blocks=(4, 6, 8, 12, 16, 24), n_sess=600, B=499, seed=2,
                scenarios=('slow drift', 'rate step, abrupt learner', 'drift + coding')):
    """Rejection rate of the local test as a function of run length."""
    rng = np.random.default_rng(seed)
    hits = {sc: {b: [] for b in blocks} for sc in scenarios}
    for sc in scenarios:
        abrupt = SCENARIOS[sc].get('learner') == 'abrupt'
        for i in range(n_sess):
            n = int(rng.integers(60, 200))
            stim, choice, target, out, pc, eng = sim_abrupt(rng, n) if abrupt else sim_behaviour(rng, n)
            x = lagged(out, 1)[0]
            y = _vst(sim_counts(rng, x, pc, eng, **SCENARIOS[sc]))
            for b in blocks:
                hits[sc][b].append(perm_p(y, x, surr_local(x, B, rng, b), block_basis(n, b)) <= 0.05)
    return {sc: {b: float(np.mean(v)) for b, v in d.items()} for sc, d in hits.items()}


def plot_calibration(res, save=None):
    import matplotlib.pyplot as plt
    res = {sc: v['prev'] for sc, v in res.items()}
    methods = list(next(iter(res.values())).keys())
    null_sc = [s for s in res if 'coding' not in s]
    pow_sc = [s for s in res if 'coding' in s]
    col = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
    mk = ['o', 's', '^', 'D', 'v', 'P']
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw=dict(width_ratios=[2.2, 1]), sharey=True)
    ypos = np.arange(len(methods))[::-1]
    for i, sc in enumerate(null_sc):
        r = [100 * np.mean(res[sc][m] <= 0.05) for m in methods]
        ax[0].plot(r, ypos + ((len(null_sc) - 1) / 2 - i) * 0.17, mk[i], color=col[i], ms=6, label=sc, mec='white', mew=0.6)
    ax[0].axvline(5, color='#52514e', lw=1)
    ax[0].text(5, len(methods) - 0.42, ' nominal 5%', fontsize=8, color='#52514e', va='bottom')
    ax[0].set_xscale('symlog', linthresh=10, linscale=1.3)
    ax[0].set_xticks([0, 2.5, 5, 10, 25, 50, 100])
    ax[0].set_xticklabels(['0', '2.5', '5', '10', '25', '50', '100'])
    ax[0].set_xlim(-0.5, 115)
    ax[0].set_yticks(ypos)
    ax[0].set_yticklabels(methods)
    ax[0].set_xlabel('false positives at alpha = .05 (% of neurons with no outcome coding)')
    ax[0].legend(frameon=False, fontsize=8, loc='lower right', title='simulated neuron', title_fontsize=8)
    for sc in pow_sc:
        r = [100 * np.mean(res[sc][m] <= 0.05) for m in methods]
        ax[1].barh(ypos, r, height=0.55, color='#c3c2b7')
        for y, v in zip(ypos, r):
            ax[1].text(v + 2, y, f'{v:.0f}', va='center', fontsize=8, color='#52514e')
    ax[1].set_xlim(0, 114)
    ax[1].set_xlabel('detected (% of neurons that do code outcome)')
    for a in ax:
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        a.grid(axis='x', color='#e1e0d9', lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=160)
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Drift-aware tests for trial-by-trial coding.')
    ap.add_argument('--demo', action='store_true', help='run the calibration simulation')
    ap.add_argument('--n-sess', type=int, default=300)
    ap.add_argument('--B', type=int, default=499)
    ap.add_argument('--block', type=int, default=8)
    ap.add_argument('--counts', help='.npy, trials x windows spike counts')
    ap.add_argument('--trials', help='csv with columns stim, choice, correct (0/1), in trial order')
    ap.add_argument('--save-fig')
    a = ap.parse_args()

    if a.counts and a.trials:
        counts = np.load(a.counts)
        tr = np.genfromtxt(a.trials, delimiter=',', names=True)
        stim = tr['stim'].astype(int); choice = tr['choice'].astype(int); corr = tr['correct'].astype(int)
        target = np.where((choice == 1) == (corr == 1), 1.0, -1.0)
        S = make_surrogates(corr, stim, choice, target, B=a.B, block=a.block, free_start=True)
        for lag, name in ((1, 'previous outcome'), (-1, 'next outcome (negative control)')):
            print(name)
            for k, v in test_all(counts, S, lag).items():
                print(f'  {k:16s} p = {v:.4f}')
        print('previous > next')
        for k, v in test_direction(counts, S).items():
            print(f'  {k:16s} p = {v:.4f}')
    else:
        res = calibrate(n_sess=a.n_sess, B=a.B, block=a.block)
        if a.save_fig:
            import matplotlib
            matplotlib.use('Agg')
            plot_calibration(res, a.save_fig)
