"""Generate synthetic trial behavior, spikes and waveforms for method validation.

The simulated recordings have known spike identities and outcome effects; they are not experimentally measured cells."""
import numpy as np

from tools.outcome_tests import sim_behaviour, sim_abrupt, ar1

FS = 20000
SNIP = 48                      # samples, 2.4 ms
T_FIX, T_SYM, ITI = 0.635, 0.800, 0.700     # s: start->fixation, fixation->symbol, movement->next start


def _bump(t, c, sd):
    return np.exp(-0.5 * ((t - c) / sd) ** 2)


def draw_cell(rng, monkey):
    """Tuning parameters of one cell (the ground truth that gets saved with the session)."""
    coding = rng.random() < 0.65
    p = dict(
        monkey=monkey,
        rate=float(np.clip(np.exp(rng.normal(np.log(45), 0.55)), 8, 130)),
        gamma_k=float(rng.uniform(2.5, 6)),
        a_fix=float(rng.normal(0, 0.15)), a_sym=float(rng.normal(0, 0.2)),
        a_mov=float(abs(rng.normal(0.35, 0.2)) * (1 if rng.random() < 0.83 else -1)),
        coding=bool(coding),
        # epoch centre in ms from symbol onset; epochs tile the trial in the data
        ep_c=float(rng.uniform(-1250, 550)), ep_sd=float(rng.uniform(60, 140)),
        ep_amp=float(rng.uniform(0.3, 0.75)) if coding else 0.0,       # guess
        ep_sign=int(rng.choice([-1, 1])),                              # +1: fires more after errors
        hand_amp=float(rng.choice([-1, 1]) * rng.uniform(0.12, 0.4)) if rng.random() < 0.4 else 0.0,
        drift_sd=float(rng.uniform(0.15, 0.5)), drift_phi=float(rng.uniform(0.6, 0.95)),
        switch_jump=float(rng.normal(0, 0.25)),
        learn_couple=float(rng.choice([-1, 1]) * rng.uniform(0.3, 0.9)) if rng.random() < 0.5 else 0.0,
        rate_step=float(rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)) if rng.random() < 0.15 else 0.0,
        cs_rate=float(rng.uniform(0.6, 1.5)), cs_pfail=float(rng.uniform(1.5, 5.0)),
        pause=float(rng.uniform(0.010, 0.025)),
        snr=float(rng.uniform(5, 12)), lost=False,
    )
    r = rng.random()
    if r < 0.06:
        p['snr'] = float(rng.uniform(2.2, 3.2))        # poorly isolated
    elif r < 0.11:
        p['lost'] = True                               # cell lost partway through
    return p


def make_trials(rng):
    n_ot = int(rng.integers(25, 50))
    n_l = int(rng.integers(60, 200))
    stim_ot = rng.integers(0, 2, n_ot)
    out_ot = (rng.random(n_ot) < 0.96).astype(int)
    ch_ot = np.where(out_ot == 1, stim_ot, 1 - stim_ot)
    abrupt = rng.random() < 0.25
    stim, ch, tg, out, pc, eng = sim_abrupt(rng, n_l) if abrupt else sim_behaviour(rng, n_l)
    tr = dict(
        block=np.r_[np.full(n_ot, 'OT'), np.full(n_l, 'L')],
        stim=np.r_[stim_ot, stim], choice=np.r_[ch_ot, ch], correct=np.r_[out_ot, out],
        target=np.r_[np.where(stim_ot == 1, 1.0, -1.0), tg],
        pc=np.r_[np.full(n_ot, 0.96), pc],
    )
    n = n_ot + n_l
    # reaction time: ~380 ms when overtrained, slower while unsure (guess at the size)
    rt = 0.38 + 0.25 * (0.96 - tr['pc']) + rng.normal(0, 0.05, n)
    rt = np.clip(rt, 0.22, 1.2)
    t0 = np.empty(n)
    t = 1.0
    for i in range(n):
        t0[i] = t
        t += T_FIX + T_SYM + rt[i] + ITI
    tr.update(rt=rt, t_start=t0, t_fix=t0 + T_FIX, t_sym=t0 + T_FIX + T_SYM, t_move=t0 + T_FIX + T_SYM + rt)
    return tr, t + 1.0, abrupt


def ss_lograte(rng, p, tr, T):
    """log rate on a 1 ms grid for the whole session."""
    n = len(tr['stim'])
    nt = int(T * 1000)
    grid = np.arange(nt) / 1000.0
    lr = np.full(nt, np.log(p['rate']))

    learn = tr['block'] == 'L'
    pc = tr['pc']
    prev = np.r_[1, tr['correct'][:-1]]
    slow = ar1(rng, n, p['drift_sd'], p['drift_phi'])
    slow[learn] += p['switch_jump'] + p['learn_couple'] * (1 - pc[learn] - 0.25)
    if p['rate_step']:
        slow[rng.integers(n // 4, 3 * n // 4):] += p['rate_step']
    gate = np.where(learn, np.clip((0.97 - pc) / 0.47, 0, 1), 0.0)
    hand = np.where(tr['choice'] == 1, 0.5, -0.5)
    hand_gain = np.where(learn, np.clip((pc - 0.5) / 0.45, 0, 1), 1.0)

    edges = np.r_[np.searchsorted(grid, tr['t_start']), nt]
    for i in range(n):
        a, b = edges[i], edges[i + 1]
        # let the kernels of this trial run a little into the next one
        b2 = min(nt, b + 600)
        a2 = max(0, a - 200)
        t = (grid[a2:b2] - tr['t_sym'][i]) * 1000.0          # ms from symbol
        tm = (grid[a2:b2] - tr['t_move'][i]) * 1000.0
        k = (p['a_fix'] * _bump(t, -T_SYM * 1000 + 120, 60) + p['a_sym'] * _bump(t, 130, 70)
             + p['a_mov'] * _bump(tm, -20, 90)
             + p['hand_amp'] * hand[i] * hand_gain[i] * _bump(tm, -80, 100))
        if p['coding']:
            k = k + p['ep_sign'] * p['ep_amp'] * gate[i] * (0.5 - prev[i]) * _bump(t, p['ep_c'], p['ep_sd'])
        lr[a2:b2] += k
        lr[a:b] += slow[i]
    if p['lost']:
        cut = rng.uniform(0.55, 0.8) * T
        lr[grid > cut] -= 3.0
    return lr


def gamma_spikes(rng, lr, k):
    """Inhomogeneous gamma renewal process by time rescaling (regular firing, like simple spikes)."""
    Lam = np.cumsum(np.exp(lr)) / 1000.0
    n = int(Lam[-1] * 1.3 + 100)
    u = np.cumsum(rng.gamma(k, 1.0 / k, n))
    u = u[u < Lam[-1]]
    t = np.interp(u, Lam, np.arange(1, len(Lam) + 1) / 1000.0)
    return t[np.r_[True, np.diff(t) > 0.002]]          # absolute refractory period


def cs_times(rng, p, tr, T):
    nt = int(T * 1000)
    grid = np.arange(nt) / 1000.0
    rate = np.full(nt, p['cs_rate'])
    for i in range(len(tr['stim'])):
        a = np.searchsorted(grid, tr['t_start'][i])
        b = min(nt, a + 3200)
        t = grid[a:b]
        pfail = 1 - tr['pc'][i]
        rate[a:b] += (2.0 * _bump(t, tr['t_fix'][i] + 0.10, 0.025)
                      + (0.5 + p['cs_pfail'] * pfail) * _bump(t, tr['t_sym'][i] + 0.10, 0.025)
                      + (0.5 + p['cs_pfail'] * pfail) * _bump(t, tr['t_move'][i] + 0.10, 0.025))
    cand = grid[rng.random(nt) < rate / 1000.0] + rng.random() * 1e-3
    keep = [cand[0]] if len(cand) else []
    for c in cand[1:]:
        if c - keep[-1] > 0.06:
            keep.append(c)
    return np.array(keep)


# waveforms -------------------------------------------------------------

def _templates():
    t = np.arange(SNIP) / FS * 1000.0      # ms, threshold crossing near 0.3 ms
    ss = -_bump(t, 0.32, 0.07) + 0.38 * _bump(t, 0.62, 0.16)
    cs0 = -1.25 * _bump(t, 0.32, 0.08) + 0.5 * _bump(t, 0.58, 0.12) + 0.4 * _bump(t, 1.6, 0.6)
    other = -0.42 * _bump(t, 0.30, 0.11) + 0.2 * _bump(t, 0.75, 0.22)
    return t, ss, cs0, other


def snippets(rng, labels, tspk, T, snr, amp=400.0):
    """One snippet per event.  labels: 0 simple spike, 1 complex spike, 2 other unit."""
    t, ss, cs0, other = _templates()
    n = len(labels)
    # slow change of spike amplitude over the session
    gain = 1 + 0.12 * np.sin(2 * np.pi * (tspk / T) * rng.uniform(0.3, 1.0) + rng.uniform(0, 6))
    gain *= 1 + rng.normal(0, 0.05, n)
    w = np.zeros((n, SNIP))
    w[labels == 0] = ss
    w[labels == 2] = other
    ics = np.where(labels == 1)[0]
    for i in ics:
        x = cs0.copy()
        for j in range(rng.integers(2, 5)):          # spikelets
            x += -rng.uniform(0.3, 0.7) * _bump(t, 0.85 + 0.42 * j + rng.normal(0, 0.04), 0.07)
        w[i] = x
    w = w * gain[:, None] * amp + rng.normal(0, amp / snr, (n, SNIP))
    return np.clip(np.round(w), -32000, 32000).astype(np.int16)


def simulate_session(seed, name, monkey='B'):
    rng = np.random.default_rng(seed)
    p = draw_cell(rng, monkey)
    tr, T, abrupt = make_trials(rng)
    p['abrupt'] = bool(abrupt)

    cs = cs_times(rng, p, tr, T)
    ss = gamma_spikes(rng, ss_lograte(rng, p, tr, T), p['gamma_k'])
    # complex spike pause
    j = np.searchsorted(cs, ss) - 1
    since = np.where(j >= 0, ss - cs[np.maximum(j, 0)], np.inf)
    ss = ss[since > p['pause'] * rng.uniform(0.8, 1.3, len(ss))]
    # neighbouring unit, unrelated to the task
    other = np.sort(rng.uniform(0, T, rng.poisson(rng.uniform(3, 12) * T)))

    tspk = np.r_[ss, cs, other]
    lab = np.r_[np.zeros(len(ss), int), np.ones(len(cs), int), np.full(len(other), 2)]
    o = np.argsort(tspk)
    tspk, lab = tspk[o], lab[o]
    # a crossing inside another event's snippet would not be stored separately
    keep = np.r_[True, np.diff(tspk) > 0.0012]
    tspk, lab = tspk[keep], lab[keep]

    return dict(name=name, fs=FS, t=tspk, snips=snippets(rng, lab, tspk, T, p['snr']),
                true_label=lab.astype(np.int8), trials=tr, truth=p, duration=T)
