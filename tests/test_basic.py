"""Basic checks for session handling and the analysis helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcvm import decoding as dc, glm, simulate, sorting
from tools import crossfit_timecourse as cf, learning_curve as lc, outcome_tests as ot


def test_learning_curve_tracks_truth():
    rng = np.random.default_rng(0)
    p = 0.5 + 0.45 / (1 + np.exp(-(np.arange(120) - 40) / 8))
    err = [np.abs(lc.fit((rng.random(120) < p).astype(int))['p'] - p).mean() for _ in range(15)]
    assert np.mean(err) < 0.07


def test_no_learning_trial_at_chance():
    rng = np.random.default_rng(1)
    found = [lc.fit((rng.random(100) < 0.5).astype(int))['learning_trial'] is not None for _ in range(40)]
    assert np.mean(found) < 0.15


def test_local_test_on_drifting_null_and_on_signal():
    rng = np.random.default_rng(2)
    p_null, p_sig = [], []
    for i in range(120):
        stim, ch, tg, out, pc, eng = ot.sim_behaviour(rng, 120)
        S = ot.make_surrogates(out, B=199, seed=i)
        prev = ot.lagged(out, 1)[0]
        p_null.append(ot.test_all(ot.sim_counts(rng, prev, pc, eng, **ot.SCENARIOS['slow drift']), S)['local'])
        p_sig.append(ot.test_all(ot.sim_counts(rng, prev, pc, eng, drift_sd=(.2, .4), phi=(.8, .9), effect=1.2), S)['local'])
    assert np.mean(np.array(p_null) <= 0.05) < 0.12
    assert np.mean(np.array(p_sig) <= 0.05) > 0.6


def test_crossfit_is_zero_without_signal():
    rng = np.random.default_rng(3)
    nv, cv = zip(*[cf.summary(cf.timecourse(*cf._sim_cell(rng, 0.0)[:2])) for _ in range(150)])
    assert np.nanmean(nv) > 0.3            # choosing the window on the same trials inflates it
    assert abs(np.nanmean(cv)) < 0.12


def test_sorter_on_a_clean_cell():
    for seed in range(20):
        raw = simulate.simulate_session(seed, 'x')
        if raw['truth']['snr'] > 6 and not raw['truth']['lost']:
            break
    r = sorting.sort_session(raw)
    sc = sorting.score_against_truth(r['label'], raw['true_label'])
    assert min(sc.values()) > 0.95
    assert r['qc']['pause_ratio'] < 0.1


def test_pseudo_trials_keep_train_and_test_apart():
    rng = np.random.default_rng(4)
    # feature = trial id, so any overlap between train and test rows would show
    cells = [(np.arange(20, dtype=float)[:, None] + 100 * c, np.r_[np.zeros(10, int), np.ones(10, int)]) for c in range(5)]
    for x, lab in cells:
        tr, te = dc._split(lab, rng, 0.25, 4)
        assert not set(np.concatenate(tr)) & set(np.concatenate(te))


def test_glm_recovers_coefficients():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(4000, 3))
    b = np.array([0.4, -0.3, 0.0])
    y = rng.poisson(np.exp(0.5 + X @ b))
    b0, bh = glm.fit(X, y.astype(float), lam=1e-3)
    assert abs(b0 - 0.5) < 0.05 and np.abs(bh - b).max() < 0.05
