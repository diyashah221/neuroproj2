"""Poisson encoding models and held-out comparisons of model terms."""
import numpy as np

BIN = 50      # ms
GROUPS = ('time', 'prev', 'hand', 'slow')


def raised_cosines(n_t, n_basis):
    """(n_t, n_basis) overlapping raised-cosine bumps tiling the window."""
    c = np.linspace(0, n_t - 1, n_basis)
    w = (c[1] - c[0]) * 2
    x = (np.arange(n_t)[:, None] - c[None]) / w * 2 * np.pi
    return np.where(np.abs(x) < np.pi, 0.5 * (1 + np.cos(x)), 0.0)


def bin_raster(raster, bin_ms=BIN):
    n, T = raster.shape
    nb = T // bin_ms
    return raster[:, :nb * bin_ms].reshape(n, nb, bin_ms).sum(2)


def design(sess, idx, n_basis=10, slow_every=10):
    """
    Design for trials `idx` of a session.  Returns X (rows = trial x bin), y,
    the trial index of each row, column slices per group, and the time basis.
    """
    y = bin_raster(sess.sym[idx])
    n, nb = y.shape
    B = raised_cosines(nb, n_basis)
    prev = 0.5 - sess.prev_correct()[idx]                  # +0.5 after an error
    hand = sess.trials.choice.values[idx] - 0.5
    unsure = 1 - sess.p_hat[idx]
    unsure = unsure - unsure.mean()
    # drift: one bump every `slow_every` trials.  A cubic in trial number was not enough;
    # boosted trees given the trial number beat the GLM badly until this was made flexible.
    slow = np.c_[raised_cosines(n, max(4, n // slow_every)), unsure]

    def scaled(v):
        return (v[:, None, None] * B[None]).reshape(n * nb, -1)

    blocks = dict(time=np.tile(B, (n, 1)),
                  prev=np.c_[scaled(prev), scaled(prev * unsure * 4)],
                  hand=scaled(hand),
                  slow=np.repeat(slow, nb, axis=0))
    X, sl, k = [], {}, 0
    for g in GROUPS:
        X.append(blocks[g]); sl[g] = slice(k, k + blocks[g].shape[1]); k += blocks[g].shape[1]
    return np.hstack(X), y.ravel().astype(float), np.repeat(np.arange(n), nb), sl, B


def fit(X, y, lam=1.0, n_iter=40, tol=1e-7):
    """Ridge-penalised Poisson regression with log link (intercept free).  Returns b0, beta."""
    n, k = X.shape
    b0 = np.log(max(y.mean(), 1e-6)); beta = np.zeros(k)
    Xa = np.c_[np.ones(n), X]
    pen = np.r_[0.0, np.full(k, lam)]
    th = np.r_[b0, beta]

    def obj(th):
        eta = np.clip(Xa @ th, -20, 20)
        return np.sum(np.exp(eta) - y * eta) + 0.5 * np.sum(pen * th ** 2)

    f = obj(th)
    for _ in range(n_iter):
        mu = np.exp(np.clip(Xa @ th, -20, 20))
        g = Xa.T @ (mu - y) + pen * th
        H = (Xa * mu[:, None]).T @ Xa + np.diag(pen + 1e-8)
        step = np.linalg.solve(H, g)
        s = 1.0
        while s > 1e-4:
            new = th - s * step
            fn = obj(new)
            if fn <= f:
                break
            s /= 2
        done = f - fn < tol * abs(f)
        th, f = new, fn
        if done:
            break
    return th[0], th[1:]


def deviance(y, mu):
    mu = np.maximum(mu, 1e-10)
    t = np.where(y > 0, y * np.log(np.maximum(y, 1e-10) / mu), 0.0)
    return 2 * np.sum(t - (y - mu))


def folds_by_trial(trial, n_folds=5, run=6):
    """Fold of each row: runs of `run` consecutive trials dealt out in turn."""
    return (trial // run) % n_folds


def cv_r2(X, y, fold, lam, cols=None):
    """Held-out fraction of deviance explained, relative to a constant rate."""
    if cols is not None:
        X = X[:, cols]
    d_model = d_null = 0.0
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        b0, b = fit(X[tr], y[tr], lam)
        d_model += deviance(y[te], np.exp(np.clip(b0 + X[te] @ b, -20, 20)))
        d_null += deviance(y[te], np.full(te.sum(), y[tr].mean()))
    return 1 - d_model / d_null


def analyse_cell(sess, idx, lams=(0.3, 3.0, 30.0), slow_grid=(6, 10, 16)):
    """
    Full model and the drop from removing each group.  The ridge penalty and the
    resolution of the drift term are chosen per cell on held-out trials.
    Returns a dict with cross-validated pseudo-R2 values and the fitted
    previous-outcome kernel (log rate, error minus reward, per 50 ms bin).
    """
    best = None
    for se in slow_grid:
        X, y, trial, sl, B = design(sess, idx, slow_every=se)
        fold = folds_by_trial(trial)
        for lam in lams:
            r2 = cv_r2(X, y, fold, lam)
            if best is None or r2 > best[0]:
                best = (r2, lam, se)
    r2, lam, se = best
    X, y, trial, sl, B = design(sess, idx, slow_every=se)
    fold = folds_by_trial(trial)
    res = dict(lam=lam, slow_every=se, r2_full=r2)
    allc = np.arange(X.shape[1])
    for g in ('prev', 'hand', 'slow'):
        keep = np.setdiff1d(allc, allc[sl[g]])
        res['d_' + g] = r2 - cv_r2(X, y, fold, lam, keep)
    b0, b = fit(X, y, lam)
    nb = B.shape[1]
    bp = b[sl['prev']]
    res['prev_kernel'] = B @ bp[:nb]
    res['prev_kernel_x_unsure'] = B @ bp[nb:]
    res['hand_kernel'] = B @ b[sl['hand']]
    res['X'], res['y'], res['fold'], res['trial'] = X, y, fold, trial
    return res
