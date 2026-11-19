"""Population decoding and cross-validated neural distances."""
import numpy as np


def local_centre(x, run=8):
    """Subtract the mean of each run of consecutive trials (rows)."""
    x = np.asarray(x, float)
    lab = np.arange(len(x)) // run
    if len(x) % run and len(x) % run < run / 2 and lab.max() > 0:
        lab[lab == lab.max()] -= 1
    out = x.copy()
    for b in range(lab.max() + 1):
        out[lab == b] -= x[lab == b].mean(0)
    return out


def shuffle_within_runs(y, rng, run=8):
    y = np.asarray(y).copy()
    for a in range(0, len(y), run):
        y[a:a + run] = rng.permutation(y[a:a + run])
    return y


def _lda_fit(X, y, shrink=0.3):
    m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
    Xc = np.r_[X[y == 0] - m0, X[y == 1] - m1]
    S = Xc.T @ Xc / len(Xc)
    S = (1 - shrink) * S + shrink * np.trace(S) / len(S) * np.eye(len(S))
    w = np.linalg.solve(S, m1 - m0)
    return w, -w @ (m0 + m1) / 2


def _split(lab, rng, test_frac, min_per_class):
    tr, te = [], []
    for k in (0, 1):
        j = rng.permutation(np.where(lab == k)[0])
        n_te = max(1, int(round(test_frac * len(j))))
        if len(j) - n_te < min_per_class - 1:
            n_te = 1
        te.append(j[:n_te]); tr.append(j[n_te:])
    return tr, te


def pseudo_trials(cells, rng, n_pseudo=150, test_frac=0.25, min_per_class=4):
    """
    cells : list of (x, lab), x (n_trials, W) features of one cell, lab (n_trials,) in {0, 1}
    Returns Xtr, ytr, Xte, yte with X of shape (2 * n_pseudo, n_cells, W), z-scored
    per cell and window with training statistics.
    """
    W = cells[0][0].shape[1]
    Xtr = np.empty((2 * n_pseudo, len(cells), W)); Xte = np.empty_like(Xtr)
    for c, (x, lab) in enumerate(cells):
        tr, te = _split(lab, rng, test_frac, min_per_class)
        for k in (0, 1):
            rows = slice(k * n_pseudo, (k + 1) * n_pseudo)
            Xtr[rows, c] = x[rng.choice(tr[k], n_pseudo)]
            Xte[rows, c] = x[rng.choice(te[k], n_pseudo)]
        mu, sd = Xtr[:, c].mean(0), Xtr[:, c].std(0)
        sd[sd == 0] = 1
        Xtr[:, c] = (Xtr[:, c] - mu) / sd
        Xte[:, c] = (Xte[:, c] - mu) / sd
    y = np.r_[np.zeros(n_pseudo, int), np.ones(n_pseudo, int)]
    return Xtr, y, Xte, y


def decode(cells, n_rep=20, seed=0, generalise=False, **kw):
    """
    Accuracy per window, averaged over resamplings of the train/test split and
    the pseudo-trials.  With generalise=True also returns the (train window,
    test window) matrix.
    """
    rng = np.random.default_rng(seed)
    W = cells[0][0].shape[1]
    acc = np.zeros(W); G = np.zeros((W, W))
    for _ in range(n_rep):
        Xtr, ytr, Xte, yte = pseudo_trials(cells, rng, **kw)
        for w in range(W):
            wt, b = _lda_fit(Xtr[:, :, w], ytr)
            if generalise:
                pred = (np.einsum('ncw,c->nw', Xte, wt) + b) > 0
                G[w] += (pred == yte[:, None]).mean(0)
                acc[w] += (pred[:, w] == yte).mean()
            else:
                acc[w] += (((Xte[:, :, w] @ wt + b) > 0) == yte).mean()
    return (acc / n_rep, G / n_rep) if generalise else acc / n_rep


def null_trial_level(cells, n_null=100, n_rep=3, seed=1, local=True, **kw):
    """Accuracy with each cell's recorded-trial labels permuted (within runs if local), rebuilt from scratch."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_null):
        sh = []
        for x, lab in cells:
            for _ in range(20):
                l2 = shuffle_within_runs(lab, rng) if local else rng.permutation(lab)
                if min(np.sum(l2 == 0), np.sum(l2 == 1)) >= 2:
                    break
            sh.append((x, l2))
        out.append(decode(sh, n_rep=n_rep, seed=int(rng.integers(1 << 30)), **kw))
    return np.array(out)


def null_pseudo_level(cells, w, n_null=100, seed=2, **kw):
    """The shortcut: build pseudo-trials once from the true labels, then shuffle labels among them."""
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = pseudo_trials(cells, rng, **kw)
    out = []
    for _ in range(n_null):
        ys, yt = rng.permutation(ytr), rng.permutation(yte)
        wt, b = _lda_fit(Xtr[:, :, w], ys)
        out.append((((Xte[:, :, w] @ wt + b) > 0) == yt).mean())
    return np.array(out)


def compare_classifiers(cells, w, n_rep=10, seed=0, **kw):
    """Same pseudo-trials, different classifiers, one window."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    rng = np.random.default_rng(seed)
    models = {
        'LDA (shrinkage)': None,
        'logistic (L2)': lambda: LogisticRegression(C=0.1, max_iter=500),
        'linear SVM': lambda: LinearSVC(C=0.01),
        'boosted trees': lambda: HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, learning_rate=0.1),
    }
    acc = {k: [] for k in models}
    for _ in range(n_rep):
        Xtr, ytr, Xte, yte = pseudo_trials(cells, rng, **kw)
        A, B = Xtr[:, :, w], Xte[:, :, w]
        for k, mk in models.items():
            if mk is None:
                wt, b = _lda_fit(A, ytr)
                acc[k].append((((B @ wt + b) > 0) == yte).mean())
            else:
                acc[k].append(mk().fit(A, ytr).score(B, yte))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cv_distance(cells, n_split=50, seed=0):
    """
    Cross-validated squared distance between the two conditions, averaged over
    cells, per window.  Each cell's trials are halved at random; the class
    difference from one half is multiplied by the difference from the other, so
    noise averages to zero instead of adding a positive floor (the idea behind
    the cross-validated Mahalanobis distance, Walther et al. 2016).
    Units: squared within-class sd.  Returns (n_cells, W).
    """
    rng = np.random.default_rng(seed)
    out = []
    for x, lab in cells:
        a, b = x[lab == 0], x[lab == 1]
        sd = np.sqrt((a.var(0, ddof=1) * (len(a) - 1) + b.var(0, ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
        sd[sd == 0] = 1
        acc = np.zeros(x.shape[1])
        for _ in range(n_split):
            ia, ib = rng.permutation(len(a)), rng.permutation(len(b))
            ha, hb = len(a) // 2, len(b) // 2
            d1 = a[ia[:ha]].mean(0) - b[ib[:hb]].mean(0)
            d2 = a[ia[ha:]].mean(0) - b[ib[hb:]].mean(0)
            acc += d1 * d2 / sd ** 2
        out.append(acc / n_split)
    return np.array(out)
