"""Separate candidate simple and complex spikes from waveform snippets.

Waveform clusters are checked against the expected post-complex-spike pause before a session is accepted."""
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def _split2(f, init_hi, seed=0):
    """Two-component mixture with one component started on the rows in `init_hi`."""
    m0 = np.vstack([np.median(f[~init_hi], 0), np.median(f[init_hi], 0)])
    gm = GaussianMixture(2, covariance_type='full', means_init=m0, random_state=seed, reg_covar=1e-4).fit(f)
    return gm.predict(f) == 1


def _rare_cluster(z, p_out=1e-4, min_n=15):
    """
    Complex spikes are ~1-2% of events, too few for a mixture model to find on
    its own.  Treat the bulk as one Gaussian (fitted robustly), take the events
    it cannot explain, and if they agree with each other assign every event to
    the bulk or to that group by likelihood.
    """
    med = np.median(z, 0)
    d0 = np.sum(((z - med) / (1.4826 * np.median(np.abs(z - med), 0))) ** 2, 1)
    core = d0 < np.quantile(d0, 0.9)
    mu, C = z[core].mean(0), np.cov(z[core].T) * 1.3        # trimmed covariance, roughly re-inflated
    d = np.einsum('ij,jk,ik->i', z - mu, np.linalg.inv(C), z - mu)
    out = d > chi2.ppf(1 - p_out, z.shape[1])
    if out.sum() < min_n:
        return np.zeros(len(z), bool)
    for _ in range(3):
        mu1, C1 = z[out].mean(0), np.cov(z[out].T) + 1e-6 * np.eye(z.shape[1])
        mu0, C0 = z[~out].mean(0), np.cov(z[~out].T)
        ll = lambda m, c: -0.5 * (np.einsum('ij,jk,ik->i', z - m, np.linalg.inv(c), z - m) + np.linalg.slogdet(c)[1])
        new = ll(mu1, C1) + np.log(out.mean()) > ll(mu0, C0) + np.log(1 - out.mean())
        if new.sum() < min_n or np.array_equal(new, out):
            break
        out = new
    return out


def classify(snips, fs, seed=0):
    """
    Two passes.  First pull out events with structure after the initial spike
    (complex-spike candidates), then split what is left into the large unit and
    everything smaller.  Returns 0 (large unit), 1 (candidate complex spike),
    2 (rest) and the baseline-subtracted waveforms.
    """
    raw = snips.astype(float)
    x = raw - np.median(raw[:, :3], axis=1, keepdims=True)
    e = int(0.0008 * fs)

    # from 1 ms on a simple spike is back at baseline, a complex spike is not
    late = raw[:, int(0.001 * fs):]
    z = PCA(4, random_state=seed).fit_transform(late)
    is_cs = _rare_cluster(z)

    lab = np.full(len(x), 2, np.int8)
    rest = np.where(~is_cs)[0]
    early = x[rest, :e]
    g = PCA(3, random_state=seed).fit_transform(early)
    g = (g - g.mean(0)) / g.std(0)
    amp = np.ptp(early, axis=1)
    big = _split2(g, amp > np.median(amp), seed)
    # keep the labelling consistent: "big" is the component with the larger mean waveform
    if np.ptp(early[big].mean(0)) < np.ptp(early[~big].mean(0)):
        big = ~big
    lab[rest[big]] = 0
    lab[is_cs] = 1
    return lab, x


def cs_triggered(ss, cs, win=(-0.05, 0.05), bin_ms=1.0):
    """Simple-spike rate around complex spikes (sp/s) and bin centres (ms)."""
    edges = np.arange(win[0], win[1] + 1e-9, bin_ms / 1000.0)
    h = np.zeros(len(edges) - 1)
    lo = np.searchsorted(ss, cs + win[0]); hi = np.searchsorted(ss, cs + win[1])
    for c, a, b in zip(cs, lo, hi):
        h += np.histogram(ss[a:b] - c, edges)[0]
    return h / max(len(cs), 1) / (bin_ms / 1000.0), (edges[:-1] + edges[1:]) / 2 * 1000


def pause_stats(ss, cs):
    """Suppression ratio (rate 1-8 ms after a complex spike / rate 10-50 ms before) and median pause."""
    if len(cs) < 10 or len(ss) < 100:
        return np.nan, np.nan
    h, c = cs_triggered(ss, cs)
    base = h[(c > -50) & (c < -10)].mean()
    post = h[(c > 1) & (c < 8)].mean()
    nxt = np.searchsorted(ss, cs)
    ok = nxt < len(ss)
    return post / max(base, 1e-9), float(np.median(ss[nxt[ok]] - cs[ok]))


def sort_session(raw, seed=0):
    """
    raw : dict from simulate_session (or the same fields from a real file:
          t, snips, fs, duration)
    Returns simple / complex spike times, a label per event (0 ss, 1 cs, 2 rest)
    and the QC numbers.
    """
    t, fs, T = raw['t'], raw['fs'], raw['duration']
    out, x = classify(raw['snips'], fs, seed)
    ss, cs = t[out == 0], t[out == 1]
    # a complex spike is only believed if simple spikes pause after it
    ratio, pause = pause_stats(ss, cs)
    if not np.isfinite(ratio):
        ratio = np.inf

    w_ss = x[out == 0]
    noise = 1.4826 * np.median(np.abs(x[:, :3] - np.median(x[:, :3])))
    third = len(w_ss) // 3
    stab = np.corrcoef(w_ss[:third].mean(0), w_ss[-third:].mean(0))[0, 1] if third > 20 else np.nan
    isi = np.diff(ss)
    tenths = np.histogram(ss, np.linspace(0, T, 11))[0] / (T / 10)
    qc = dict(
        ss_rate=len(ss) / T, cs_rate=len(cs) / T,
        snr=float(np.ptp(w_ss.mean(0)) / 2 / noise) if len(w_ss) else 0.0,
        isi_viol=float(np.mean(isi < 0.0015)) if len(isi) else 1.0,
        wf_stability=float(stab),
        presence=float(tenths.min() / max(np.median(tenths), 1e-9)),
        pause_ratio=float(ratio), pause_ms=float(pause * 1000) if np.isfinite(pause) else np.nan,
    )
    # thresholds: stability >= .85 is the lab's published criterion; the rest are mine
    qc['keep'] = bool(qc['snr'] >= 4 and qc['isi_viol'] < 0.01 and qc['wf_stability'] >= 0.85
                      and qc['presence'] >= 0.25 and 0.2 <= qc['cs_rate'] <= 4 and qc['pause_ratio'] < 0.25)
    return dict(ss=ss, cs=cs, label=out, qc=qc)


def score_against_truth(label, truth):
    """Precision / recall per class when the true labels are known (simulation)."""
    res = {}
    for k, name in ((0, 'ss'), (1, 'cs')):
        tp = np.sum((label == k) & (truth == k))
        res[name + '_precision'] = float(tp / max(np.sum(label == k), 1))
        res[name + '_recall'] = float(tp / max(np.sum(truth == k), 1))
    return res
