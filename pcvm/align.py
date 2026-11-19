import numpy as np
from scipy.ndimage import gaussian_filter1d


def sliding_counts(raster, t, width=200, step=50):
    """Spike counts in sliding windows.  Returns counts (n_trials, W) and window centres (ms)."""
    cs = np.cumsum(np.c_[np.zeros(len(raster), np.int32), raster.astype(np.int32)], axis=1)
    starts = np.arange(0, raster.shape[1] - width + 1, step)
    counts = cs[:, starts + width] - cs[:, starts]
    return counts, t[starts] + width / 2.0


def window_count(raster, t, t0, t1):
    return raster[:, (t >= t0) & (t < t1)].sum(1)


def sdf(raster, sigma=15):
    """Trial-averaged spike density (sp/s), Gaussian kernel with sd `sigma` ms."""
    return gaussian_filter1d(raster.mean(0) * 1000.0, sigma, mode='reflect')


def trial_windows(sess, width=200, step=50):
    """
    Sliding-window counts over the whole trial: symbol-aligned windows up to
    +300 ms, then movement-aligned windows from -100 ms on.  Returns counts,
    a label per window ('sym' / 'mov') and its centre.
    """
    cs, ts = sliding_counts(sess.sym, sess.t_sym, width, step)
    cm, tm = sliding_counts(sess.mov, sess.t_mov, width, step)
    ks = ts <= 300
    km = tm >= -100
    counts = np.c_[cs[:, ks], cm[:, km]]
    lab = np.r_[np.full(ks.sum(), 'sym'), np.full(km.sum(), 'mov')]
    return counts, lab, np.r_[ts[ks], tm[km]]


def events_to_raster(spk, ev, win):
    """Spike times (s) -> (n_events, T) uint8 raster at 1 ms around each event time (s)."""
    T = win[1] - win[0]
    out = np.zeros((len(ev), T), np.uint8)
    lo = np.searchsorted(spk, ev + win[0] / 1000.0)
    hi = np.searchsorted(spk, ev + win[1] / 1000.0)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a and np.isfinite(ev[i]):
            j = np.floor((spk[a:b] - ev[i]) * 1000.0).astype(int) - win[0]
            out[i, j[(j >= 0) & (j < T)]] = 1
    return out
