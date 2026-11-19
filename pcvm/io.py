"""Session loading, saving and conversion to aligned spike rasters.

Synthetic sessions and public recordings use the same downstream representation."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SYM_WIN = (-1400, 500)     # ms around symbol onset
MOV_WIN = (-500, 700)      # ms around movement onset


@dataclass
class Session:
    name: str
    trials: pd.DataFrame            # block, stim, choice, correct, target (+ rt if known)
    sym: np.ndarray                 # (n_trials, T) uint8, simple spikes, symbol aligned
    mov: np.ndarray                 # (n_trials, T) uint8, movement aligned
    cs_sym: np.ndarray = None       # same for complex spikes when we have them
    cs_mov: np.ndarray = None
    meta: dict = field(default_factory=dict)

    t_sym = np.arange(SYM_WIN[0], SYM_WIN[1])
    t_mov = np.arange(MOV_WIN[0], MOV_WIN[1])

    def block(self, name):
        return np.where(self.trials.block.values == name)[0]

    @property
    def learn(self):
        return self.block('L')

    def prev_correct(self):
        """Outcome of the trial before each trial (first trial of the session: 1)."""
        c = self.trials.correct.values.astype(float)
        return np.r_[1.0, c[:-1]]

    def next_correct(self):
        c = self.trials.correct.values.astype(float)
        return np.r_[c[1:], np.nan]


def save_session(s, path):
    path = Path(path)
    arr = dict(sym=np.packbits(s.sym, axis=1), mov=np.packbits(s.mov, axis=1),
               shape_sym=s.sym.shape, shape_mov=s.mov.shape)
    if s.cs_sym is not None:
        arr.update(cs_sym=np.packbits(s.cs_sym, axis=1), cs_mov=np.packbits(s.cs_mov, axis=1))
    for c in s.trials.columns:
        v = np.asarray(s.trials[c])
        arr['tr_' + c] = v if v.dtype.kind in 'biuf' else v.astype('U8')
    np.savez_compressed(path, name=s.name, meta=json.dumps(s.meta), **arr)


def load_session(path):
    z = np.load(path, allow_pickle=False)
    n_s, n_m = z['shape_sym'][1], z['shape_mov'][1]
    tr = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith('tr_')})
    unpack = lambda k, n: np.unpackbits(z[k], axis=1)[:, :n]
    cs = 'cs_sym' in z.files
    return Session(str(z['name']), tr, unpack('sym', n_s), unpack('mov', n_m),
                   unpack('cs_sym', n_s) if cs else None, unpack('cs_mov', n_m) if cs else None,
                   json.loads(str(z['meta'])))


def load_dir(d):
    return [load_session(p) for p in sorted(Path(d).glob('*.npz'))]


def load_public(matfile, min_learn_trials=40):
    """
    Rasters from github.com/naveen-7/Cerebellum_reward (Data/CS_DATASET.mat,
    released with Sendhilnathan, Ipata & Goldberg 2021).  25 Purkinje cells,
    an overtrained block and a learning block each.

    Behavior{blk}{cell}{trial} = [symbol id, hand, outcome(1 correct / 2 wrong)].
    Signal.symbol covers -3000..500 ms, Signal.movement -500..3000 ms, but only
    about -1435..500 and -500..730 hold data on every trial, so I cut to those.

    The file is named for the complex-spike paper, but the event rates
    (roughly 5-125 /s) are simple-spike rates.  I treat the rasters as simple
    spikes and say so wherever the results are reported.
    """
    from scipy.io import loadmat
    m = loadmat(matfile)
    B = m['Behavior'][0, 0]
    S = m['Signal'][0, 0]
    sym, mov = S['symbol'][0, 0], S['movement'][0, 0]
    t0s = int(m['Times'][0, 0]['start_S'][0, 0]); t0m = int(m['Times'][0, 0]['start_M'][0, 0])

    def cell_block(c, blk):
        b = B[blk][c, 0]
        beh = np.array([np.asarray(b[0, i]).ravel() for i in range(b.shape[1])]).astype(int)
        rs = np.vstack([np.asarray(sym[blk][c, 0][0, i]).ravel() for i in range(b.shape[1])])
        rm = np.vstack([np.asarray(mov[blk][c, 0][0, i]).ravel() for i in range(b.shape[1])])
        rs = rs[:, SYM_WIN[0] - t0s: SYM_WIN[1] - t0s].astype(np.uint8)
        rm = rm[:, MOV_WIN[0] - t0m: MOV_WIN[1] - t0m].astype(np.uint8)
        return beh, rs, rm

    out = []
    for c in range(B['L'].shape[0]):
        parts = [cell_block(c, 'OT'), cell_block(c, 'L')]
        if len(parts[1][0]) < min_learn_trials:
            continue
        beh = np.vstack([p[0] for p in parts])
        blk = np.r_[np.full(len(parts[0][0]), 'OT'), np.full(len(parts[1][0]), 'L')]
        ids = beh[:, 0]
        stim = np.zeros(len(ids), int)
        for b in ('OT', 'L'):
            j = blk == b
            stim[j] = (ids[j] == np.unique(ids[j]).max()).astype(int)
        choice = beh[:, 1] - 1
        correct = (beh[:, 2] == 1).astype(int)
        target = np.where((choice == 1) == (correct == 1), 1.0, -1.0)
        tr = pd.DataFrame(dict(block=blk, stim=stim, choice=choice, correct=correct, target=target, symbol=ids))
        reversal = set(ids[blk == 'OT']) == set(ids[blk == 'L'])
        out.append(Session(f'pub{c:02d}', tr, np.vstack([p[1] for p in parts]), np.vstack([p[2] for p in parts]),
                           meta=dict(source='public', cell=c, reversal=bool(reversal))))
    return out


def load_all(root, qc_only=True):
    """Simulated sessions (QC-passed) and, if the file is there, the public rasters."""
    root = Path(root)
    data = {}
    sim = load_dir(root / 'data' / 'sessions')
    if sim:
        data['sim'] = [s for s in sim if s.meta['qc']['keep'] or not qc_only]
    pub = root / 'data' / 'public' / 'CS_DATASET.mat'
    if pub.exists():
        ok = []
        for s in load_public(pub):
            acc = s.trials.correct.values[s.learn].mean()
            if acc < 0.25:       # far below chance on a two-choice task: not a learning block as I understand it
                continue
            ok.append(s)
        data['public'] = ok
    return data
