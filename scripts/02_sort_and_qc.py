"""Sort synthetic spike waveforms and save trial-aligned sessions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pcvm import sorting
from pcvm.align import events_to_raster
from pcvm.io import Session, save_session, SYM_WIN, MOV_WIN

ap = argparse.ArgumentParser()
ap.add_argument('--rm-raw', action='store_true', help='delete raw files once sorted')
a = ap.parse_args()

raw_dir, out = ROOT / 'data' / 'raw', ROOT / 'data' / 'sessions'
out.mkdir(parents=True, exist_ok=True)
for f in out.glob('*.npz'):
    f.unlink()

rows, examples = [], []
for path in sorted(raw_dir.glob('*.npz')):
    z = np.load(path)
    raw = dict(t=z['t'], snips=z['snips'], fs=int(z['fs']), duration=float(z['duration']))
    truth = json.loads(str(z['truth']))
    s = sorting.sort_session(raw)
    qc = dict(s['qc'])
    if 'true_label' in z.files:
        qc.update(sorting.score_against_truth(s['label'], z['true_label']))

    tr = pd.DataFrame({k[3:]: z[k] for k in z.files if k.startswith('tr_')})
    t_sym, t_mov = tr.t_sym.values, tr.t_move.values
    sess = Session(path.stem, tr[['block', 'stim', 'choice', 'correct', 'target', 'rt', 'pc']],
                   events_to_raster(s['ss'], t_sym, SYM_WIN), events_to_raster(s['ss'], t_mov, MOV_WIN),
                   events_to_raster(s['cs'], t_sym, SYM_WIN), events_to_raster(s['cs'], t_mov, MOV_WIN),
                   meta=dict(source='sim', truth=truth, qc={k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                                                            for k, v in qc.items()}))
    save_session(sess, out / f'{path.stem}.npz')
    rows.append(dict(session=path.stem, monkey=truth['monkey'], true_snr=truth['snr'], lost=truth['lost'], **qc))
    if len(examples) < 3 and qc['keep']:
        x = raw['snips'].astype(float)
        h, c = sorting.cs_triggered(s['ss'], s['cs'])
        examples.append(dict(name=path.stem, ss=x[s['label'] == 0][:150], cs=x[s['label'] == 1][:60],
                             other=x[s['label'] == 2][:150], h=h, c=c, fs=raw['fs']))
    if a.rm_raw:
        path.unlink()

qc = pd.DataFrame(rows)
qc.to_csv(ROOT / 'results' / 'qc.csv', index=False)
k = qc.keep
print(f'{len(qc)} sessions, {k.sum()} pass QC')
print('rejected for:', dict(low_snr=int((qc.snr < 4).sum()), isi=int((qc.isi_viol >= .01).sum()),
                            unstable=int((qc.wf_stability < .85).sum()), presence=int((qc.presence < .25).sum()),
                            no_pause=int((~(qc.pause_ratio < .25)).sum())))
if 'cs_recall' in qc:
    print('sorter vs ground truth, kept cells (median): ' +
          ', '.join(f'{c} {qc[k][c].median():.3f}' for c in ('ss_precision', 'ss_recall', 'cs_precision', 'cs_recall')))

# figure ---------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pcvm import plotting as P
P.style()
fig, ax = plt.subplots(2, 3, figsize=(10.5, 6))
ex = examples[0]
t = np.arange(ex['ss'].shape[1]) / ex['fs'] * 1000
for w, col, lab in ((ex['other'], P.LIGHT, 'other events'), (ex['ss'], P.BLUE, 'simple spikes'), (ex['cs'], P.ORANGE, 'complex spikes')):
    ax[0, 0].plot(t, w[:40].T, color=col, lw=0.5, alpha=0.5)
    ax[0, 0].plot([], [], color=col, label=lab)
ax[0, 0].set(xlabel='ms', ylabel='amplitude (a.u.)', title=f'Waveforms, {ex["name"]}')
ax[0, 0].legend(loc='lower right')
for e, col in zip(examples, (P.BLUE, P.AQUA, P.VIOLET)):
    ax[0, 1].plot(e['c'], e['h'], color=col, lw=1.4, label=e['name'])
P.events(ax[0, 1])
ax[0, 1].set(xlabel='ms from complex spike', ylabel='simple spikes / s', title='Pause after a complex spike')
ax[0, 1].legend()
ax[0, 2].scatter(qc.ss_rate[k], qc.cs_rate[k], s=14, color=P.BLUE, label='kept', edgecolor='white', lw=0.4)
ax[0, 2].scatter(qc.ss_rate[~k], qc.cs_rate[~k], s=18, color=P.ORANGE, marker='s', label='rejected', edgecolor='white', lw=0.4)
ax[0, 2].set(xlabel='simple spike rate (/s)', ylabel='complex spike rate (/s)', title='Firing rates')
ax[0, 2].legend()
for a_, col, thr, lab in ((ax[1, 0], 'snr', 4, 'SNR'), (ax[1, 1], 'presence', 0.25, 'presence (min tenth / median)'),
                          (ax[1, 2], 'pause_ratio', 0.25, 'rate after CS / before')):
    v = qc[col].replace([np.inf], np.nan).dropna()
    a_.hist(v, bins=25, color=P.LIGHT)
    a_.axvline(thr, color=P.GREY, lw=1)
    a_.text(thr, a_.get_ylim()[1] * 0.95, ' threshold', fontsize=8, color=P.GREY, va='top')
    a_.set(xlabel=lab, ylabel='sessions')
ax[1, 0].set_title('Isolation')
ax[1, 1].set_title('Cell held for the whole session')
ax[1, 2].set_title('Same cell fires both spike types')
fig.tight_layout()
P.save(fig, ROOT / 'docs' / 'figures' / 'sorting_qc.png')
