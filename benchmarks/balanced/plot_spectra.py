"""Singular-value decay at a few cuts of bench30_thr1e-6.pt (coverage study, plan Section 7, 2026-10-25).

Input: results/spectra/spectra_<ensemble>.csv written by PTB_balanced_truncate -balanced_spectra
(eta=0.3, no effective rank limit, so the spectra describe the untruncated PT):
  op     protected operational core sigma_i (what eta=0 truncates), one curve per training ensemble
  hs     unprotected HS Schmidt values sigma(R L) (dictionary multiplicities), ensemble independent
Output: plots/spectra/spectra_decay.{png,pdf}; prints the HS mismatch between runs as a check.
"""
import csv
import os
from collections import defaultdict

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results', 'spectra')
OUT = os.path.join(HERE, 'plots', 'spectra')
FLOOR = 1e-16

ENSEMBLES = [  # (file tag, label, style)
    ('dephasing', 'operational: dephasing (16 pulse-free)', dict(color=ps.SERIES[0], ls='-', marker='o')),
    ('dephasing_p4', 'operational: dephasing + 4 pulses', dict(color=ps.SERIES[1], ls='-', marker='^')),
    ('pulses', 'operational: pulses (16 single pulses)', dict(color=ps.SERIES[2], ls='-', marker='D')),
]
HS_STYLE = dict(color=ps.BASELINE, ls='--', marker='s')


def read(tag):
    sp = defaultdict(list)
    with open(os.path.join(RES, f'spectra_{tag}.csv')) as f:
        for r in csv.DictReader(f):
            sp[(int(r['cut']), r['kind'])].append(float(r['sigma']))
    return sp


def normalized(s):
    return [max(x / s[0], FLOOR) for x in s] if s and s[0] > 0 else []


def main():
    ps.setup()
    data = {tag: read(tag) for tag, _, _ in ENSEMBLES}
    cuts = sorted({c for c, _ in data['dephasing']})
    # HS Schmidt values do not depend on the training ensemble (untruncated past): check
    hs_dev = max(abs(a - b) / data['dephasing'][(c, 'hs')][0]
                 for tag in data for c in cuts
                 for a, b in zip(data[tag][(c, 'hs')], data['dephasing'][(c, 'hs')]))
    print(f'max relative HS-spectrum mismatch between runs: {hs_dev:.1e}')

    fig, axes = plt.subplots(1, len(cuts), figsize=(2.5 * len(cuts), 2.9), sharey=True)
    for ax, c in zip(axes, cuts):
        hs = normalized(data['dephasing'][(c, 'hs')])
        ax.semilogy(range(1, len(hs) + 1), hs, label='HS Schmidt values (SVD)', **HS_STYLE)
        for tag, label, st in ENSEMBLES:
            s = normalized(data[tag][(c, 'op')])
            ax.semilogy(range(1, len(s) + 1), s, label=label, **st)
        ax.set_title(f'n = {c}  (t = {0.1 * (c + 1):.1f} ps)')
        ax.set_ylim(1e-14, 2)
    axes[0].set_ylabel('σ_i / σ_1')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.supxlabel('index i', fontsize=9, y=-0.04)
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle('Singular-value decay: operational cores by training ensemble vs HS Schmidt spectrum '
                 '(30-mode QD-phonon PT)', y=1.02)
    os.makedirs(OUT, exist_ok=True)
    ps.save(fig, os.path.join(OUT, 'spectra_decay'))

    # summary: number of values above 1e-3 sigma_1 per cut and ensemble
    print('cut  ' + '  '.join(f'{t:>13s}' for t, _, _ in ENSEMBLES) + '            hs   (#sigma_i/sigma_1 > 1e-3)')
    for c in cuts:
        cnt = [sum(x > 1e-3 for x in normalized(data[t][(c, 'op')])) for t, _, _ in ENSEMBLES]
        hs = sum(x > 1e-3 for x in normalized(data['dephasing'][(c, 'hs')]))
        print(f'{c:4d} ' + '  '.join(f'{n:13d}' for n in cnt) + f'  {hs:12d}')


if __name__ == '__main__':
    main()
