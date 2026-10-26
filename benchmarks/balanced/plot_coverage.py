"""Coverage study (plan Section 7, 2026-10-25): SVD vs balanced at equal bond dimension when the
pulse-free (dephasing) training ensemble is augmented by K in {0, 2, 4} pulsed contexts.

Input: results/coverage_dephasing_p{K}/results.csv (run_benchmark.py --family dephasing --ood_grid
--pulsed_train K; 3 seeds). Pulsed training areas {pi, 3pi}; OOD test pulses {2pi, 4pi, 5pi} at
detunings {-1, 0, 1} meV, disjoint from training.
Output: plots/coverage/coverage.{png,pdf}; prints the OOD ratio SVD / balanced(eta=0.3).
Curves: geometric mean over seeds of rms_final.
"""
import csv
import os

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'plots', 'coverage')
PULSED = [0, 2, 4]
METHODS = [  # (method, eta, label, style)
    ('svd', '', 'plain ACE (SVD)', dict(color=ps.BASELINE, ls='--', marker='s')),
    ('balanced', '0.3', 'balanced, η=0.3', dict(color=ps.SERIES[1], ls='-', marker='^')),
    ('balanced', '0.6', 'balanced, η=0.6', dict(color=ps.SERIES[0], ls='-', marker='o')),
    ('balanced', '1.0', 'balanced, η=1 (HS)', dict(color=ps.SERIES[2], ls='-', marker='D')),
]
SETS = [('ood', 'held-out pulses (OOD)'), ('test', 'held-out dephasing (in distribution)')]


def read(k):
    with open(os.path.join(HERE, 'results', f'coverage_dephasing_p{k}', 'results.csv')) as f:
        return list(csv.DictReader(f))


def curve(rows, split, method, eta):
    maxks = sorted({int(r['maxk']) for r in rows})
    ys = [ps.gmean([float(r['rms_final']) for r in rows if r['set'] == split and int(r['maxk']) == k
                    and r['method'] == method and r['eta'] == eta]) for k in maxks]
    return maxks, ys


def main():
    ps.setup()
    data = {k: read(k) for k in PULSED}
    fig, axes = plt.subplots(len(SETS), len(PULSED), figsize=(2.7 * len(PULSED), 2.5 * len(SETS)),
                             sharex=True, sharey='row')
    for row, (split, split_label) in zip(axes, SETS):
        for ax, k in zip(row, PULSED):
            for method, eta, label, st in METHODS:
                ax.semilogy(*curve(data[k], split, method, eta), label=label, **st)
            ax.set_title(f'{k}/16 pulsed training contexts')
        row[0].set_ylabel(f'RMS error\n{split_label}')
    for ax in axes[-1]:
        ax.set_xlabel('max. bond dimension m')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(METHODS), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Coverage: dephasing training ensemble augmented by pulsed contexts '
                 '(30-mode QD-phonon PT, 3 seeds)', y=1.01)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    ps.save(fig, os.path.join(OUT, 'coverage'))

    print('OOD rms_final ratio SVD / balanced(eta=0.3)  (>1: balanced better)')
    maxks, _ = curve(data[0], 'ood', 'svd', '')
    print('K   ' + ''.join(f'{f"m={m}":>8s}' for m in maxks))
    for k in PULSED:
        _, svd = curve(data[k], 'ood', 'svd', '')
        _, bal = curve(data[k], 'ood', 'balanced', '0.3')
        print(f'{k:<4d}' + ''.join(f'{s / b:8.2f}' for s, b in zip(svd, bal)))


if __name__ == '__main__':
    main()
