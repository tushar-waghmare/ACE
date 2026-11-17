"""Joined-PT benchmark (plan Section 7, 2026-10-25): bench30 phonons x lossy cavity, exact join
(D = 1620), mixed ensemble 16/32/32, 3 seeds, grid 15870779-81 (+ rerun of tasks 2-7).

Input: results/joined_cavity_mixed/results.csv (merged by hpc/merge_results.py on ARCHER2).
Output: plots/joined_cavity/joined_cavity.{png,pdf}; prints the ratio SVD / balanced (eta=0.3).
Curves: geometric mean over seeds; trace error: max over seeds.
"""
import csv
import os

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results', 'joined_cavity_mixed', 'results.csv')
OUT = os.path.join(HERE, 'plots', 'joined_cavity')
METHODS = [  # (method, eta, label, style)
    ('svd', '', 'plain ACE (SVD)', dict(color=ps.BASELINE, ls='--', marker='s')),
    ('balanced', '0.0', 'balanced, η=0', dict(color=ps.SERIES[0], ls='-', marker='o')),
    ('balanced', '0.3', 'balanced, η=0.3', dict(color=ps.SERIES[1], ls='-', marker='^')),
    ('balanced', '1.0', 'balanced, η=1 (HS)', dict(color=ps.SERIES[2], ls='-', marker='D')),
]
SETS = [('train', 'training contexts'), ('test', 'held out, in distribution'), ('ood', 'held out, OOD')]


def read():
    with open(CSV) as f:
        return list(csv.DictReader(f))


def series(rows, method, eta, key, split='train', agg=ps.gmean):
    maxks = sorted({int(r['maxk']) for r in rows})
    ys = [agg([float(r[key]) for r in rows if r['set'] == split and int(r['maxk']) == k
               and r['method'] == method and r['eta'] == eta]) for k in maxks]
    return maxks, ys


def main():
    ps.setup()
    rows = read()
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.2), sharex=True)
    for ax, (split, label) in zip(axes[0], SETS):
        for method, eta, lab, st in METHODS:
            ax.semilogy(*series(rows, method, eta, 'rms_final', split), label=lab, **st)
        ax.set_title(f'RMS error, {label}')
    axes[0][0].set_ylabel('RMS error of final readout')

    ax = axes[1][0]
    for method, eta, lab, st in METHODS:
        ax.semilogy(*series(rows, method, eta, 'max_trace_err', agg=max), **st)
    ax.set_title('max |tr ρ − 1| (max over seeds)')
    ax = axes[1][1]
    for method, eta, lab, st in METHODS:
        ax.semilogy(*series(rows, method, eta, 'hs_rel'), **st)
    ax.set_title('relative HS error')
    ax = axes[1][2]
    ax.axhline(1, color=ps.INK2, lw=0.8)
    for (split, label), c, m in zip(SETS, ps.SERIES[3:6], 'os^'):
        k, sv = series(rows, 'svd', '', 'rms_final', split)
        _, b = series(rows, 'balanced', '0.3', 'rms_final', split)
        ax.semilogy(k, [x / y for x, y in zip(sv, b)], color=c, marker=m, label=label)
    ax.set_title('SVD / balanced η=0.3 (>1: balanced better)')
    ax.legend(loc='upper left')
    for ax in axes[1]:
        ax.set_xlabel('max. bond dimension m')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(METHODS), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Joined PT: 30-mode QD phonons × lossy cavity (D = 1620), 16 training contexts, 3 seeds',
                 y=1.01)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    ps.save(fig, os.path.join(OUT, 'joined_cavity'))

    print('SVD / balanced(eta=0.3), geometric mean over seeds')
    k = sorted({int(r['maxk']) for r in rows})
    print('maxk  ' + ' '.join(f'{x:6d}' for x in k))
    for split, _ in SETS:
        _, sv = series(rows, 'svd', '', 'rms_final', split)
        _, b = series(rows, 'balanced', '0.3', 'rms_final', split)
        print(f'{split:5s} ' + ' '.join(f'{x / y:6.2f}' for x, y in zip(sv, b)))


if __name__ == '__main__':
    main()
