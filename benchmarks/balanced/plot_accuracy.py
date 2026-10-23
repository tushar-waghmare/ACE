"""
Accuracy figures: plain ACE (SVD sweep to compress_maxk) vs protected balanced truncation
at equal bond dimension, from run_benchmark.py results (geometric mean over seeds, band =
min-max over seeds).

    ~/.venvs/ace/bin/python benchmarks/balanced/plot_accuracy.py [--outdir benchmarks/balanced/plots/accuracy]

Writes (PNG + PDF):
  accuracy_<pt>.*          rms_trace vs maxk, rows = family, columns = train / test / ood
  trace_preservation.*     max |tr rho_S(t) - tr_ref(t)| vs maxk (closure protection)
  improvement.*            rms_trace(plain ACE) / rms_trace(balanced, eta=0.3), all PTs
  amplification.*          max ||X|| ||Y|| over cuts vs maxk (oblique amplification)
"""
import argparse
import csv
import os
from collections import defaultdict

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FAMILIES = ['mixed', 'pulses', 'dephasing']
SETS = [('train', 'training contexts'), ('test', 'held-out (same family)'), ('ood', 'out of distribution')]
FAMILY_LABEL = {'mixed': 'mixed family', 'pulses': 'pulse family', 'dephasing': 'pure dephasing'}
PTS = [('bench30_thr1e-6', '30 modes (thr 1e-6)'), ('bench30_thr3e-7', '30 modes (thr 3e-7)'),
       ('bench100', '100 modes (thr 5e-8)')]


def load(pt, family):
    path = os.path.join(RES, f'{pt}_{family}', 'results.csv')
    if not os.path.exists(path):
        return None
    data = defaultdict(list)   # (set, method, maxk) -> rows
    for r in csv.DictReader(open(path)):
        method = 'svd' if r['method'] == 'svd' else f"eta={float(r['eta']):g}"
        data[(r['set'], method, int(r['maxk']))].append(r)
    return data


def series(data, split, method, metric):
    ks = sorted({k for (s, m, k) in data if s == split and m == method})
    vals = [[float(r[metric]) for r in data[(split, method, k)]] for k in ks]
    return ks, [ps.gmean(v) for v in vals], [min(v) for v in vals], [max(v) for v in vals]


def draw(ax, data, split, metric, methods=ps.METHODS, band=True):
    for key, label, style in methods:
        ks, g, lo, hi = series(data, split, key, metric)
        if not ks:
            continue
        ax.plot(ks, g, label=label, **style)
        if band:
            ax.fill_between(ks, lo, hi, color=style['color'], alpha=0.12, lw=0)
    ax.set_yscale('log')
    ax.set_xticks([4, 8, 12, 16, 24])


def fig_accuracy(pt, ptlabel, out):
    fams = [f for f in FAMILIES if load(pt, f)]
    if not fams:
        return
    fig, axes = plt.subplots(len(fams), 3, figsize=(9.6, 2.5 * len(fams) + 0.6), squeeze=False,
                             sharex=True, layout='constrained')
    for i, fam in enumerate(fams):
        data = load(pt, fam)
        for j, (split, slabel) in enumerate(SETS):
            ax = axes[i][j]
            draw(ax, data, split, 'rms_trace')
            if i == 0:
                ax.set_title(slabel)
            if j == 0:
                ax.set_ylabel(f'{FAMILY_LABEL[fam]}\nRMS error of ⟨O⟩(t)')
            if i == len(fams) - 1:
                ax.set_xlabel('max. bond dimension (maxk)')
        # same y-range within a row: compare train/test/ood directly
        lo = min(a.get_ylim()[0] for a in axes[i]); hi = max(a.get_ylim()[1] for a in axes[i])
        for a in axes[i]:
            a.set_ylim(lo, hi)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=4)
    fig.suptitle(f'Observable error vs bond dimension, {ptlabel} PT '
                 '(geometric mean of 3 seeds, band: min–max)', fontsize=10.5)
    ps.save(fig, os.path.join(out, f'accuracy_{pt}'))


def fig_trace(out):
    pt = 'bench100'
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True, layout='constrained')
    for ax, fam in zip(axes, FAMILIES):
        data = load(pt, fam)
        draw(ax, data, 'test', 'max_trace_dev_vs_ref')
        ax.set_title(FAMILY_LABEL[fam])
        ax.set_xlabel('max. bond dimension (maxk)')
    axes[0].set_ylabel('max |tr ρ_S(t) − tr_ref(t)|')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=4)
    fig.suptitle('Trace preservation on held-out contexts, 100-mode PT (deviation from the input PT)', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'trace_preservation'))


def fig_improvement(out):
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.4), sharex=True, sharey=True, layout='constrained')
    styles = [dict(color=ps.SERIES[0], marker='o'), dict(color=ps.SERIES[1], marker='^'),
              dict(color=ps.SERIES[2], marker='D')]
    for i, split in enumerate(['test', 'ood']):
        for j, fam in enumerate(FAMILIES):
            ax = axes[i][j]
            ax.axhline(1, color=ps.INK2, lw=0.9, ls=':')
            for (pt, ptlabel), st in zip(PTS, styles):
                data = load(pt, fam)
                if not data:
                    continue
                ks, svd, _, _ = series(data, split, 'svd', 'rms_trace')
                _, bal, _, _ = series(data, split, 'eta=0.3', 'rms_trace')
                ax.plot(ks, [a / b for a, b in zip(svd, bal)], label=ptlabel, **st)
            ax.set_yscale('log')
            ax.set_xticks([4, 8, 12, 16, 24])
            if i == 0:
                ax.set_title(FAMILY_LABEL[fam])
            else:
                ax.set_xlabel('max. bond dimension (maxk)')
            if j == 0:
                ax.set_ylabel(('held-out' if split == 'test' else 'out of distribution') +
                              '\nerror ratio SVD / balanced')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=3)
    fig.suptitle('Error ratio plain ACE / balanced (η=0.3) at equal bond dimension '
                 '(RMS error of ⟨O⟩(t); above 1: balanced is more accurate)', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'improvement'))


def fig_amplification(out):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True, layout='constrained')
    for ax, fam in zip(axes, FAMILIES):
        data = load('bench100', fam)
        draw(ax, data, 'train', 'max_amplification', methods=ps.METHODS[1:])
        ax.set_title(FAMILY_LABEL[fam])
        ax.set_xlabel('max. bond dimension (maxk)')
    axes[0].set_ylabel('max over cuts of ‖X‖₂‖Y‖₂')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=3)
    fig.suptitle('Oblique amplification of the balanced projectors, 100-mode PT', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'amplification'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'plots', 'accuracy'))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ps.setup()
    for pt, label in PTS:
        fig_accuracy(pt, label, a.outdir)
    fig_trace(a.outdir)
    fig_improvement(a.outdir)
    fig_amplification(a.outdir)
    print(f'wrote figures to {a.outdir}')


if __name__ == '__main__':
    main()
