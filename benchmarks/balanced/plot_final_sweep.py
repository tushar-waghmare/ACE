"""
Figures for the end-to-end balanced final sweep benchmark (final_sweep_benchmark.py).

    ~/.venvs/ace/bin/python benchmarks/balanced/plot_final_sweep.py [--indir results/final_sweep]

Writes (PNG + PDF) to plots/final_sweep/:
  final_sweep_accuracy.*  RMS error of <O>(t) vs final bond dimension m, both arms
                          (identical calculation, only the last truncation differs);
                          rows = family, columns = reference (base / tight / exact)
  final_sweep_cost.*      wall time of each bin/ACE calculation (builds ran in parallel)
                          and the share of the balanced final sweep in it
"""
import argparse
import csv
import os
from collections import defaultdict

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [('svd', 'plain ACE: SVD final sweep to m', ps.METHODS[0][2]),
        ('balanced', 'balanced final sweep to m (η=0.3)', ps.METHODS[2][2])]
REFS = [('base', 'truncation error\n(vs. same calculation, no final target)'),
        ('tight', 'total error\n(vs. tighter PT, threshold 3e-7)'),
        ('exact', 'total error\n(vs. exact solution)')]
SETS = [('test', '-'), ('ood', ':')]
FAMILY_LABEL = {'mixed': 'mixed family', 'dephasing': 'pure dephasing'}


def fig_accuracy(rows, out):
    fams = [f for f in FAMILY_LABEL if any(r['family'] == f for r in rows)]
    refs = [(k, l) for k, l in REFS if any(r['reference'] == k for r in rows)]
    fig, axes = plt.subplots(len(fams), len(refs), figsize=(3.3 * len(refs), 2.8 * len(fams) + 0.6),
                             squeeze=False, layout='constrained')
    for i, fam in enumerate(fams):
        for j, (ref, rlabel) in enumerate(refs):
            ax = axes[i][j]
            sub = [r for r in rows if r['family'] == fam and r['reference'] == ref]
            if not sub:
                ax.set_visible(False)
                continue
            for split, ls in SETS:
                base = [float(r['rms_trace']) for r in sub if r['method'] == 'base' and r['set'] == split]
                for key, label, style in ARMS:
                    pts = sorted((int(r['maxk']), float(r['rms_trace'])) for r in sub
                                 if r['method'] == key and r['set'] == split)
                    if not pts:
                        continue
                    st = dict(style, ls=ls)
                    ax.plot(*zip(*pts), label=f'{label}, {"held-out" if split == "test" else "OOD"}', **st)
                if base and ref != 'base':
                    ax.axhline(base[0], color=ps.INK2, lw=0.9, ls=ls)
            ax.set_yscale('log')
            ax.set_xticks(sorted({int(r['maxk']) for r in sub if r['maxk']}))
            if not any(axes[k][j].get_title() for k in range(i)):
                ax.set_title(rlabel, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f'{FAMILY_LABEL[fam]}\nRMS error of ⟨O⟩(t)')
            if i == len(fams) - 1:
                ax.set_xlabel('final bond dimension m')
    h, l = axes[0][0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=ps.INK2, lw=0.9))
    l.append('untruncated calculation (base; solid held-out, dotted OOD)')
    fig.legend(h, l, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('bin/ACE with balanced vs SVD final sweep (30 modes, threshold 1e-6, '
                 'identical intermediate sweeps)', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'final_sweep_accuracy'))


def fig_cost(builds, indir, out):
    names = [b['name'] for b in builds]
    secs = [float(b['build_seconds']) for b in builds]
    bal = []
    for n in names:
        # 'Balanced sweep timings [s]: ... total=<s>' in the calculation's log (verbosity 1)
        t = 0.
        for line in open(os.path.join(indir, f'{n}.log')):
            if line.startswith('Balanced sweep timings'):
                t = float(line.rsplit('total=', 1)[1])
        bal.append(t)
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(names) + 1.2), layout='constrained')
    ys = list(range(len(names)))[::-1]
    col = [ps.METHODS[2][2]['color'] if n.startswith('balanced') else
           (ps.BASELINE if n.startswith('svd') else '#a8a7a2') for n in names]
    ax.barh(ys, secs, color=col, height=0.65)
    for y, s, b in zip(ys, secs, bal):
        txt = f' {s:.0f} s' + (f'  (balanced sweep {b:.1f} s)' if b > 0 else '')
        ax.text(s, y, txt, va='center', fontsize=7.5, color=ps.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('wall-clock time of the bin/ACE calculation [s] (builds ran 9 in parallel)')
    ax.set_xlim(0, max(secs) * 1.35)
    ax.grid(axis='y', visible=False)
    fig.suptitle('Cost of the PT calculation with SVD vs balanced final sweep', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'final_sweep_cost'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--indir', default=os.path.join(HERE, 'results', 'final_sweep'))
    ap.add_argument('--outdir', default=os.path.join(HERE, 'plots', 'final_sweep'))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ps.setup()
    rows = list(csv.DictReader(open(os.path.join(a.indir, 'results.csv'))))
    builds = list(csv.DictReader(open(os.path.join(a.indir, 'builds.csv'))))
    fig_accuracy(rows, a.outdir)
    fig_cost(builds, a.indir, a.outdir)
    print(f'wrote figures to {a.outdir}')


if __name__ == '__main__':
    main()
