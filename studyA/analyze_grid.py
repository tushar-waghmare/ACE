"""
Study A grid analysis (docs/studyA_spec.md Sec. 1 and 5; deviations in studyA/CHANGES.md).

  python studyA/analyze_grid.py [--results studyA/results_cirrus] [--figs studyA/figs]

Reads ranks.csv and heldout.csv of the grid run, evaluates the decision rule A1-A5 at
alpha = 0.05, eps = 3e-3, middle cut, theta = 1e-6 (primary; theta/10 = 1e-7 for convergence),
prints the numbers used in studyA/REPORT.md and writes fig2_{a,b,c,d}.pdf.
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EPS, THETA, THETA10 = 3e-3, '1e-6', '1e-7'
NBATHS = [('ohmic', 0), ('oob2', 2), ('oob4', 4), ('oob6', 6)]
CONTROLS = ['ohmic', 'edge5', 'edge3', 'inband0.8']
NARROW = ['bernoulli:0.002', 'bernoulli:0.005', 'coarse:20:0.3']
FAMS = ['flat', 'bernoulli:0.002', 'bernoulli:0.005', 'coarse:20:0.3', 'bernoulli:0.01', 'coarse:10:0.3']
NAME = {'flat': 'HS (flat)', 'bernoulli:0.002': 'Bernoulli p=0.002', 'bernoulli:0.005': 'Bernoulli p=0.005',
        'bernoulli:0.01': 'Bernoulli p=0.01', 'coarse:10:0.3': 'coarse k=10', 'coarse:20:0.3': 'coarse k=20'}
# reference categorical palette (dataviz skill, references/palette.md), fixed order; markers as 2nd encoding
COLOR = {'flat': '#2a78d6', 'bernoulli:0.002': '#eb6834', 'bernoulli:0.005': '#1baf7a',
         'coarse:20:0.3': '#eda100', 'bernoulli:0.01': '#e87ba4', 'coarse:10:0.3': '#008300'}
MARKER = {'flat': 'o', 'bernoulli:0.002': 's', 'bernoulli:0.005': 'D', 'coarse:20:0.3': '^',
          'bernoulli:0.01': 'v', 'coarse:10:0.3': 'P'}
# Stage 0 predictions (spec Sec. 1 table, central values; tolerance +-2)
PRED = {'flat': {0: 2, 2: 6, 4: 10, 6: 14}, 'bernoulli:0.002': {0: 2, 2: 2, 4: 2, 6: 2},
        'bernoulli:0.005': {0: 2, 2: 2, 4: 2, 6: 2}, 'coarse:20:0.3': {0: 2, 2: 3, 4: 3, 6: 3},
        'bernoulli:0.01': {0: 3, 2: 3, 4: 3, 6: 2}}
TEXT, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=TEXT, labelsize=8)


def load_ranks(path):
    R = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r['cut'] != 'mid':
                continue
            key = (float(r['alpha']), r['bath'], r['theta'], r['family'], float(r['eps']))
            R[key] = (int(r['chi']), float(r['exponent_scale']))
    return R


def load_heldout(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ('eta', 'rms_err', 'rms_y', 'cert_stepindep', 'cert_paired', 'max_trace_err'):
                r[k] = float(r[k])
            for k in ('maxk', 'bond'):
                r[k] = int(r[k])
            rows.append(r)
    return rows


def chi(R, alpha, bath, fam, theta=THETA, eps=EPS):
    v = R.get((alpha, bath, theta, fam, eps))
    return None if v is None else v[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results_cirrus'))
    ap.add_argument('--figs', default=os.path.join(HERE, 'figs'))
    a = ap.parse_args()
    os.makedirs(a.figs, exist_ok=True)
    R = load_ranks(os.path.join(a.results, 'ranks.csv'))
    H = load_heldout(os.path.join(a.results, 'heldout.csv'))

    # ---------------------------------------------------------------- tables
    print('## ranks, alpha 0.05, eps 3e-3, mid cut, theta 1e-6')
    for fam in FAMS:
        print(f'{NAME[fam]:20s}', [chi(R, 0.05, b, fam) for b, _ in NBATHS], 'pred', PRED.get(fam))
    hs = [chi(R, 0.05, b, 'flat') for b, _ in NBATHS]
    print(f'\nA1: chi_HS {hs[0]} -> {hs[-1]}: increase {hs[-1]-hs[0]} (>= 6): {hs[-1]-hs[0] >= 6}')
    a2 = []
    for fam in NARROW:
        c = [chi(R, 0.05, b, fam) for b, _ in NBATHS]
        ok = (max(c) - c[0] <= 1) and (c[-1] - c[0] <= 1) and hs[-1] / c[-1] >= 3
        a2.append(ok)
        print(f'A2 {NAME[fam]}: chi_op {c}, increase {c[-1]-c[0]}, ratio at n=6 {hs[-1]/c[-1]:.1f}: {ok}')
    print(f'A5: A2 holds for {sum(a2)} narrow families (>= 2): {sum(a2) >= 2}')

    print('\nA3 (oob6, bond chi_op+2, in-family held-out set)')
    for fam in NARROW:
        c = chi(R, 0.05, 'oob6', fam)
        sel = {(r['method'], r['eta']): r for r in H if r['label'] == 'oob6_a0.05_thr1e-6'
               and r['family'] == fam and r['set'] == 'in' and r['bond'] == c + 2}
        b, s = sel[('balanced', 0.0)], sel[('svd', -1.0)]
        thr = EPS * b['rms_y']
        print(f'  {NAME[fam]}: bond {c+2} balanced {b["rms_err"]:.3g} vs eps*RMS(y) {thr:.3g}: '
              f'{b["rms_err"] <= thr}; cert_si {b["cert_stepindep"]:.3g} >= err: {b["cert_stepindep"] >= b["rms_err"]}; '
              f'SVD {s["rms_err"]:.3g} ({s["rms_err"]/b["rms_err"]:.0f}x)')

    print('\nA4 controls (chi_op and exponent scale vs ohmic)')
    for fam in FAMS[1:]:
        co, ci = chi(R, 0.05, 'ohmic', fam), chi(R, 0.05, 'inband0.8', fam)
        eo, ei = R[(0.05, 'ohmic', THETA, fam, EPS)][1], R[(0.05, 'inband0.8', THETA, fam, EPS)][1]
        print(f'  {NAME[fam]}: chi {co} -> {ci} (+{ci-co}), exponent scale {eo:.3g} -> {ei:.3g} (x{ei/eo:.1f})')
    print('A4 theta convergence: |chi(1e-6) - chi(theta)| at eps 3e-3, mid cut')
    worst = []
    for b in [b for b, _ in NBATHS] + ['edge5', 'edge3', 'inband0.8']:
        for th in ('1e-7', '1e-8', '1e-9'):
            d = {f: (chi(R, 0.05, b, f), chi(R, 0.05, b, f, th)) for f in FAMS}
            d = {f: abs(x - y) for f, (x, y) in d.items() if x is not None and y is not None}
            if d:
                f = max(d, key=d.get)
                print(f'  {b:9s} {th}: max change {d[f]} ({NAME[f]})')
                worst.append((b, th, d[f]))
            else:
                print(f'  {b:9s} {th}: no PT')
    print('\nsupplementary: oob6 in-family held-out error vs bond (balanced eta 0 / eta 0.3 / SVD)')
    for fam in NARROW + ['bernoulli:0.01']:
        print(' ', NAME[fam])
        for bond in sorted({r['bond'] for r in H if r['label'] == 'oob6_a0.05_thr1e-6' and r['family'] == fam}):
            sel = {(r['method'], r['eta']): r for r in H if r['label'] == 'oob6_a0.05_thr1e-6'
                   and r['family'] == fam and r['set'] == 'in' and r['bond'] == bond}
            b0, b3, s = sel[('balanced', 0.0)], sel[('balanced', 0.3)], sel[('svd', -1.0)]
            print(f'    bond {bond:2d}: {b0["rms_err"]:.2e} / {b3["rms_err"]:.2e} / {s["rms_err"]:.2e} '
                  f'(thr {EPS*b0["rms_y"]:.2e}, cert_si {b0["cert_stepindep"]:.2e}, cert_p {b0["cert_paired"]:.2e}, '
                  f'trace balanced {b0["max_trace_err"]:.1e} SVD {s["max_trace_err"]:.1e})')
    bad = [r for r in H if r['method'] == 'balanced' and r['cert_paired'] < r['rms_err']]
    print(f'\npaired certificate below the RMS error in {len(bad)} of '
          f'{sum(r["method"] == "balanced" for r in H)} balanced rows')
    for r in bad:
        print('   ', r['label'], r['family'], r['set'], r['eta'], r['bond'], r['rms_err'], r['cert_paired'])

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({'font.size': 9, 'axes.labelcolor': TEXT, 'pdf.fonttype': 42})

    def chi_vs_n(alpha, fname, title):
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        style(ax)
        for j, fam in enumerate(FAMS):
            dx = (j - (len(FAMS) - 1) / 2) * 0.09        # dodge: equal ranks would hide each other
            xs = [n + dx for b, n in NBATHS if chi(R, alpha, b, fam) is not None]
            ys = [chi(R, alpha, b, fam) for b, n in NBATHS if chi(R, alpha, b, fam) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, '-', color=COLOR[fam], lw=2, marker=MARKER[fam], ms=6, mec='white', mew=1,
                    label=NAME[fam], zorder=3)
            if alpha == 0.05 and fam in PRED:
                ax.plot([n + dx for n in PRED[fam]], list(PRED[fam].values()), ls='none', marker=MARKER[fam], ms=9,
                        mfc='none', mec=COLOR[fam], mew=0.8, alpha=0.6, zorder=2)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xlabel('number of out-of-band peaks n')
        ax.set_ylabel(r'rank $\chi(\varepsilon = 3\times10^{-3})$, middle cut')
        ax.set_title(title, fontsize=9, color=TEXT, loc='left')
        leg = ax.legend(fontsize=7, frameon=False, loc='upper left')
        for t in leg.get_texts():
            t.set_color(TEXT)
        if alpha == 0.05:
            ax.text(0.98, 0.45, 'open markers:\nStage 0 prediction', transform=ax.transAxes, ha='right',
                    va='center', fontsize=7, color=MUTED)
        else:
            ax.text(0.98, 0.45, 'n = 4, 6: PTs not built\n(rebuild pending)', transform=ax.transAxes,
                    ha='right', va='center', fontsize=7, color=MUTED)
        fig.tight_layout()
        fig.savefig(os.path.join(a.figs, fname))
        plt.close(fig)

    chi_vs_n(0.05, 'fig2_a.pdf', r'(a) $\alpha = 0.05$, $\theta = 10^{-6}$')
    chi_vs_n(0.2, 'fig2_b.pdf', r'(b) $\alpha = 0.2$, $\theta = 10^{-6}$')

    # (c) controls: small multiples (rank, exponent scale), one axis each
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    width = 0.8 / len(FAMS)
    for k, (ax, what) in enumerate(zip(axs, ('rank', 'scale'))):
        style(ax)
        for j, fam in enumerate(FAMS):
            if what == 'scale' and fam == 'flat':
                continue
            vals = [R[(0.05, b, THETA, fam, EPS)][0 if what == 'rank' else 1] for b in CONTROLS]
            x = np.arange(len(CONTROLS)) + (j - (len(FAMS) - 1) / 2) * width
            ax.bar(x, vals, width * 0.85, color=COLOR[fam], label=NAME[fam], zorder=3)
        ax.set_xticks(np.arange(len(CONTROLS)))
        ax.set_xticklabels(['Ohmic', 'edge 5', 'edge 3', 'in-band 0.8'])
        if what == 'rank':
            ax.set_ylabel(r'$\chi(3\times10^{-3})$, middle cut')
            ax.set_title(r'(c) controls, $\alpha = 0.05$: ranks', fontsize=9, loc='left', color=TEXT)
        else:
            ax.set_yscale('log')
            ax.set_ylabel(r'family exponent scale $\|B_F\|$ (Stage 0 proxy)')
            ax.set_title('exponent scales', fontsize=9, loc='left', color=TEXT)
    leg = axs[0].legend(fontsize=7, frameon=False, loc='upper left')
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(a.figs, 'fig2_c.pdf'))
    plt.close(fig)

    # (d) held-out error vs bond at n = 6, one panel per family
    fams = NARROW + ['bernoulli:0.01']
    fig, axs = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    series = [('balanced', 0.0, 'balanced, η = 0', '#2a78d6', 'o'),
              ('balanced', 0.3, 'balanced, η = 0.3', '#1baf7a', 'D'),
              ('svd', -1.0, 'SVD', '#eb6834', 's')]
    for ax, fam in zip(axs, fams):
        style(ax)
        rows = [r for r in H if r['label'] == 'oob6_a0.05_thr1e-6' and r['family'] == fam and r['set'] == 'in']
        bonds = sorted({r['bond'] for r in rows})
        for meth, eta, lab, col, mk in series:
            y = [next(r['rms_err'] for r in rows if r['method'] == meth and r['eta'] == eta and r['bond'] == b)
                 for b in bonds]
            ax.plot(bonds, y, '-', color=col, lw=2, marker=mk, ms=6, mec='white', mew=1, label=lab, zorder=3)
        for key, ls, lab in (('cert_stepindep', '--', 'step-indep. certificate (η = 0)'),
                             ('cert_paired', ':', 'paired certificate (η = 0)')):
            y = [next(r[key] for r in rows if r['method'] == 'balanced' and r['eta'] == 0.0 and r['bond'] == b)
                 for b in bonds]
            ax.plot(bonds, y, ls, color='#2a78d6', lw=1.2, label=lab, zorder=2)
        thr = EPS * rows[0]['rms_y']
        ax.axhline(thr, color=MUTED, lw=1, ls='-.', zorder=1, label=r'$\varepsilon\,$RMS(y)')
        c = chi(R, 0.05, 'oob6', fam)
        ax.axvline(c + 2, color=GRID, lw=6, zorder=0)
        ax.text(c + 2, 0.97, r' $\chi_{op}+2$', transform=ax.get_xaxis_transform(), fontsize=7,
                color=MUTED, va='top')
        ax.set_yscale('log')
        ax.set_xlabel('bond dimension')
        ax.set_title(NAME[fam] + ('  (leaky contrast)' if fam == 'bernoulli:0.01' else ''), fontsize=8,
                     loc='left', color=TEXT)
    axs[0].set_ylabel('held-out RMS outcome error (in family)')
    h, l = axs[0].get_legend_handles_labels()
    leg = fig.legend(h, l, fontsize=7, frameon=False, loc='lower center', ncol=6)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.suptitle(r'(d) closing the loop at n = 6, $\alpha = 0.05$, $\theta = 10^{-6}$', fontsize=9, x=0.01,
                 ha='left', color=TEXT)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(a.figs, 'fig2_d.pdf'))
    plt.close(fig)
    print('\nfigures written to', a.figs)


if __name__ == '__main__':
    main()
