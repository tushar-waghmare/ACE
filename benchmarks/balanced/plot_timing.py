"""
Runtime figures from profile_timing.py (phases.csv, baseline.csv).

    ~/.venvs/ace/bin/python benchmarks/balanced/plot_timing.py [--indir results/timing] [--outdir plots/timing]

Writes (PNG + PDF):
  timing_breakdown.*   where the time of PTB_balanced_truncate goes (default settings:
                       16 contexts, maxk 16, eta 0.3, readout stride 5), per input PT
  timing_scaling.*     time per phase group vs number of contexts, maxk, eta, readout stride
  cost_comparison.*    balanced vs plain ACE (SVD sweeps) truncation time vs maxk, and both
                       next to the PT build time and one simulation with the input / truncated PT
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict

import plotstyle as ps
from plotstyle import plt

HERE = os.path.dirname(os.path.abspath(__file__))

# phase groups (stack order = execution order); raw phase names from PTB_balanced_truncate
GROUPS = [
    ('future factors (backward pass)', ['future', 'hs_future']),
    ('past propagation', ['past']),
    ('past factor, rhat, blend', ['factors']),
    ('core SVD → X, Y', ['core_svd']),
    ('diagnostics (norms, bounds, E_aug)', ['noise_floor', 'proj_diag', 'diagnostics']),
    ('insert X, Y into PT', ['apply']),
    ('I/O and setup', ['contexts', 'read_PT', 'write_PT', 'setup']),
]
COLORS = ps.SERIES[:6] + ['#a8a7a2']


def load_phases(path):
    """(scan, pt, n_ctx, maxk, eta, stride) -> {phase: median seconds over repeats}"""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(path)):
        key = (r['scan'], r['pt'], int(r['n_ctx']), int(r['maxk']), float(r['eta']), int(r['stride']))
        acc[key][r['phase']].append(float(r['seconds']))
    return {k: {p: statistics.median(v) for p, v in d.items()} for k, d in acc.items()}


def grouped(ph):
    return [sum(ph.get(p, 0.) for p in names) for _, names in GROUPS]


def fig_breakdown(phases, out):
    rows = sorted([(k, v) for k, v in phases.items() if k[0] == 'pt'], key=lambda kv: kv[1]['total'])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.4, 0.75 * len(rows) + 1.9), layout='constrained',
                                  gridspec_kw=dict(width_ratios=[1.35, 1]))
    labels = [k[1] for k, _ in rows]
    for side, (axis, frac) in enumerate(((ax, False), (ax2, True))):
        left = [0.] * len(rows)
        for gi, (gname, _) in enumerate(GROUPS):
            vals = []
            for (_, ph) in rows:
                g = grouped(ph)
                vals.append(g[gi] / sum(g) * 100 if frac else g[gi])
            axis.barh(labels, vals, left=left, color=COLORS[gi], height=0.6, label=gname,
                      edgecolor=ps.SURFACE, linewidth=1.2)
            if frac:
                for y, (l0, v) in enumerate(zip(left, vals)):
                    if v >= 7:
                        axis.text(l0 + v / 2, y, f'{v:.0f}%', ha='center', va='center', fontsize=7.5,
                                  color='white' if gi not in (6,) else ps.INK)
            left = [a + b for a, b in zip(left, vals)]
        if not frac:
            for y, (_, ph) in enumerate(rows):
                axis.text(left[y], y, f'  {ph["total"]:.1f} s', va='center', color=ps.INK, fontsize=8.5)
            axis.set_xlabel('wall-clock time [s]')
            axis.set_xlim(0, max(left) * 1.18)
            axis.set_title('absolute')
        else:
            axis.set_xlabel('share of total [%]')
            axis.set_xlim(0, 100)
            axis.set_yticklabels([])
            axis.set_title('relative')
        axis.grid(axis='y', visible=False)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=4)
    fig.suptitle('Where the time goes in balanced truncation (16 contexts, maxk 16, η=0.3, '
                 'readout stride 5; median of repeats)', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'timing_breakdown'))


SCANS = [('n_ctx', 'number of training contexts', 2, True),
         ('maxk', 'max. bond dimension (maxk)', 3, False),
         ('eta', 'blend weight η', 4, False),
         ('stride', 'readout stride (0 = final time only)', 5, False)]


def fig_scaling(phases, out):
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4), layout='constrained', sharey=True)
    for ax, (scan, xlabel, idx, logx) in zip(axes, SCANS):
        pts = sorted([(k[idx], v) for k, v in phases.items() if k[0] == scan])
        if not pts:
            continue
        xs = [x for x, _ in pts]
        g = [grouped(v) for _, v in pts]
        ax.plot(xs, [v['total'] for _, v in pts], color=ps.INK, ls='--', marker='s', label='total')
        for gi, (gname, _) in enumerate(GROUPS):
            ys = [row[gi] for row in g]
            if max(ys) < 1e-3:
                continue
            ax.plot(xs, ys, color=COLORS[gi], marker='o', label=gname)
        ax.set_yscale('log')
        if logx:
            ax.set_xscale('log', base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{x:g}' for x in xs])
        ax.minorticks_off() if logx else None
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel('wall-clock time [s]')
    handles, labels = {}, []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in handles:
                handles[l] = h
                labels.append(l)
    fig.legend([handles[l] for l in labels], labels, loc='outside lower center', ncol=4)
    pt = next(k[1] for k in phases if k[0] == 'n_ctx')
    fig.suptitle(f'Runtime scaling of balanced truncation, {pt} PT (others fixed at 16 contexts, '
                 'maxk 16, η=0.3, stride 5)', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'timing_scaling'))


def fig_cost(phases, base_path, out, build_times):
    base = list(csv.DictReader(open(base_path)))
    pts = []
    for r in base:
        if r['pt'] not in pts:
            pts.append(r['pt'])
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), layout='constrained',
                             gridspec_kw=dict(width_ratios=[1, 1.15]))
    # left: truncation time vs maxk, both methods, on the PT of the maxk scan
    ax = axes[0]
    scan_pt = next(k[1] for k in phases if k[0] == 'maxk')
    svd = dict(sorted((int(r['maxk']), float(r['seconds'])) for r in base
                      if r['pt'] == scan_pt and r['method'] == 'svd' and r['what'] == 'truncate'))
    bal = dict(sorted((k[3], v['total']) for k, v in phases.items() if k[0] == 'maxk'))
    ax.plot(list(svd), list(svd.values()), label='plain ACE (SVD backward + forward sweep)',
            **ps.METHODS[0][2])
    ax.plot(list(bal), list(bal.values()), label='balanced (16 contexts, η=0.3)', **ps.METHODS[2][2])
    for k in bal:
        if k in svd:
            ax.annotate(f'×{bal[k] / svd[k]:.1f}', (k, bal[k]), textcoords='offset points',
                        xytext=(0, 7), ha='center', fontsize=7.5, color=ps.INK2)
    ax.set_ylim(0, max(bal.values()) * 1.25)
    ax.set_xticks(list(bal))
    ax.set_xlabel('max. bond dimension (maxk)')
    ax.set_ylabel('truncation wall-clock time [s]')
    ax.set_title(f'truncation cost, {scan_pt} PT')
    ax.legend(loc='lower right', fontsize=7.5)

    # right: the whole pipeline for each PT at maxk 16 (log scale)
    ax = axes[1]
    items, colors = [], []
    for pt in pts:
        get = lambda m, w, k='16': next((float(r['seconds']) for r in base if r['pt'] == pt and
                                         r['method'] == m and r['what'] == w and r['maxk'] == k), None)
        bal = next((v['total'] for k, v in phases.items() if k[0] == 'pt' and k[1] == pt), None)
        rows = [('build PT (logged)', build_times.get(pt), '#a8a7a2'),
                ('SVD truncation', get('svd', 'truncate'), ps.BASELINE),
                ('balanced truncation', bal, ps.SERIES[1]),
                ('1 simulation, input PT', get('input PT', 'simulate', ''), '#a8a7a2'),
                ('1 simulation, SVD PT', get('svd', 'simulate'), ps.BASELINE),
                ('1 simulation, balanced PT', get('balanced', 'simulate'), ps.SERIES[1])]
        for name, v, col in rows:
            if v is not None:
                items.append((f'{pt}: {name}', v))
                colors.append(col)
        items.append(('', 0.))
        colors.append(ps.SURFACE)
    items, colors = items[:-1], colors[:-1]
    ys = list(range(len(items)))[::-1]
    ax.barh(ys, [max(v, 1e-9) for _, v in items], color=colors, height=0.7)
    for y, (_, v) in zip(ys, items):
        if v > 0:
            ax.text(v, y, f' {v:.3g} s', va='center', fontsize=7.5, color=ps.INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in items], fontsize=7.5)
    ax.set_xscale('log')
    ax.set_xlim(left=min(v for _, v in items if v > 0) / 3,
                right=max(v for _, v in items) * 30)
    ax.set_xlabel('wall-clock time [s] (log)')
    ax.set_title('pipeline at maxk 16')
    ax.grid(axis='y', visible=False)
    fig.suptitle('Cost of balanced truncation compared with plain ACE', fontsize=10.5)
    ps.save(fig, os.path.join(out, 'cost_comparison'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--indir', default=os.path.join(HERE, 'results', 'timing'))
    ap.add_argument('--outdir', default=os.path.join(HERE, 'plots', 'timing'))
    # single-core build times (builds_done.log; the 3e-7 build ran next to the 1e-6 one)
    ap.add_argument('--build', nargs=2, action='append', metavar=('PT', 'SECONDS'),
                    default=[['30 modes', '1536'], ['30 modes (tight)', '49217']])
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    ps.setup()
    phases = load_phases(os.path.join(a.indir, 'phases.csv'))
    fig_breakdown(phases, a.outdir)
    fig_scaling(phases, a.outdir)
    fig_cost(phases, os.path.join(a.indir, 'baseline.csv'), a.outdir,
             {pt: float(s) for pt, s in a.build})
    print(f'wrote figures to {a.outdir}')


if __name__ == '__main__':
    main()
