"""Shared matplotlib style for the balanced-truncation benchmark figures."""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
BASELINE = '#52514e'     # plain ACE (SVD): neutral, dashed
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

# plain ACE + the three balanced variants shown everywhere (first 3 slots: all-pairs safe)
METHODS = [
    ('svd', 'plain ACE (SVD)', dict(color=BASELINE, ls='--', marker='s')),
    ('eta=0', 'balanced, η=0', dict(color=SERIES[0], ls='-', marker='o')),
    ('eta=0.3', 'balanced, η=0.3', dict(color=SERIES[1], ls='-', marker='^')),
    ('eta=1', 'balanced, η=1 (HS)', dict(color=SERIES[2], ls='-', marker='D')),
]


def setup():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 9, 'axes.titlesize': 9.5, 'axes.titleweight': 'semibold',
        'legend.frameon': False, 'legend.fontsize': 8.5, 'lines.linewidth': 1.6,
        'lines.markersize': 4.5, 'lines.markeredgecolor': SURFACE, 'lines.markeredgewidth': 0.8,
    })


def gmean(xs):
    xs = [max(x, 1e-300) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def save(fig, path_noext):
    fig.savefig(path_noext + '.png', dpi=200, bbox_inches='tight')
    fig.savefig(path_noext + '.pdf', bbox_inches='tight')
    plt.close(fig)
