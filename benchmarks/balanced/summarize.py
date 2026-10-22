"""
Summarize run_benchmark.py results over seeds.

    python3 benchmarks/balanced/summarize.py results/bench30_thr1e-6_mixed/results.csv [...]

For every (family, set, metric) prints a table maxk x method (SVD, balanced eta=...)
with the geometric mean over seeds (errors span decades) and the min-max range factor,
and the ratio SVD / best balanced. Writes <dir>/summary.csv next to each input.
"""
import csv
import math
import os
import sys
from collections import defaultdict

METRICS = ['rms_final', 'rms_trace', 'max_trace_err', 'max_trace_dev_vs_ref', 'max_population_drift']


def gmean(xs):
    xs = [max(x, 1e-300) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def summarize(path):
    rows = list(csv.DictReader(open(path)))
    groups = defaultdict(list)
    for r in rows:
        method = 'svd' if r['method'] == 'svd' else f"eta={float(r['eta']):g}"
        groups[(r['family'], r['set'], int(r['maxk']), method)].append(r)
    families = sorted({k[0] for k in groups})
    sets = [s for s in ('train', 'test', 'ood') if any(k[1] == s for k in groups)]
    maxks = sorted({k[2] for k in groups})
    methods = ['svd'] + sorted({k[3] for k in groups if k[3] != 'svd'}, key=lambda m: float(m[4:]))

    out_rows = []
    for fam in families:
        for metric in METRICS:
            if metric == 'max_population_drift' and fam != 'dephasing':
                continue
            if metric not in rows[0]:
                continue
            for st in sets:
                if metric == 'max_population_drift' and st != 'test':
                    continue
                print(f"\n[{fam}] {st}: {metric} (geometric mean over seeds)")
                print(f"{'maxk':>5} " + ' '.join(f"{m:>10}" for m in methods) + f" {'svd/best':>9}")
                for mk in maxks:
                    vals = {}
                    for m in methods:
                        g = groups.get((fam, st, mk, m))
                        if not g:
                            continue
                        xs = [float(r[metric]) for r in g]
                        vals[m] = gmean(xs)
                        out_rows.append(dict(family=fam, set=st, metric=metric, maxk=mk, method=m,
                                             gmean=vals[m], min=min(xs), max=max(xs), n_seeds=len(xs)))
                    bal = [v for m, v in vals.items() if m != 'svd']
                    ratio = vals['svd'] / min(bal) if bal and 'svd' in vals else float('nan')
                    print(f"{mk:>5} " + ' '.join(f"{vals.get(m, float('nan')):>10.2e}" for m in methods)
                          + f" {ratio:>9.1f}")
    out = os.path.join(os.path.dirname(path), 'summary.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        summarize(p)
