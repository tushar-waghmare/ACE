"""
Add trace metrics relative to the reference PT to existing run_benchmark.py results.

    python3 benchmarks/balanced/reanalyze.py results/bench30_thr1e-6_mixed [...]

The input PT is itself SVD-truncated, so |tr rho - 1| has a floor set by the reference.
This adds, per row of results.csv,
  max_trace_dev_vs_ref : max over contexts and times of |tr rho(t) - tr rho_ref(t)|
  ref_max_trace_err    : max |tr rho_ref(t) - 1| of the reference PT (same contexts)
computed from the saved simulation outputs (runs/*.out). results.csv is rewritten in place.
"""
import csv
import os
import sys


def read_trace(path):
    tr = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) < 7:
                continue
            tr.append(complex(float(v[3]) + float(v[5]), float(v[4]) + float(v[6])))
    return tr


def seed_dirs(root):
    subs = sorted(d for d in os.listdir(root) if d.startswith('seed_'))
    return [(int(d[5:]), os.path.join(root, d)) for d in subs] if subs else [(None, root)]


def analyze(root):
    path = os.path.join(root, 'results.csv')
    rows = list(csv.DictReader(open(path)))
    metrics = {}
    for seed, d in seed_dirs(root):
        ctx = sorted(os.listdir(os.path.join(d, 'ctx')))
        splits = {s: [c for c in ctx if c.startswith(s + '_')] for s in ('train', 'test', 'ood')}
        runs = os.path.join(d, 'runs')
        ref = {c: read_trace(os.path.join(runs, f'ref_{c}.out')) for c in ctx}

        def stats(tag, names):
            dev = 0.0
            for c in names:
                for a, b in zip(read_trace(os.path.join(runs, f'{tag}_{c}.out')), ref[c]):
                    dev = max(dev, abs(a - b))
            return dev
        for r in rows:
            if seed is not None and int(r['seed']) != seed:
                continue
            if r['method'] == 'svd':
                tag = f"svd_{r['maxk']}.pt"
            else:
                tag = f"balanced_{r['maxk']}_{float(r['eta'])}.pt"
            names = splits[r['set']]
            key = (seed, tag, r['set'])
            if key not in metrics:
                ref_err = max(abs(x - 1) for c in names for x in ref[c])
                metrics[key] = (stats(tag, names), ref_err)
            r['max_trace_dev_vs_ref'], r['ref_max_trace_err'] = metrics[key]
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"updated {path} ({len(rows)} rows)")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        analyze(p)
