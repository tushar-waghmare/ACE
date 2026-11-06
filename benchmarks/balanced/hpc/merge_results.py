"""Merge the per-(seed, maxk) results under <root>/seed_<s>/maxk_<k>/ into <root>/results.csv.

A run directory holds either results.csv (whole run: run_seed_parallel.sh / --stage all) or
results_svd.csv and/or results_balanced.csv (grid stages); results.csv takes precedence.
Safe to run while other array tasks are still writing: the merged file is replaced atomically and
lists only finished parts. Prints which (seed, maxk) have both arms.
"""
import csv
import glob
import os
import sys


def main(root):
    rows, arms = [], {}
    for d in sorted(glob.glob(os.path.join(root, 'seed_*', 'maxk_*'))):
        whole = os.path.join(d, 'results.csv')
        parts = [whole] if os.path.exists(whole) else \
            [p for p in (os.path.join(d, 'results_svd.csv'), os.path.join(d, 'results_balanced.csv')) if os.path.exists(p)]
        for p in parts:
            with open(p) as f:
                part = list(csv.DictReader(f))
            rows += part
            for r in part:
                arms.setdefault((int(r['seed']), int(r['maxk'])), set()).add(r['method'])
    if not rows:
        print(f'no finished runs under {root}')
        return
    fields = list(rows[0].keys())
    rows.sort(key=lambda r: (int(r['seed']), int(r['maxk']), r['method'] != 'svd', r['eta'], r['set']))
    tmp = os.path.join(root, f'.results.csv.{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, os.path.join(root, 'results.csv'))
    complete = sorted(k for k, v in arms.items() if v >= {'svd', 'balanced'})
    partial = sorted((k, sorted(v)) for k, v in arms.items() if not v >= {'svd', 'balanced'})
    print(f'merged {len(rows)} rows into {os.path.join(root, "results.csv")}: complete (seed, maxk) {complete}'
          + (f'; partial {partial}' if partial else ''))


if __name__ == '__main__':
    main(sys.argv[1])
