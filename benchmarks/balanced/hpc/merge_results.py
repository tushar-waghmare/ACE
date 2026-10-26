"""Merge <root>/seed_<s>/maxk_<k>/results.csv (run_seed_parallel.sh) into <root>/results.csv.

Safe to run while other array tasks are still writing: the merged file is replaced atomically and
lists only finished (seed, maxk) runs.
"""
import csv
import glob
import os
import sys


def main(root):
    parts = sorted(glob.glob(os.path.join(root, 'seed_*', 'maxk_*', 'results.csv')))
    rows = []
    for p in parts:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    if not rows:
        print(f'no finished runs under {root}')
        return
    rows.sort(key=lambda r: (int(r['seed']), int(r['maxk']), r['method'] != 'svd', r['eta'], r['set']))
    tmp = os.path.join(root, f'.results.csv.{os.getpid()}')
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, os.path.join(root, 'results.csv'))
    done = sorted({(int(r['seed']), int(r['maxk'])) for r in rows})
    print(f'merged {len(parts)} runs ({len(rows)} rows) into {os.path.join(root, "results.csv")}: {done}')


if __name__ == '__main__':
    main(sys.argv[1])
