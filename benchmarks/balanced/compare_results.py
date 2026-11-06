"""Compare two benchmark result sets row by row (reproduction check).

    python3 compare_results.py OLD NEW [--rtol 1e-9] [--atol 1e-14]

OLD / NEW: a results CSV or a run directory (results.csv, or results_svd.csv + results_balanced.csv).
Rows are matched on (family, seed, readout_stride, rhat, method, maxk, eta, set); every numeric column
except runtime_s must agree (a column empty in OLD but set in NEW, e.g. certificate_train of a build
from before the certificate, is reported, not compared) to |a - b| <= atol + rtol * max(|a|, |b|), text columns exactly. Exit code 1 on
any mismatch or missing row (rows only in NEW are reported but allowed, e.g. a subset in OLD).
"""
import argparse
import csv
import math
import os
import sys

KEY = ('family', 'seed', 'readout_stride', 'rhat', 'method', 'maxk', 'eta', 'set')
SKIP = {'runtime_s'}


def load(path):
    if os.path.isdir(path):
        files = [os.path.join(path, 'results.csv')]
        if not os.path.exists(files[0]):
            files = [os.path.join(path, f) for f in ('results_svd.csv', 'results_balanced.csv')
                     if os.path.exists(os.path.join(path, f))]
    else:
        files = [path]
    rows = {}
    for fn in files:
        with open(fn) as f:
            for r in csv.DictReader(f):
                r['eta'] = '' if r['eta'] == '' else repr(float(r['eta']))
                rows[tuple(r[k] for k in KEY)] = r
    if not rows:
        sys.exit(f'no rows in {path}')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('old')
    ap.add_argument('new')
    ap.add_argument('--rtol', type=float, default=1e-9)
    ap.add_argument('--atol', type=float, default=1e-14)
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    bad, worst, filled = 0, {}, {}
    for k, ro in sorted(old.items()):
        rn = new.get(k)
        if rn is None:
            print('MISSING in new:', dict(zip(KEY, k)))
            bad += 1
            continue
        for col, vo in ro.items():
            if col in KEY or col in SKIP:
                continue
            vn = rn.get(col)
            if vo == '' and vn not in ('', None):
                filled[col] = filled.get(col, 0) + 1   # e.g. certificate_train of a pre-certificate build
                continue
            try:
                x, y = float(vo), float(vn)
            except (TypeError, ValueError):
                if vo != vn:
                    print(f'DIFF {dict(zip(KEY, k))} {col}: {vo!r} vs {vn!r}')
                    bad += 1
                continue
            if math.isnan(x) and math.isnan(y):
                continue
            d = abs(x - y)
            rel = d / max(abs(x), abs(y)) if max(abs(x), abs(y)) > 0 else 0.
            worst[col] = max(worst.get(col, 0.), rel)
            if not d <= a.atol + a.rtol * max(abs(x), abs(y)):
                print(f'DIFF {dict(zip(KEY, k))} {col}: {x!r} vs {y!r} (rel {rel:.2e})')
                bad += 1
    extra = len(set(new) - set(old))
    print(f'{len(old)} rows compared, {extra} extra rows in new; max relative difference per column: '
          + ', '.join(f'{c} {v:.1e}' for c, v in sorted(worst.items())))
    if filled:
        print('empty in old, set in new (not compared): ' + ', '.join(f'{c} ({n} rows)' for c, n in filled.items()))
    print('REPRODUCED' if bad == 0 else f'NOT REPRODUCED: {bad} difference(s)')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
