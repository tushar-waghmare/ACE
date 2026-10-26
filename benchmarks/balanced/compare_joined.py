"""Joined-PT validation (plan Section 7, 2026-10-25): bench30_thr1e-6 vs the exact join of its two
15-mode halves (joined/halves_joined_b10.pt, D = 550), same contexts (mixed family, seed 1).

Input: results/joined_halves_mixed/{results.csv,runs/} and results/bench30_thr1e-6_mixed/
{results.csv,seed_1/runs/} (run outputs are not in git; rerun run_benchmark.py to regenerate).
Prints (1) rms_final joined / bench30, each against its own untruncated reference, (2) the
SVD / balanced(eta=0.3) ratio in both, (3) both arms scored against the common joined reference,
next to the difference between the two untruncated PTs (the input model difference).
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
J_DIR = os.path.join(HERE, 'results', 'joined_halves_mixed')
B_DIR = os.path.join(HERE, 'results', 'bench30_thr1e-6_mixed')
MAXK = [4, 6, 8, 12, 16, 24]
METHODS = [('svd', None)] + [('balanced', e) for e in (0., 0.1, 0.3, 0.5, 1.)]
SETS = ('test', 'ood')


def load(path, seed=None):
    with open(path) as f:
        return {(r['method'], int(r['maxk']), float(r['eta']) if r['eta'] else None, r['set']): float(r['rms_final'])
                for r in csv.DictReader(f) if seed is None or r['seed'] == seed}


def final_obs(path):
    last = None
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) >= 7:
                last = complex(float(v[1]), float(v[2]))
    return last


def rms(values):
    return math.sqrt(sum(abs(x) ** 2 for x in values) / len(values))


def main():
    j = load(os.path.join(J_DIR, 'results.csv'))
    b = load(os.path.join(B_DIR, 'results.csv'), '1')
    for s in SETS:
        print(f'\n{s}: rms_final joined / bench30 (own references), m = {MAXK}')
        for m, e in METHODS:
            print(f"  {m:8s} {'' if e is None else e!s:4s} "
                  + ' '.join(f'{j[(m, k, e, s)] / b[(m, k, e, s)]:5.2f}' for k in MAXK))
        for label, d in (('joined ', j), ('bench30', b)):
            print(f'  SVD/bal(0.3) {label}: '
                  + ' '.join(f"{d[('svd', k, None, s)] / d[('balanced', k, 0.3, s)]:6.2f}" for k in MAXK))

    j_runs, b_runs = os.path.join(J_DIR, 'runs'), os.path.join(B_DIR, 'seed_1', 'runs')
    if not os.path.isdir(b_runs):
        print(f'\n{b_runs} missing: skipping the common-reference comparison')
        return
    for s in SETS:
        ctx = sorted(f[4:-4] for f in os.listdir(j_runs) if f.startswith('ref_' + s))
        ref = {c: final_obs(os.path.join(j_runs, f'ref_{c}.out')) for c in ctx}
        model = rms([final_obs(os.path.join(b_runs, f'ref_{c}.out')) - ref[c] for c in ctx])
        print(f'\n{s}: rms_final against the common joined reference (joined arm | bench30 arm); '
              f'untruncated bench30 vs joined: {model:.1e}')
        for tag, label in (('svd_{k}.pt', 'svd'), ('balanced_{k}_0.3.pt', 'bal 0.3'), ('balanced_{k}_1.0.pt', 'bal 1.0')):
            cells = []
            for k in MAXK:
                t = tag.format(k=k)
                ej = rms([final_obs(os.path.join(j_runs, f'{t}_{c}.out')) - ref[c] for c in ctx])
                eb = rms([final_obs(os.path.join(b_runs, f'{t}_{c}.out')) - ref[c] for c in ctx])
                cells.append(f'm={k}: {ej:.1e}|{eb:.1e}')
            print(f'  {label:8s}' + '  '.join(cells))


if __name__ == '__main__':
    main()
