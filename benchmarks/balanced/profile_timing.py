"""
Runtime profile: plain ACE (SVD sweeps) vs protected balanced truncation.

    python3 benchmarks/balanced/profile_timing.py --outdir benchmarks/balanced/results/timing

(run from anywhere after `make` and `make tools`; stdlib only; runs sequentially so
that timings are not distorted by parallel jobs). Uses the context files written by
run_benchmark.py (results/bench100_mixed/seed_1/ctx: train, test, ood) as a pool of
up to 80 contexts with the te=20, dt=0.1 grid of the bench30/bench100 PTs.

Outputs (in --outdir):
  phases.csv    one row per balanced run and phase (PTB_balanced_truncate -balanced_timing),
                columns: pt, scan, n_ctx, maxk, eta, stride, repeat, phase, seconds
  baseline.csv  SVD baseline (PTB_sweep_backward + PTB_sweep_forward) and simulation
                time of one context with the input / truncated PTs
"""
import argparse
import csv
import glob
import os
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
ACE = os.path.join(ROOT, 'bin', 'ACE')
TOOLS = os.path.join(ROOT, 'tools')


def run(cmd):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return time.perf_counter() - t0


def balanced(pt, ctx, maxk, eta, stride, tmp):
    out_pt, tfile = os.path.join(tmp, 'bal.pt'), os.path.join(tmp, 'timing.csv')
    cmd = [os.path.join(TOOLS, 'PTB_balanced_truncate'), '-read_PT', pt, '-write_PT', out_pt,
           '-balanced_maxk', str(maxk), '-balanced_eta', str(eta),
           '-balanced_readout_stride', str(stride), '-balanced_timing', tfile, '-verbosity', '0']
    for c in ctx:
        cmd += ['-add_balanced_context', c]
    run(cmd)
    with open(tfile) as f:
        return {r['phase']: float(r['seconds']) for r in csv.DictReader(f)}, out_pt


def svd(pt, maxk, tmp):
    out_pt = os.path.join(tmp, 'svd.pt')
    shutil.copyfile(pt, out_pt)
    t = run([os.path.join(TOOLS, 'PTB_sweep_backward'), '-read_PT', out_pt, '-threshold', '0', '-verbosity', '0'])
    t += run([os.path.join(TOOLS, 'PTB_sweep_forward'), '-read_PT', out_pt, '-threshold', '0',
              '-compress_maxk', str(maxk), '-verbosity', '0'])
    return t, out_pt


def simulate(ctx, pt, tmp):
    return run([ACE, ctx, '-add_PT', pt, '-outfile', os.path.join(tmp, 'sim.out')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--ctxdir', default=os.path.join(HERE, 'results', 'bench100_mixed', 'seed_1', 'ctx'))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--quick', action='store_true', help='one repeat, fewer scan points')
    a = ap.parse_args()
    if a.quick:
        a.repeats = 1

    out = os.path.abspath(a.outdir)
    tmp = os.path.join(out, 'tmp')
    os.makedirs(tmp, exist_ok=True)
    pool = (sorted(glob.glob(os.path.join(a.ctxdir, 'train_*.param'))) +
            sorted(glob.glob(os.path.join(a.ctxdir, 'test_*.param'))) +
            sorted(glob.glob(os.path.join(a.ctxdir, 'ood_*.param'))))
    if len(pool) < 64:
        sys.exit(f"need >= 64 context files in {a.ctxdir}, found {len(pool)}")

    pts = {'30 modes': os.path.join(HERE, 'bench30_thr1e-6.pt'),
           '30 modes (tight)': os.path.join(HERE, 'bench30_thr3e-7.pt'),
           '100 modes': os.path.join(HERE, 'bench100.pt')}
    pts = {k: v for k, v in pts.items() if os.path.exists(v)}
    main_pt = '100 modes' if '100 modes' in pts else next(iter(pts))

    # (scan, pt, n_ctx, maxk, eta, stride); default point: 16 contexts, maxk 16, eta 0.3, stride 5
    default = dict(n_ctx=16, maxk=16, eta=0.3, stride=5)
    configs = []
    for n in ([4, 16, 64] if a.quick else [4, 8, 16, 32, 64]):
        configs.append(('n_ctx', main_pt, dict(default, n_ctx=n)))
    for k in ([4, 16, 32] if a.quick else [4, 8, 12, 16, 24, 32]):
        configs.append(('maxk', main_pt, dict(default, maxk=k)))
    for e in [0.0, 0.3, 1.0]:
        configs.append(('eta', main_pt, dict(default, eta=e)))
    for s in [0, 1, 5, 20]:
        configs.append(('stride', main_pt, dict(default, stride=s)))
    for name in pts:
        configs.append(('pt', name, dict(default)))

    phase_rows, base_rows = [], []
    for i, (scan, ptname, c) in enumerate(configs):
        for rep in range(a.repeats):
            ph, _ = balanced(pts[ptname], pool[:c['n_ctx']], c['maxk'], c['eta'], c['stride'], tmp)
            for k, v in ph.items():
                phase_rows.append(dict(pt=ptname, scan=scan, repeat=rep, phase=k, seconds=v, **c))
        print(f"[{i + 1}/{len(configs)}] {scan:6s} {ptname:16s} {c}: total {ph['total']:.2f} s", flush=True)

    # SVD baseline and simulation with full / truncated PTs (one context, maxk scan)
    sim_ctx = pool[0]
    for ptname, pt in pts.items():
        t_full = statistics.median(simulate(sim_ctx, pt, tmp) for _ in range(a.repeats))
        base_rows.append(dict(pt=ptname, method='input PT', maxk='', what='simulate', seconds=t_full))
        for k in ([4, 16, 32] if a.quick else [4, 8, 12, 16, 24, 32]):
            ts, tsim_s, tsim_b = [], [], []
            for _ in range(a.repeats):
                t, svd_pt = svd(pt, k, tmp)
                ts.append(t)
                tsim_s.append(simulate(sim_ctx, svd_pt, tmp))
                _, bal_pt = balanced(pt, pool[:16], k, 0.3, 5, tmp)
                tsim_b.append(simulate(sim_ctx, bal_pt, tmp))
            base_rows.append(dict(pt=ptname, method='svd', maxk=k, what='truncate', seconds=statistics.median(ts)))
            base_rows.append(dict(pt=ptname, method='svd', maxk=k, what='simulate', seconds=statistics.median(tsim_s)))
            base_rows.append(dict(pt=ptname, method='balanced', maxk=k, what='simulate', seconds=statistics.median(tsim_b)))
        print(f"baseline {ptname}: done", flush=True)

    for name, rows in (('phases.csv', phase_rows), ('baseline.csv', base_rows)):
        with open(os.path.join(out, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    shutil.rmtree(tmp)
    print(f"wrote {out}/phases.csv, {out}/baseline.csv")


if __name__ == '__main__':
    main()
