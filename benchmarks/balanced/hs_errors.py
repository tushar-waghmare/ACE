"""
Backfill HS errors ||T - T_trunc||_HS / ||T||_HS for finished run_benchmark.py results.

    python3 benchmarks/balanced/hs_errors.py [--jobs 8] [results/bench100_mixed ...]

The truncated PTs are not kept by run_benchmark.py; they are regenerated here (SVD: input PT,
backward sweep without truncation, forward sweep with compress_maxk; balanced: the seed's
training contexts from <dir>/seed_<s>/ctx) and compared with tools/PTB_HS_distance. Writes
<dir>/hs_errors.csv (seed, method, maxk, eta, hs_rel, maxdim, maxdim_recorded).
"""
import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
TOOLS = os.path.join(ROOT, 'tools')


def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return p.stdout


def hs_rel(pt_in, pt):
    out = run([os.path.join(TOOLS, 'PTB_HS_distance'), '-read_PT', pt_in, '-compare_PT', pt])
    return float(next(l.split()[1] for l in out.splitlines() if l.startswith('relative')))


def maxdim_of(pt):
    out = run([os.path.join(TOOLS, 'PTB_analyze'), '-read_PT', pt])
    return int(next(l.split()[1] for l in out.splitlines() if l.startswith('Maxdim')))


def svd_job(pt_in, maxk):
    tmp = tempfile.mkdtemp(dir=HERE, prefix='_tmp_hs_')
    try:
        pt = os.path.join(tmp, 'svd.pt')
        shutil.copyfile(pt_in, pt)
        run([os.path.join(TOOLS, 'PTB_sweep_backward'), '-read_PT', pt, '-threshold', '0', '-verbosity', '0'])
        run([os.path.join(TOOLS, 'PTB_sweep_forward'), '-read_PT', pt, '-threshold', '0',
             '-compress_maxk', str(maxk), '-verbosity', '0'])
        return hs_rel(pt_in, pt), maxdim_of(pt)
    finally:
        shutil.rmtree(tmp)


def balanced_job(pt_in, ctx, maxk, eta, stride, rhat):
    tmp = tempfile.mkdtemp(dir=HERE, prefix='_tmp_hs_')
    try:
        pt = os.path.join(tmp, 'bal.pt')
        cmd = [os.path.join(TOOLS, 'PTB_balanced_truncate'), '-read_PT', pt_in, '-write_PT', pt,
               '-balanced_maxk', str(maxk), '-balanced_eta', str(eta), '-balanced_rhat', rhat,
               '-balanced_readout_stride', str(stride), '-verbosity', '0']
        for c in ctx:
            cmd += ['-add_balanced_context', c]
        run(cmd)
        return hs_rel(pt_in, pt), maxdim_of(pt)
    finally:
        shutil.rmtree(tmp)


def pt_of(resdir):
    name = os.path.basename(os.path.normpath(resdir))
    stem = name.rsplit('_', 1)[0]   # bench100_mixed -> bench100
    return os.path.join(HERE, f'{stem}.pt')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--jobs', type=int, default=8)
    a = ap.parse_args()
    dirs = a.dirs or sorted(d for d in glob.glob(os.path.join(HERE, 'results', 'bench*')) if os.path.isdir(d))

    svd_cache = {}
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for d in dirs:
            pt_in = pt_of(d)
            rows = list(csv.DictReader(open(os.path.join(d, 'results.csv'))))
            configs = {}
            for r in rows:
                key = (int(r['seed']), r['method'], int(r['maxk']), r['eta'])
                configs[key] = r
            futures = {}
            for (seed, method, maxk, eta), r in sorted(configs.items()):
                if method == 'svd':
                    ck = (pt_in, maxk)
                    if ck not in svd_cache:
                        svd_cache[ck] = ex.submit(svd_job, pt_in, maxk)
                    futures[(seed, method, maxk, eta)] = svd_cache[ck]
                else:
                    ctx = sorted(glob.glob(os.path.join(d, f'seed_{seed}', 'ctx', 'train_*.param')))
                    futures[(seed, method, maxk, eta)] = ex.submit(
                        balanced_job, pt_in, ctx, maxk, float(eta), int(r['readout_stride']), r['rhat'])
            out = []
            for key, f in sorted(futures.items()):
                h, md = f.result()
                out.append(dict(seed=key[0], method=key[1], maxk=key[2], eta=key[3], hs_rel=h,
                                maxdim=md, maxdim_recorded=int(configs[key]['maxdim'])))
            with open(os.path.join(d, 'hs_errors.csv'), 'w', newline='') as fo:
                w = csv.DictWriter(fo, fieldnames=list(out[0].keys()))
                w.writeheader()
                w.writerows(out)
            bad = [o for o in out if o['maxdim'] != o['maxdim_recorded']]
            print(f"{d}: {len(out)} rows, maxdim mismatches: {len(bad)}", flush=True)


if __name__ == '__main__':
    main()
