"""
Study A pilot / grid driver (docs/studyA_spec.md, Sec. 3-5; deviations in studyA/CHANGES.md).

  python studyA/run_pilot.py spectra  --baths ohmic oob2 --thetas 1e-7 1e-9 [--jobs 4]
  python studyA/run_pilot.py ranks    --baths ohmic oob2 --thetas 1e-7 1e-9
  python studyA/run_pilot.py loop     --baths ohmic oob2 --theta 1e-7 [--families ...] [--jobs 4]

PT files: studyA/pts/<bath>_a0.05_N200_thr<theta>.pt, cut from dnc256_<bath>_<theta>.pt with
tools/PTB_truncate_time (built by the matching dnc256_<bath>_<theta>.param in studyA/pts).
Outputs: studyA/results/spectra/<bath>_a0.05_N200_dt0.1_thr<theta>_<family>.csv (cut,i,sigma),
studyA/results/ranks.csv, studyA/results/heldout.csv.
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, 'tools')
PTS = os.path.join(HERE, 'pts')
RES = os.path.join(HERE, 'results')
SPEC = os.path.join(RES, 'spectra')
sys.path.insert(0, os.path.join(ROOT, 'tools', 'studyA'))
sys.path.insert(0, HERE)

ALPHA, N, DT = 0.05, 200, 0.1
MID = N // 2 - 1                      # cut 99: 100 steps of past (Stage 0: cut N//2)
FAMILIES = ['bernoulli:0.002', 'bernoulli:0.005', 'bernoulli:0.01', 'coarse:10:0.3', 'coarse:20:0.3']
NARROW = ['bernoulli:0.002', 'bernoulli:0.005', 'coarse:20:0.3']
EPS = [1e-2, 3e-3, 1e-3]
PROXY_NAME = {'bernoulli:0.002': 'p=.002', 'bernoulli:0.005': 'p=.005', 'bernoulli:0.01': 'p=.01',
              'coarse:10:0.3': 'coarse10', 'coarse:20:0.3': 'coarse20', 'flat': 'HS'}
PROXY_BATH = {'ohmic': '0 out-of-band peaks', 'oob2': '2 out-of-band peaks', 'oob4': '4 out-of-band peaks',
              'oob6': '6 out-of-band peaks', 'edge5': 'edge peak at 5', 'edge3': 'edge peak at 3',
              'inband0.8': 'in-band peak at 0.8'}


def label(fam):
    return fam.replace(':', '_')


def pt_file(bath, theta):
    return os.path.join(PTS, f'{bath}_a{ALPHA}_N{N}_thr{theta}.pt')


def spectra_prefix(bath, theta):
    return os.path.join(SPEC, f'{bath}_a{ALPHA}_N{N}_dt{DT}_thr{theta}')


def run(cmd, log):
    t0 = time.time()
    with open(log, 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f'failed ({r.returncode}): {" ".join(cmd)}; see {log}')
    return time.time() - t0


def ensure_pt(bath, theta):
    out = pt_file(bath, theta)
    if not os.path.exists(out):
        src = os.path.join(PTS, f'dnc256_{bath}_{theta}.pt')
        if not os.path.exists(src):
            raise FileNotFoundError(f'{src} missing: build it with bin/ACE {src[:-3]}.param')
        run([os.path.join(TOOLS, 'PTB_truncate_time'), '-read_PT', src, '-write_PT', out, '-n_tot', str(N)],
            out + '.truncate.log')
    return out


def cmd_spectra(a):
    os.makedirs(SPEC, exist_ok=True)
    jobs = []
    for bath in a.baths:
        for theta in a.thetas:
            pt = ensure_pt(bath, theta)
            for fam in FAMILIES + ['flat']:
                out = spectra_prefix(bath, theta) + f'_{label(fam)}.csv'
                if os.path.exists(out) and not a.force:
                    continue
                jobs.append((bath, theta, fam, [os.path.join(TOOLS, 'PTB_memory_spectra'), '-read_PT', pt,
                                                '-family', fam, '-outprefix', spectra_prefix(bath, theta)],
                             out[:-4] + '.log'))
    times = []
    with ThreadPoolExecutor(a.jobs) as ex:
        for (bath, theta, fam, _, _), t in zip(jobs, ex.map(lambda j: run(j[3], j[4]), jobs)):
            times.append((bath, theta, fam, t))
            print(f'spectra {bath} {theta} {fam}: {t:.1f} s', flush=True)
    with open(os.path.join(RES, 'runtimes_spectra.csv'), 'a') as f:
        for r in times:
            f.write(','.join(map(str, r)) + '\n')


def read_spectra(fname):
    d = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
    out = {}
    for cut in np.unique(d[:, 0]).astype(int):
        s = d[d[:, 0] == cut, 2]
        out[cut] = np.sort(s)[::-1]
    return out


def chi(s, eps):
    """min{m : sum_{i>m} s_i^2 <= eps^2 sum_i s_i^2} (spec Sec. 2, Eq. (4) of the Letter)."""
    s2 = np.asarray(s, float)**2
    tail = np.concatenate([np.cumsum(s2[::-1])[::-1], [0.]])   # tail[m] = sum_{i>=m}
    return int(np.argmax(tail <= eps**2 * s2.sum()))


def exponent_scale(bath, fam):
    import stage0_proxy as P
    B = P.blocks(ALPHA, P.BATHS[PROXY_BATH[bath]], N)
    return float(np.linalg.norm(B[PROXY_NAME[fam]]))


def cmd_ranks(a):
    rows = []
    for bath in a.baths:
        for theta in a.thetas:
            for fam in FAMILIES + ['flat']:
                f = spectra_prefix(bath, theta) + f'_{label(fam)}.csv'
                if not os.path.exists(f):
                    print('missing', f)
                    continue
                sp = read_spectra(f)
                es = exponent_scale(bath, fam)
                for eps in EPS:
                    ranks = {c: chi(s, eps) for c, s in sp.items()}
                    cmax = max(ranks, key=ranks.get)
                    for cut_label, c in (('mid', MID), ('max', cmax)):
                        s = sp[c]
                        rows.append(dict(bath=bath, alpha=ALPHA, N=N, dt=DT, theta=theta, cut=cut_label,
                                         cut_index=c, family=fam, eps=eps, chi=ranks[c],
                                         sum_s2=float(np.sum(s**2)), exponent_scale=es))
    out = os.path.join(RES, 'ranks.csv')
    keys = ['bath', 'alpha', 'N', 'dt', 'theta', 'cut', 'cut_index', 'family', 'eps', 'chi', 'sum_s2',
            'exponent_scale']
    old = []
    if os.path.exists(out):
        with open(out) as f:
            old = [r for r in csv.DictReader(f)
                   if not (r['bath'] in a.baths and r['theta'] in a.thetas)]
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in old + rows:
            w.writerow({k: r[k] for k in keys})
    # console table: middle cut, all eps
    print(f'{"bath":8s} {"theta":6s} ' + ' '.join(f'{PROXY_NAME[f]:>10s}' for f in ['flat'] + FAMILIES))
    for bath in a.baths:
        for theta in a.thetas:
            cells = []
            for fam in ['flat'] + FAMILIES:
                v = [r['chi'] for r in rows if r['bath'] == bath and r['theta'] == theta
                     and r['family'] == fam and r['cut'] == 'mid']
                cells.append('/'.join(map(str, v)) if v else '-')
            print(f'{bath:8s} {theta:6s} ' + ' '.join(f'{c:>10s}' for c in cells))


def cmd_loop(a):
    os.makedirs(RES, exist_ok=True)
    out = os.path.join(RES, 'heldout.csv')
    jobs = []
    for bath in a.baths:
        pt = ensure_pt(bath, a.theta)
        for fam in a.families:
            sp = read_spectra(spectra_prefix(bath, a.theta) + f'_{label(fam)}.csv')
            c = chi(sp[MID], 3e-3)
            maxks = sorted({c + 2, c + 4, c + 8, 16})
            tmp = os.path.join(RES, f'heldout_{bath}_thr{a.theta}_{label(fam)}.csv')
            if os.path.exists(tmp) and not a.force:
                continue
            if os.path.exists(tmp):
                os.remove(tmp)
            cmd = [os.path.join(TOOLS, 'PTB_family_truncate'), '-read_PT', pt, '-family', fam,
                   '-csv', tmp, '-label', f'{bath}_thr{a.theta}', '-n_heldout', str(a.n_heldout),
                   '-eta', '0', '-eta', '0.3']
            for m in maxks:
                cmd += ['-maxk', str(m)]
            jobs.append((bath, fam, cmd, tmp[:-4] + '.log', c))
    with ThreadPoolExecutor(a.jobs) as ex:
        for (bath, fam, _, _, c), t in zip(jobs, ex.map(lambda j: run(j[2], j[3]), jobs)):
            print(f'loop {bath} {fam} (chi_op(3e-3) = {c}): {t:.1f} s', flush=True)
            with open(os.path.join(RES, 'runtimes_loop.csv'), 'a') as f:
                f.write(f'{bath},{a.theta},{fam},{t}\n')
    # merge per-job files
    files = sorted(f for f in os.listdir(RES) if f.startswith('heldout_') and f.endswith('.csv'))
    with open(out, 'w') as fo:
        for i, fn in enumerate(files):
            with open(os.path.join(RES, fn)) as fi:
                lines = fi.readlines()
            fo.writelines(lines if i == 0 else lines[1:])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['spectra', 'ranks', 'loop'])
    ap.add_argument('--baths', nargs='+', default=['ohmic', 'oob2'])
    ap.add_argument('--thetas', nargs='+', default=['1e-7'])
    ap.add_argument('--theta', default='1e-7')
    ap.add_argument('--families', nargs='+', default=NARROW + ['bernoulli:0.01'])
    ap.add_argument('--n_heldout', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    {'spectra': cmd_spectra, 'ranks': cmd_ranks, 'loop': cmd_loop}[a.what](a)
