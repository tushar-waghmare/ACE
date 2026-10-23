"""
Exact reference for the pure-dephasing benchmark (independent boson model, same discretized
modes as the PT) and errors of the input / SVD / balanced PTs against it.

    ~/.venvs/ace/bin/python benchmarks/balanced/exact_dephasing.py [results/bench100_dephasing ...]

For pure dephasing H = E1 |1><1| + sum_k [E_k n_k + hbar g_k |1><1| (a_k + a_k^+)], H_S commutes
with the coupling, so there is no Trotter error and the bath factorizes per mode:
    rho_10(t) = rho_10(0) exp(-i E1 t/hbar) prod_k Tr[exp(-i H1_k t/hbar) rho_th,k exp(i H0_k t/hbar)]
with H0_k = E_k n, H1_k = E_k n + hbar g_k (a + a^+) on M Fock levels and rho_th,k the Gibbs
state on those levels (as ACE's Boson generator). M = Boson_M (3) is the Hamiltonian the PT
represents ("exact_M3"); M = 20 shows the Fock-truncation error of the model itself.
Mode energies/couplings come from ACE (Boson_print_E_g), so the discretization is identical.

Inputs: <dir>/seed_*/ctx/*.param (dephasing contexts: train/test; the OOD set of the dephasing
family has pulses and is skipped) and <dir>/seed_*/runs/<tag>_<ctx>.out (kept by
run_benchmark.py). Output: <dir>/exact_errors.csv with one row per (seed, PT tag, set):
rms_trace_exact (RMS over contexts and times of |<O>(t) - exact|), rms_final_exact, M.
"""
import argparse
import csv
import glob
import math
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
HBAR = 0.6582119569      # meV ps (include/Constants.hpp)
KB = 8.6173303e-2        # meV / K


def ace_modes(n_modes, workdir):
    """(E_k [meV], g_k [1/ps]) exactly as ACE discretizes the QDPhonon bath (bench*_PT.param)."""
    eg = os.path.join(workdir, f'E_g_{n_modes}.dat')
    par = os.path.join(workdir, f'eg_{n_modes}.param')
    with open(par, 'w') as f:
        f.write(f"""ta 0
dt 1e-1
te 0.2
add_Hamiltonian {{-1.5*|1><1|_2}}
Boson_N_modes {n_modes}
Boson_subtract_polaron_shift false
Boson_M 3
Boson_E_max 5
Boson_J_type QDPhonon
Boson_sort none
temperature 4
Boson_print_E_g {eg}
outfile /dev/null
""")
    subprocess.run([os.path.join(ROOT, 'bin', 'ACE'), par], cwd=workdir, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = np.loadtxt(eg)
    return d[:, 0], d[:, 1]


def bath_factor(E, g, M, T, t):
    """prod_k Tr[exp(-i H1 t) rho exp(i H0 t)] on the time grid t (hbar units handled)."""
    a = np.diag(np.sqrt(np.arange(1, M)), 1)
    n = np.diag(np.arange(M)).astype(float)
    fac = np.ones(len(t), dtype=complex)
    for Ek, gk in zip(E, g):
        H0 = Ek * n
        H1 = H0 + HBAR * gk * (a + a.T)
        w = np.exp(-np.arange(M) * Ek / (KB * T))
        rho = np.diag(w / w.sum())
        l1, V1 = np.linalg.eigh(H1)
        e0 = np.diag(H0)
        # Tr[V1 e^{-i l1 t} V1^+ rho e^{i e0 t}] = sum_{j,m} |V1_mj|^2 rho_mm e^{-i (l1_j - e0_m) t}
        W = (np.abs(V1) ** 2) * np.diag(rho)[:, None]      # [m, j]
        phase = np.exp(-1j * (l1[None, None, :] - e0[None, :, None]) * t[:, None, None] / HBAR)
        fac *= np.einsum('mj,tmj->t', W, phase)
    return fac


def parse_context(path):
    txt = open(path).read()
    E1 = float(re.search(r'add_Hamiltonian\s+\{([-+0-9.eE]+)\*\|1><1\|_2\}', txt).group(1))
    # written by run_benchmark.random_pure_state as '(re-im*i)*|1><0|_2' with im signed (.12f)
    m = re.search(r'\(([-+]?[0-9.]+)-([-+]?[0-9.]+)\*i\)\*\|1><0\|_2', txt)
    rho10 = complex(float(m.group(1)), -float(m.group(2)))
    return E1, rho10


def read_out(path):
    t, obs = [], []
    for line in open(path):
        v = line.split()
        if len(v) >= 3:
            t.append(float(v[0]))
            obs.append(complex(float(v[1]), float(v[2])))
    return np.array(t), np.array(obs)


def tag_of(fname, ctx):
    """runs/<tag>_<ctx>.param.out -> (method, maxk, eta)"""
    tag = os.path.basename(fname)[:-len(f'_{ctx}.param.out')]
    if tag == 'ref':
        return 'input', '', ''
    m = re.match(r'(svd|balanced)_(\d+)(?:_([0-9.]+))?\.pt$', tag)
    return m.group(1), int(m.group(2)), (m.group(3) or '')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--M', type=int, nargs='+', default=[3, 20])
    ap.add_argument('--temperature', type=float, default=4.)
    a = ap.parse_args()
    dirs = a.dirs or sorted(glob.glob(os.path.join(HERE, 'results', 'bench*_dephasing')))
    work = tempfile.mkdtemp(prefix='exact_dephasing_')

    for d in dirs:
        n_modes = 100 if 'bench100' in d else 30
        E, g = ace_modes(n_modes, work)
        rows = []
        for seed_dir in sorted(glob.glob(os.path.join(d, 'seed_*'))):
            seed = int(seed_dir.rsplit('_', 1)[1])
            acc = defaultdict(lambda: [0., 0, []])   # (method, maxk, eta, set, M) -> [sum sq, n, final sq]
            for ctx_path in sorted(glob.glob(os.path.join(seed_dir, 'ctx', '*.param'))):
                ctx = os.path.basename(ctx_path)[:-len('.param')]
                split = ctx.split('_')[0]
                if split == 'ood':
                    continue
                E1, rho10 = parse_context(ctx_path)
                t, _ = read_out(os.path.join(seed_dir, 'runs', f'ref_{ctx}.param.out'))
                exact = {M: rho10 * np.exp(-1j * E1 * t / HBAR) * bath_factor(E, g, M, a.temperature, t)
                         for M in a.M}
                for out in glob.glob(os.path.join(seed_dir, 'runs', f'*_{ctx}.param.out')):
                    method, maxk, eta = tag_of(out, ctx)
                    tt, obs = read_out(out)
                    for M, ex in exact.items():
                        err = np.abs(obs - ex)
                        s = acc[(method, maxk, eta, split, M)]
                        s[0] += float(np.sum(err ** 2)); s[1] += len(err); s[2].append(err[-1] ** 2)
            for (method, maxk, eta, split, M), (sq, n, fin) in acc.items():
                rows.append(dict(seed=seed, method=method, maxk=maxk, eta=eta, set=split, M=M,
                                 rms_trace_exact=math.sqrt(sq / n), rms_final_exact=math.sqrt(sum(fin) / len(fin))))
        rows.sort(key=lambda r: (r['seed'], r['set'], r['M'], r['method'], str(r['maxk']), str(r['eta'])))
        with open(os.path.join(d, 'exact_errors.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for M in a.M:
            inp = [r['rms_trace_exact'] for r in rows if r['method'] == 'input' and r['M'] == M]
            print(f"{d}: input PT vs exact (M={M}): rms_trace {min(inp):.2e} .. {max(inp):.2e}")


if __name__ == '__main__':
    main()
