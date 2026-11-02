"""
Global error certificate for a forward sweep: correlated (Hamiltonian-family) contexts.

Each context uses ONE system Hamiltonian for all steps, so past and future at a cut are
correlated. The telescoping identity and the paired certificate still hold exactly:

  y(c) - y~(c) = sum_k e_k(c),   e_k(c) = pair(chi_k(c), psi~_k(c) @ (I - Pi_k).T)
  sqrt(E_c|y - y~|^2) <= sum_k sqrt(E_c|e_k(c)|^2)          (paired: same c on both sides)

but the product-ensemble number sum_k sqrt(E_aug,k^prod), built from all (past a, future b)
pairs, is NOT a bound here. This script shows both, on training and held-out contexts.

The sweep uses the same context list for pasts and futures, so the paired terms of the
training set are available at no extra propagation cost.

Run:  python certificate_paired.py      (about 1-2 minutes; exits non-zero on failure)
"""
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    sys.path.insert(0, _p)          # balanced_ref.py next to this file or at the repo root
from balanced_ref import (ToyPT, random_context, run_forward, tester, outcome, pair,
                          balanced_projector, trace_weighted_rhat)

rng = np.random.default_rng(8)
pt = ToyPT(N=10, dt=0.3, n_env=2, seed=3)
train = [random_context(rng) for _ in range(24)]
test = [random_context(rng) for _ in range(30)]


def protected_sweep(m):
    """Forward sweep (Route T, same contexts on both sides). Returns the inserted projectors
    {k: Pi_k} in original bond coordinates, the paired local errors per context, and the
    product-ensemble local errors E_aug,k^prod."""
    projs, paired, prod = {}, {id(c): [] for c in train + test}, []
    for k in range(pt.N - 1):
        Ps = [run_forward(pt, c, k + 1, insert=projs) for c in train]    # hybrid pasts
        Cs = [tester(pt, c, k) for c in train]                            # exact futures
        A = np.hstack([p.T for p in Ps]) / np.sqrt(len(train))
        F = np.vstack(Cs) / np.sqrt(len(train))
        X, Y, _ = balanced_projector(A, F, m, pt.closure[k], trace_weighted_rhat(Ps, pt.NL))
        R = np.eye(pt.D) - X @ Y
        prod.append(np.mean([abs(pair(ch, p @ R.T)) ** 2 for p in Ps for ch in Cs]))
        for c in train + test:
            paired[id(c)].append(pair(tester(pt, c, k), run_forward(pt, c, k + 1, insert=projs) @ R.T))
        projs[k] = X @ Y
    return projs, paired, prod


if __name__ == '__main__':
    ok = True
    print(f"Hamiltonian-family contexts (correlated across steps), D={pt.D}, N={pt.N}")
    print(f"{'m':>2} {'set':>5} | {'max|err - sum e_k|':>18} {'RMS error':>10} {'paired cert':>12} "
          f"{'product estimate':>17} {'product >= RMS?':>16}")
    for m in (3, 4, 6):
        projs, paired, prod = protected_sweep(m)
        prod_est = sum(np.sqrt(t) for t in prod)
        for name, cs in (('train', train), ('test', test)):
            err = np.array([outcome(pt, c) - outcome(pt, c, insert=projs) for c in cs])
            resid = np.max(np.abs(err - np.array([sum(paired[id(c)]) for c in cs])))
            rms = np.sqrt(np.mean(np.abs(err) ** 2))
            cert = sum(np.sqrt(np.mean([abs(paired[id(c)][j]) ** 2 for c in cs]))
                       for j in range(len(prod)))
            print(f"{m:>2} {name:>5} | {resid:>18.1e} {rms:>10.3e} {cert:>12.3e} {prod_est:>17.3e} "
                  f"{'yes' if prod_est >= rms else 'NO':>16}")
            ok &= resid < 1e-10                        # telescoping identity
            ok &= rms <= cert * (1 + 1e-9)             # paired certificate
    print("\nALL CHECKS PASSED (the product estimate is informational, not a bound here)"
          if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
