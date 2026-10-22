"""
Export matrices from the Python reference (balanced_ref.py) for the C++ tests.

    python tools/balanced_ref/export_reference_data.py [outdir]

Default outdir: tests/data/ref (relative to the repository root). Uses the same toy
problem, contexts and cut as test T3 in balanced_ref.run_tests (cut k=4, m=5).

File format (read by tests/test_tools.hpp: read_matrix): first line 'rows cols',
then one line per row with 're im' pairs; vectors are stored as (n x 1).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import balanced_ref as br  # noqa: E402


def write_matrix(path, M):
    M = np.atleast_2d(np.asarray(M, dtype=complex))
    with open(path, 'w') as f:
        f.write(f"{M.shape[0]} {M.shape[1]}\n")
        for row in M:
            f.write(' '.join(f"{z.real:.17e} {z.imag:.17e}" for z in row) + '\n')


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    # identical construction to run_tests() so the data is the T3 problem
    rng = np.random.default_rng(7)
    pt = br.ToyPT(N=10, dt=0.3, n_env=2, seed=3)
    ctx_p = [br.random_context(rng) for _ in range(12)]
    ctx_f = [br.random_context(rng) for _ in range(12)]
    k, m = 4, 5

    A, F = br.past_factor(pt, ctx_p, k), br.future_factor(pt, ctx_f, k)
    q = pt.closure[k]
    rhat = br.trace_weighted_rhat([br.run_forward(pt, cc, k + 1) for cc in ctx_p], pt.NL)

    X, Y, S = br.balanced_projector(A, F, m, q, rhat)
    Xu, Yu, Su = br.balanced_projector(A, F, m)
    Pi, Piu = X @ Y, Xu @ Yu

    # per-context past states at the cut (to test trace_weighted_rhat)
    for a, cc in enumerate(ctx_p):
        write_matrix(os.path.join(outdir, f"state_{a:02d}.txt"), br.run_forward(pt, cc, k + 1))

    write_matrix(os.path.join(outdir, "A.txt"), A)
    write_matrix(os.path.join(outdir, "F.txt"), F)
    write_matrix(os.path.join(outdir, "q.txt"), q[:, None])
    write_matrix(os.path.join(outdir, "rhat.txt"), rhat[:, None])
    write_matrix(os.path.join(outdir, "sigma_protected.txt"), S[:, None])
    write_matrix(os.path.join(outdir, "Pi_protected.txt"), Pi)
    write_matrix(os.path.join(outdir, "sigma_unprotected.txt"), Su[:, None])
    write_matrix(os.path.join(outdir, "Pi_unprotected.txt"), Piu)
    with open(os.path.join(outdir, "scalars.txt"), 'w') as f:
        f.write(f"m {m}\n")
        f.write(f"n_contexts {len(ctx_p)}\n")
        f.write(f"E_abs_protected {br.err_abs(A, F, Pi):.17e}\n")
        f.write(f"E_abs_unprotected {br.err_abs(A, F, Piu):.17e}\n")
    print(f"wrote reference data to {outdir}: D={pt.D}, NL={pt.NL}, A {A.shape}, F {F.shape}, m={m}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join('tests', 'data', 'ref'))
