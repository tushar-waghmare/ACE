"""
Export reference data for tests/test_studyA.cpp (Study A, docs/studyA_spec.md).

  python studyA/export_test_data.py      (from the repository root)

1. T3 (route R): the toy PT of tools/balanced_ref/certificate_iid.py (exact second moments of a
   step-independent ensemble: 5 random unitaries per half step, 4 initial states, 3 observables).
   Writes the elements, ensemble, absorbed exact Gramians at all cuts of the untruncated chain,
   and for protected balanced sweeps (m = 3, 6) the projectors and per-cut E_abs, E_aug.
2. T2(b) (filter function): exact K_n for the Ohmic bath (cheap, essentially exact PT) at dt = 0.1, sampled pulse sequences and
   the exact Gaussian outcomes from the continuum filter-function integrals (no K_n involved).
File format: tests/test_tools.hpp read_matrix ('rows cols', then 're im' pairs per row).
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'balanced_ref'))
sys.path.insert(0, os.path.join(ROOT, 'studyA'))
OUT = os.path.join(ROOT, 'tests', 'data', 'studyA')


def write_matrix(path, M):
    M = np.atleast_2d(np.asarray(M, dtype=complex))
    with open(path, 'w') as f:
        f.write(f"{M.shape[0]} {M.shape[1]}\n")
        for row in M:
            f.write(' '.join(f"{z.real:.17e} {z.imag:.17e}" for z in row) + '\n')


def export_toy():
    import certificate_iid as C
    d = os.path.join(OUT, 'toy')
    os.makedirs(d, exist_ok=True)
    pt, NL, D, N = C.pt, C.NL, C.D, C.N
    for n in range(N):
        write_matrix(os.path.join(d, f'M_{n:02d}.txt'), pt.M[n].reshape(NL * NL, D * D))
    write_matrix(os.path.join(d, 'rhoE.txt'), C.rE[:, None])
    write_matrix(os.path.join(d, 'closure.txt'), C.q_end[:, None])
    for j, S in enumerate(C.STEPS):
        write_matrix(os.path.join(d, f'step_{j}.txt'), S)
    for a, r in enumerate(C.RHOS):
        write_matrix(os.path.join(d, f'rho_{a}.txt'), r[:, None])
    for b, o in enumerate(C.OBS):
        write_matrix(os.path.join(d, f'obs_{b}.txt'), o[:, None])
    # absorbed exact Gramians of the untruncated chain
    W = np.kron(C.E_RHO, np.outer(C.rE, C.rE.conj()))
    Wf = C.exact_future_grams()
    for k in range(N):
        T = C.Tmat(pt.M[k])
        W = C.avg_fwd(T @ C.avg_fwd(W, D) @ T.conj().T, D)
        write_matrix(os.path.join(d, f'Wp_{k:02d}.txt'), C.ptrace(W))
        write_matrix(os.path.join(d, f'Wf_{k:02d}.txt'), C.ptrace(Wf[k]))
    # protected balanced sweeps with exact second moments
    for m in (3, 6):
        projs = C.sweep(m, 'balanced')
        E = np.array([[ea, eb] for (_, _, eb, ea) in projs])  # columns: E_aug, E_abs
        write_matrix(os.path.join(d, f'sweep_m{m}_errors.txt'), E)
        for (k, Pi, _, _) in projs:
            write_matrix(os.path.join(d, f'sweep_m{m}_Pi_{k:02d}.txt'), Pi)
        rms = np.sqrt(C.exact_global_ms_error(projs))
        write_matrix(os.path.join(d, f'sweep_m{m}_rms.txt'), [[rms]])
    print('toy: N', N, 'D', D, 'NL', NL)


def export_filter_function():
    import baths as B
    d = OUT
    dt, N = 0.1, 30
    Jint = B.InterpJ.from_file(os.path.join(ROOT, 'studyA', 'baths', 'J_ohmic_a0.05.dat'))
    K = B.K_coefficients(Jint, 64, dt=dt)
    B.write_K_file(os.path.join(d, 'K_ff.dat'), K, dt=dt)
    rng = np.random.default_rng(11)
    seqs = [np.zeros(N - 1, int), (rng.random(N - 1) < 0.15).astype(int),
            (rng.random(N - 1) < 0.3).astype(int)]
    seqs[0][[9, 19]] = 1   # CPMG-like: pulses after steps 9 and 19
    W, Wt, J = Jint.nodes()
    c = B.coth(B.BETA * W / 2)
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    rhos = [0.5 * np.array([[1, 1], [1, 1]], complex), 0.5 * np.array([[1, -1j], [1j, 1]], complex)]
    rows = []
    for s in seqs:
        # y_k for step k: -1 while the coherence is rho_01, +1 after an odd number of pulses
        y = -np.ones(N)
        for k in range(1, N):
            y[k] = -y[k - 1] if s[k - 1] else y[k - 1]
        t = dt * np.arange(N + 1)
        # Re part: (1/2) int J coth |int y e^{iwt} dt|^2,  |.|^2 = |sum_k y_k (e^{iw t_k+1} - e^{iw t_k})|^2 / w^2
        Yw = np.zeros(len(W), complex)
        Sw = np.zeros(len(W))
        for k in range(N):
            Yw += y[k] * (np.exp(1j * W * t[k + 1]) - np.exp(1j * W * t[k]))
            Sw += y[k] * (np.sin(W * t[k + 1]) - np.sin(W * t[k]))
        re = 0.5 * np.sum(Wt * J * c * np.abs(Yw)**2 / W**2)
        im = np.sum(Wt * J * Sw / W**2)
        E = -re - 1j * im      # exponent of the carried coherence (sign: T1 calibration)
        for rho in rhos:
            # carried value: initial rho_01 is carried; after a pulse it sits in rho_10
            val = rho[0, 1] * np.exp(E)
            out = np.zeros((2, 2), complex)
            out[0, 0], out[1, 1] = (rho[0, 0], rho[1, 1]) if np.sum(s) % 2 == 0 else (rho[1, 1], rho[0, 0])
            if y[-1] < 0:
                out[0, 1] = val; out[1, 0] = np.conj(val)
            else:
                out[1, 0] = val; out[0, 1] = np.conj(val)
            rows.append([np.trace(sx @ out), np.trace(sy @ out)])
    write_matrix(os.path.join(d, 'ff_sequences.txt'), np.array(seqs, complex))
    write_matrix(os.path.join(d, 'ff_expected.txt'), np.array(rows))
    print('filter function: N', N, 'sequences', len(seqs))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    export_filter_function()
    export_toy()
