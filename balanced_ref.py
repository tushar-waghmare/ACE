"""
Reference implementation: protected, square-root operational balanced truncation
of a PT-MPO, written in ACE's index conventions.

Ground truth for the C++ implementation in ACE. Run

    python balanced_ref.py          # all self-tests + a small benchmark

ACE conventions reproduced here (see ProcessTensorElementAccessor::propagate):
  * augmented state  psi : shape (NL, D)  rows = system Liouville index, cols = bond
  * PT element       M   : shape (NL*NL, D1, D2), outer index beta = i*NL + j,
                           i = output, j = input Liouville index (no dictionary here)
  * propagation      out[i] = sum_j psi[j] @ M[i*NL + j]        (row vector on bond)
  * closure          vector c on the element's d2 bond; tr-readout = psi @ c
  * a forward-sweep PassOn {P, Pinv} multiplies P into the next element's d1 and
    Pinv into this element's d2; here P = X^T, Pinv = Y^T.

Paper conventions (column vectors) are recovered by p_mu = psi[mu, :] (no conj),
f_mu = chi[mu, :], Q^beta = M[beta].T, Pi = X @ Y.  Inserting Pi on a bond is
psi -> psi @ Pi.T in ACE convention.
"""
import numpy as np
from scipy.linalg import expm

rng_global = np.random.default_rng(1234)

# ---------------------------------------------------------------- basics
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.array([[1, 0], [0, -1]], complex)
I2 = np.eye(2, dtype=complex)


def vec(rho):                      # row-major: vec[a*d+b] = rho[a, b]
    return rho.reshape(-1)


def sop_unitary(U):                # rho -> U rho U^dagger
    return np.kron(U, U.conj())


def obs_vec(O):                    # o @ vec(rho) = tr(O rho)
    return O.T.reshape(-1)


def kron_all(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def propagate(psi, M):
    NL = psi.shape[0]
    Mr = M.reshape(NL, NL, M.shape[1], M.shape[2])
    return np.einsum('jd,ijde->ie', psi, Mr)


def propagate_adjoint(chi, M):
    """Tester on the d2 bond -> tester on the d1 bond, such that
    pair(propagate_adjoint(chi, M), psi) == pair(chi, propagate(psi, M))."""
    NL = chi.shape[0]
    Mr = M.reshape(NL, NL, M.shape[1], M.shape[2])
    return np.einsum('ie,ijde->jd', chi, Mr)


def pair(chi, psi):                # bilinear pairing, no conjugation
    return np.sum(chi * psi)


# ------------------------------------------- exact PT-MPO of a small environment
class ToyPT:
    """Exact PT-MPO for a qubit coupled to n_env environment qubits.
    Bond = environment Liouville space (D = 4**n_env). All elements identical."""

    def __init__(self, N=10, dt=0.3, n_env=2, seed=0):
        rng = np.random.default_rng(seed)
        dS, dE = 2, 2 ** n_env
        self.N, self.dt, self.dS, self.dE = N, dt, dS, dE
        self.NL, self.D = dS * dS, dE * dE
        H_E = np.zeros((dE, dE), complex)
        B = np.zeros((dE, dE), complex)
        for k in range(n_env):
            ops = [I2] * n_env
            ops[k] = SZ
            H_E += rng.uniform(0.5, 1.5) / 2 * kron_all(*ops)
            ops[k] = SX
            B += rng.uniform(0.3, 0.8) * kron_all(*ops)
        if n_env > 1:
            H_E += 0.3 * kron_all(SX, SX, *([I2] * (n_env - 2)))
        H = np.kron(SZ, B) + np.kron(I2, H_E)
        S = sop_unitary(expm(-1j * H * dt))                    # joint vec order (s,e,s',e')
        S = S.reshape([dS, dE, dS, dE] * 2).transpose(0, 2, 1, 3, 4, 6, 5, 7)
        S = S.reshape(self.NL, self.D, self.NL, self.D)        # (alpha, eps_out, alpha~, eps_in)
        M = S.transpose(0, 2, 3, 1).reshape(self.NL * self.NL, self.D, self.D)
        self.M = [M.copy() for _ in range(N)]
        beta = 0.7
        rhoE = expm(-beta * H_E)
        self.rhoE = rhoE / np.trace(rhoE)
        self.H_E, self.B = H_E, B
        self.closure = [vec(np.eye(dE, dtype=complex)) for _ in range(N)]  # env trace

    def copy(self):
        new = ToyPT.__new__(ToyPT)
        new.__dict__.update(self.__dict__)
        new.M = [m.copy() for m in self.M]
        new.closure = [c.copy() for c in self.closure]
        return new


# ------------------------------------------------------------- contexts
def random_context(rng):
    """One Hamiltonian H_S = Delta/2 sz + Omega/2 sx, initial pure state, observable."""
    Delta, Omega = rng.uniform(-1, 1), rng.uniform(0.2, 1.5)
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    return dict(H=Delta / 2 * SZ + Omega / 2 * SX, rho0=np.outer(v, v.conj()),
                O=n[0] * SX + n[1] * SY + n[2] * SZ)


def half_step(pt, H):
    return sop_unitary(expm(-1j * H * pt.dt / 2))


def run_forward(pt, ctx, k_stop=None, insert=None):
    """Augmented state after element k_stop-1 (cut k_stop), symmetric Trotter.
    insert: dict {cut: Pi} applied as psi -> psi @ Pi.T right after that cut."""
    Sh = half_step(pt, ctx['H'])
    psi = np.outer(vec(ctx['rho0']), vec(pt.rhoE))
    k_stop = pt.N if k_stop is None else k_stop
    for n in range(k_stop):
        psi = Sh @ propagate(Sh @ psi, pt.M[n])
        if insert and n in insert:
            psi = psi @ insert[n].T
    return psi


def outcome(pt, ctx, insert=None):
    psi = run_forward(pt, ctx, insert=insert)
    return obs_vec(ctx['O']) @ psi @ pt.closure[-1]


def tester(pt, ctx, k):
    """Future tester chi at cut k (bond d2 of element k), shape (NL, D)."""
    Sh = half_step(pt, ctx['H'])
    chi = np.outer(obs_vec(ctx['O']), pt.closure[-1])
    for n in range(pt.N - 1, k, -1):
        chi = Sh.T @ propagate_adjoint(Sh.T @ chi, pt.M[n])
    return chi


def direct_outcome(pt, ctx):
    """Brute-force joint S+E simulation for validating ToyPT."""
    U_h = np.kron(expm(-1j * ctx['H'] * pt.dt / 2), np.eye(pt.dE))
    U_int = expm(-1j * (np.kron(SZ, pt.B) + np.kron(I2, pt.H_E)) * pt.dt)
    rho = np.kron(ctx['rho0'], pt.rhoE)
    for _ in range(pt.N):
        for U in (U_h, U_int, U_h):
            rho = U @ rho @ U.conj().T
    rhoS = np.einsum('aebe->ab', rho.reshape(2, pt.dE, 2, pt.dE))
    return np.trace(ctx['O'] @ rhoS)


# ------------------------------------------------------- factors and Gramians
def past_factor(pt, ctxs, k, weights=None):
    """Absorbed past factor A (D x n*NL): columns sqrt(w) p_{a,mu}."""
    w = np.ones(len(ctxs)) / len(ctxs) if weights is None else weights
    return np.hstack([np.sqrt(wa) * run_forward(pt, c, k + 1).T for wa, c in zip(w, ctxs)])


def future_factor(pt, ctxs, k, weights=None):
    """Absorbed future factor F (n*NL x D): rows sqrt(v) f_{b,mu}^T."""
    v = np.ones(len(ctxs)) / len(ctxs) if weights is None else weights
    return np.vstack([np.sqrt(vb) * tester(pt, c, k) for vb, c in zip(v, ctxs)])


def hs_gramians(pt, k):
    """Flat (Hilbert-Schmidt) Gramians of the bare PT-MPO at cut k (paper convention)."""
    rE = vec(pt.rhoE)
    Wp = np.outer(rE, rE.conj())
    for n in range(k + 1):
        Wp = sum(Mb.T @ Wp @ Mb.conj() for Mb in pt.M[n])
    q = pt.closure[-1]
    Wf = np.outer(q.conj(), q)
    for n in range(pt.N - 1, k, -1):
        Wf = sum(Mb.conj() @ Wf @ Mb.T for Mb in pt.M[n])
    return Wp, Wf


def factor_from_gramians(Wp, Wf, tol=1e-14):
    lp, Vp = np.linalg.eigh((Wp + Wp.conj().T) / 2)
    lf, Vf = np.linalg.eigh((Wf + Wf.conj().T) / 2)
    keep_p, keep_f = lp > tol * lp.max(), lf > tol * lf.max()
    A = Vp[:, keep_p] * np.sqrt(lp[keep_p])
    F = (Vf[:, keep_f] * np.sqrt(lf[keep_f])).conj().T
    return A, F


def blend_gauge_invariant(Wp_op, Wf_op, Wp_hs, Wf_hs, eta):
    """Normalizers c_p = tr(W_p H_f), c_f = tr(H_p W_f) with H = HS Gramians."""
    c_po, c_ph = np.trace(Wp_op @ Wf_hs).real, np.trace(Wp_hs @ Wf_hs).real
    c_fo, c_fh = np.trace(Wp_hs @ Wf_op).real, c_ph
    Wp = (1 - eta) * Wp_op / c_po + eta * Wp_hs / c_ph
    Wf = (1 - eta) * Wf_op / c_fo + eta * Wf_hs / c_fh
    return Wp, Wf


# ----------------------------------------------------------- the projector
def balanced_projector(A, F, m, q=None, rhat=None, rtol=1e-13):
    """Square-root balanced truncation to total bond dimension m.
    With (q, rhat): protected version, q^T Pi = q^T and Pi rhat = rhat,
    using m-1 complementary directions. Returns X, Y, sigma (of the core)."""
    D = A.shape[0]
    if q is None:
        K, Ap, Fp, lead_X, lead_Y, r = F @ A, A, F, None, None, m
    else:
        s = q @ rhat
        if abs(s) < 1e-14:
            raise ValueError("q^T rhat = 0: protected pair is degenerate")
        Pperp = np.eye(D) - np.outer(rhat, q) / s
        Ap, Fp = Pperp @ A, F @ Pperp
        K, lead_X, lead_Y, r = Fp @ Ap, (rhat / s)[:, None], q[None, :], m - 1
    U, S, Vh = np.linalg.svd(K, full_matrices=False)
    r = min(r, int(np.sum(S > rtol * max(S[0], 1e-300))))
    Xr = Ap @ Vh[:r].conj().T / np.sqrt(S[:r])
    Yr = (U[:, :r].conj().T @ Fp) / np.sqrt(S[:r])[:, None]
    if q is None:
        return Xr, Yr, S
    return np.hstack([lead_X, Xr]), np.vstack([lead_Y, Yr]), S


def trace_weighted_rhat(A_rows_by_ctx, NL):
    """Default rhat: mean over contexts of sum_nu psi[(nu,nu), :] ('trace' the system).
    Gives q^T rhat = mean tr rho_S = 1 for an exact closure."""
    d = int(round(np.sqrt(NL)))
    diag = [nu * d + nu for nu in range(d)]
    return np.mean([psi[diag].sum(axis=0) for psi in A_rows_by_ctx], axis=0)


def err_abs(A, F, Pi):
    return np.linalg.norm(F @ (np.eye(Pi.shape[0]) - Pi) @ A) ** 2


# ---------------------------------------------------------------- self-tests
def run_tests(verbose=True):
    rng = np.random.default_rng(7)
    pt = ToyPT(N=10, dt=0.3, n_env=2, seed=3)
    ctx_p = [random_context(rng) for _ in range(12)]
    ctx_f = [random_context(rng) for _ in range(12)]
    held = [random_context(rng) for _ in range(30)]
    k, m = 4, 5
    ok = {}

    # T1: exact PT-MPO contraction == brute-force joint simulation
    ok['T1 PT == joint simulation'] = max(abs(outcome(pt, c) - direct_outcome(pt, c)) for c in held[:5]) < 1e-10

    # T2: adjoint identity at every cut
    c = held[0]
    y = outcome(pt, c)
    ok['T2 y == chi_k . psi_k all cuts'] = max(abs(pair(tester(pt, c, kk), run_forward(pt, c, kk + 1)) - y)
                                             for kk in range(pt.N - 1)) < 1e-10

    # factors at cut k, protected pair
    A, F = past_factor(pt, ctx_p, k), future_factor(pt, ctx_f, k)
    q = pt.closure[k]
    rhat = trace_weighted_rhat([run_forward(pt, cc, k + 1) for cc in ctx_p], pt.NL)
    X, Y, S = balanced_projector(A, F, m, q, rhat)
    Pi = X @ Y

    # T3: proposition (projector, protection, exact tail)
    ok['T3a YX = I'] = np.allclose(Y @ X, np.eye(X.shape[1]), atol=1e-9)
    ok['T3b q^T Pi = q^T, Pi rhat = rhat'] = np.allclose(q @ Pi, q, atol=1e-9) and np.allclose(Pi @ rhat, rhat, atol=1e-9)
    ok['T3c q^T rhat == 1 (trace-weighted rhat)'] = abs(q @ rhat - 1) < 1e-10
    ok['T3d E_abs == core tail'] = abs(err_abs(A, F, Pi) - np.sum(S[m - 1:] ** 2)) < 1e-9 * max(1, np.sum(S ** 2))

    # T4: rho_S at the cut is exact under the protected truncation, held-out pasts
    worst = 0
    for cc in held:
        psi = run_forward(pt, cc, k + 1)
        worst = max(worst, np.max(abs(psi @ Pi.T @ q - psi @ q)))
    ok['T4 rho_S(t_k) exact at cut (held-out)'] = worst < 1e-10

    # T5: bracket  tail_{d^2 m}(F_aug A_aug) <= E_aug(Pi) <= d^2 E_abs(Pi)
    Ps = [run_forward(pt, cc, k + 1) for cc in ctx_p]
    Cs = [tester(pt, cc, k) for cc in ctx_f]
    R = np.eye(pt.D) - Pi
    E_aug = np.mean([abs(pair(ch, ps @ R.T)) ** 2 for ps in Ps for ch in Cs])
    Aaug = np.stack([ps.reshape(-1) for ps in Ps], 1) / np.sqrt(len(Ps))
    Faug = np.stack([ch.reshape(-1) for ch in Cs], 0) / np.sqrt(len(Cs))
    s_aug = np.linalg.svd(Faug @ Aaug, compute_uv=False)
    lower = np.sum(s_aug[pt.NL * m:] ** 2)
    ok['T5 bracket holds'] = lower - 1e-12 <= E_aug <= pt.NL * err_abs(A, F, Pi) + 1e-12

    # T6: gauge invariance of balanced values and of the gauge-invariant blend
    G = rng.normal(size=(pt.D, pt.D)) + 1j * rng.normal(size=(pt.D, pt.D))
    Gi = np.linalg.inv(G)
    Wp_op, Wf_op = A @ A.conj().T, F.conj().T @ F
    Wp_hs, Wf_hs = hs_gramians(pt, k)
    def spec(Wp, Wf):
        ev = np.sort(np.linalg.eigvals(Wp @ Wf).real)[::-1]
        return ev / ev[0]
    gp = lambda W: Gi @ W @ Gi.conj().T
    gf = lambda W: G.conj().T @ W @ G
    b0 = blend_gauge_invariant(Wp_op, Wf_op, Wp_hs, Wf_hs, 0.2)
    b1 = blend_gauge_invariant(gp(Wp_op), gf(Wf_op), gp(Wp_hs), gf(Wf_hs), 0.2)
    ok['T6 gauge invariance (op + blend)'] = (np.allclose(spec(Wp_op, Wf_op)[:6], spec(gp(Wp_op), gf(Wf_op))[:6], atol=1e-8)
                                              and np.allclose(spec(*b0)[:6], spec(*b1)[:6], atol=1e-8))

    # T7: absorbing into ACE tensors (M_k <- M_k Y^T, M_{k+1} <- X^T M_{k+1}) == inserting Pi
    pt2 = pt.copy()
    pt2.M[k] = np.einsum('bde,fe->bdf', pt.M[k], Y)          # M_k[beta] @ Y.T
    pt2.M[k + 1] = np.einsum('ed,bdf->bef', X.T, pt.M[k + 1])  # X.T @ M_{k+1}[beta]
    pt2.closure[k] = X.T @ pt.closure[k]
    ok['T7 ACE absorption == insertion'] = max(abs(outcome(pt2, cc) - outcome(pt, cc, insert={k: Pi})) for cc in held[:5]) < 1e-10

    # T8: protected sweep over all cuts keeps tr rho_S(t) = 1 for every held-out trajectory
    pt_bal = balanced_sweep(pt, ctx_p, ctx_f, m, protect=True)
    worst_tr = 0
    for cc in held[:10]:
        Sh = half_step(pt_bal, cc['H'])
        psi = np.outer(vec(cc['rho0']), vec(pt_bal.rhoE))
        for n in range(pt_bal.N):
            psi = Sh @ propagate(Sh @ psi, pt_bal.M[n])
            rhoS = (psi @ pt_bal.closure[n]).reshape(2, 2)
            worst_tr = max(worst_tr, abs(np.trace(rhoS) - 1))
    ok['T8 protected sweep: tr rho_S(t) = 1 at all t'] = worst_tr < 1e-9

    if verbose:
        for key, val in ok.items():
            print(f"{'PASS' if val else 'FAIL'}  {key}")
    return all(ok.values())


# ------------------------------------------------------------------ sweeps
def balanced_sweep(pt, ctx_p, ctx_f, m, protect=True, mode='op', eta=0.0):
    """Forward sweep over all cuts. Future factors from the not-yet-truncated part,
    past factors from the already-truncated part (recomputed; fine for a toy).
    mode: 'op' operational, 'hs' Hilbert-Schmidt (= canonical SVD truncation),
          'blend' gauge-invariant blend with weight eta."""
    out = pt.copy()
    for k in range(out.N - 1):
        if out.M[k].shape[2] <= m:
            continue
        if mode == 'op':
            A, F = past_factor(out, ctx_p, k), future_factor(out, ctx_f, k)
        else:
            Wp_hs, Wf_hs = hs_gramians(out, k)
            if mode == 'hs':
                A, F = factor_from_gramians(Wp_hs, Wf_hs)
            else:
                Ao, Fo = past_factor(out, ctx_p, k), future_factor(out, ctx_f, k)
                A, F = factor_from_gramians(*blend_gauge_invariant(
                    Ao @ Ao.conj().T, Fo.conj().T @ Fo, Wp_hs, Wf_hs, eta))
        if protect:
            q = out.closure[k]
            rhat = trace_weighted_rhat([run_forward(out, cc, k + 1) for cc in ctx_p], out.NL)
            X, Y, _ = balanced_projector(A, F, m, q, rhat)
        else:
            X, Y, _ = balanced_projector(A, F, m)
        out.M[k] = np.einsum('bde,fe->bdf', out.M[k], Y)
        out.M[k + 1] = np.einsum('ed,bdf->bef', X.T, out.M[k + 1])
        out.closure[k] = X.T @ out.closure[k]
    return out


def benchmark():
    rng = np.random.default_rng(11)
    pt = ToyPT(N=10, dt=0.3, n_env=2, seed=3)
    ctx_p = [random_context(rng) for _ in range(12)]
    ctx_f = [random_context(rng) for _ in range(12)]
    held = [random_context(rng) for _ in range(40)]
    exact = np.array([outcome(pt, c) for c in held])
    print(f"\nBenchmark: qubit + 2 env qubits (D={pt.D}), N={pt.N}; RMS error of <O>(t_N) on 40 held-out contexts")
    print(f"{'m':>3} {'SVD (HS)':>12} {'balanced op':>12} {'bal. eta=0.1':>13}")
    for m in (3, 4, 6, 8):
        row = []
        for mode, prot, eta in (('hs', False, 0), ('op', True, 0), ('blend', True, 0.1)):
            ptc = balanced_sweep(pt, ctx_p, ctx_f, m, protect=prot, mode=mode, eta=eta)
            approx = np.array([outcome(ptc, c) for c in held])
            row.append(np.sqrt(np.mean(abs(approx - exact) ** 2)))
        print(f"{m:>3} {row[0]:>12.3e} {row[1]:>12.3e} {row[2]:>13.3e}")


if __name__ == '__main__':
    passed = run_tests()
    print("\nALL TESTS PASSED" if passed else "\nSOME TESTS FAILED")
    benchmark()
