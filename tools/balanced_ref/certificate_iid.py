"""
Global error certificate for a forward sweep: step-independent (i.i.d.) context ensemble.

Checks, on the exact toy PT-MPO of balanced_ref.py (qubit + 2 environment qubits, D = 16):

  (i)   telescoping identity  y - y~ = sum_k e_k  for every sampled trajectory
  (ii)  certificate           sqrt(E|y - y~|^2) <= sum_k sqrt(E|e_k|^2)
  (iii) for step-independent ensembles  E|e_k|^2 = E_aug,k  (local system-coupled error,
        computable during the sweep) and  E_aug,k <= d^2 * E_abs,k  (d^2 * reported tail)

E|y - y~|^2 is computed EXACTLY from joint second moments of (psi, psi~); (iii) is checked
by Monte Carlo. Works for any inserted matrices: protected balanced (oblique) and SVD.

Definitions (ACE conventions, see balanced_ref.py):
  hybrid past   psi~_k : propagated through cuts < k WITH insertions, cut k not yet truncated
  exact future  chi_k  : propagated backward through the UNTRUNCATED chain
  e_k = pair(chi_k, psi~_k @ (I - Pi_k).T)

Run:  python certificate_iid.py      (about 1-2 minutes; exits non-zero on failure)
"""
import os
import sys
import numpy as np
from scipy.stats import unitary_group

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.dirname(_here), os.path.dirname(os.path.dirname(_here))):
    sys.path.insert(0, _p)          # balanced_ref.py next to this file or at the repo root
from balanced_ref import (ToyPT, sop_unitary, vec, obs_vec, SX, SY, SZ, propagate,
                          propagate_adjoint, pair, factor_from_gramians, balanced_projector,
                          hs_gramians)

rng = np.random.default_rng(4)
pt = ToyPT(N=10, dt=0.3, n_env=2, seed=3)
NL, D, N = pt.NL, pt.D, pt.N
d = int(round(np.sqrt(NL)))

# ---- step-independent ensemble: every half-step draws one of K unitaries; rho0, O independent
K = 5
STEPS = [sop_unitary(unitary_group.rvs(2, random_state=rng)) for _ in range(K)]
RHOS = []
for _ in range(4):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v /= np.linalg.norm(v)
    RHOS.append(vec(np.outer(v, v.conj())))
OBS = []
for _ in range(3):
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    OBS.append(obs_vec(n[0] * SX + n[1] * SY + n[2] * SZ))
E_RHO = np.mean([np.outer(r, r.conj()) for r in RHOS], 0)
M_RHO = np.mean(RHOS, 0)
E_OBS = np.mean([np.outer(o.conj(), o) for o in OBS], 0)
S_MEAN = np.mean(STEPS, 0)
rE = vec(pt.rhoE)
q_end = pt.closure[-1]


def Tmat(M):
    """vec(propagate(psi, M)) == Tmat(M) @ vec(psi), row-major vec of psi (NL x D)."""
    D1, D2 = M.shape[1], M.shape[2]
    return M.reshape(NL, NL, D1, D2).transpose(0, 3, 1, 2).reshape(NL * D2, NL * D1)


def lift(S, dim):
    return np.kron(S, np.eye(dim))


def avg_fwd(W, dim):
    return np.mean([lift(S, dim) @ W @ lift(S, dim).conj().T for S in STEPS], 0)


def avg_bwd(W, dim):
    return np.mean([lift(S, dim).conj().T @ W @ lift(S, dim) for S in STEPS], 0)


def ptrace(Waug):
    """Absorb the system index: sum of diagonal mu-blocks -> bond Gramian."""
    dim = Waug.shape[0] // NL
    return np.einsum('mamb->ab', Waug.reshape(NL, dim, NL, dim))


def exact_future_grams():
    """E[chi_k^* chi_k^T] at every cut k on the untruncated chain (augmented, NL*D square)."""
    Wf = [None] * N
    W = np.kron(E_OBS, np.outer(q_end.conj(), q_end))
    Wf[N - 1] = W
    for n in range(N - 1, 0, -1):
        T = Tmat(pt.M[n])
        W = avg_bwd(T.conj().T @ avg_bwd(W, D) @ T, D)
        Wf[n - 1] = W
    return Wf


def sweep(m, mode):
    """Forward sweep with exact second moments. Returns [(k, Pi_k, E_abs_k, E_aug_k)].
    Pi_k acts on the original bond k (the bond is untouched until cut k)."""
    Wf_aug = exact_future_grams()
    W = np.kron(E_RHO, np.outer(rE, rE.conj()))
    mu = np.kron(M_RHO, rE)
    out = []
    for k in range(N - 1):
        T = Tmat(pt.M[k])
        W = avg_fwd(T @ avg_fwd(W, D) @ T.conj().T, D)          # half step, element, half step
        mu = lift(S_MEAN, D) @ (T @ (lift(S_MEAN, D) @ mu))
        Wp, Wf = ptrace(W), ptrace(Wf_aug[k])
        if mode == 'svd':
            # flat (Hilbert-Schmidt) Gramians of the partially truncated chain
            chain = pt.copy()
            for (j, Pj, _, _) in out:
                chain.M[j] = np.einsum('bde,fe->bdf', chain.M[j], Pj)
            A, F = factor_from_gramians(*hs_gramians(chain, k))
            X, Y, _ = balanced_projector(A, F, m)
        else:
            A, F = factor_from_gramians(Wp, Wf)
            rhat = mu.reshape(NL, D)[[i * d + i for i in range(d)]].sum(0)   # trace-weighted
            X, Y, _ = balanced_projector(A, F, m, pt.closure[k], rhat)
        Pi = X @ Y
        R = np.eye(D) - Pi
        E_abs = max(np.trace(R.conj().T @ Wf @ R @ Wp).real, 0.0)
        RR = np.kron(np.eye(NL), R)
        E_aug = max(np.trace(RR.conj().T @ Wf_aug[k] @ RR @ W).real, 0.0)
        out.append((k, Pi, E_abs, E_aug))
        PP = np.kron(np.eye(NL), Pi)
        W, mu = PP @ W @ PP.conj().T, PP @ mu
    return out


def exact_global_ms_error(projs):
    """E|y - y~|^2 from joint second moments of z = [vec psi; vec psi~]."""
    P = {k: Pi for (k, Pi, _, _) in projs}
    a0 = np.kron(E_RHO, np.outer(rE, rE.conj()))
    Z = np.block([[a0, a0], [a0, a0]])
    zero = np.zeros((NL * D, NL * D))
    for k in range(N):
        T = Tmat(pt.M[k])
        B = np.block([[T, zero], [zero, T]])
        Z = np.mean([np.block([[lift(S, D), zero], [zero, lift(S, D)]]) @ Z @
                     np.block([[lift(S, D), zero], [zero, lift(S, D)]]).conj().T for S in STEPS], 0)
        Z = B @ Z @ B.conj().T
        Z = np.mean([np.block([[lift(S, D), zero], [zero, lift(S, D)]]) @ Z @
                     np.block([[lift(S, D), zero], [zero, lift(S, D)]]).conj().T for S in STEPS], 0)
        if k in P:
            G = np.block([[np.eye(NL * D), zero], [zero, np.kron(np.eye(NL), P[k])]])
            Z = G @ Z @ G.conj().T
    Wo = np.kron(E_OBS, np.outer(q_end.conj(), q_end))
    C = np.block([[Wo, -Wo], [-Wo, Wo]])
    return np.trace(Z @ C).real


def monte_carlo(projs, n_samples, seed=0):
    """Sample trajectories; return identity residual, E|y-y~|^2, and E|e_k|^2 per cut."""
    r = np.random.default_rng(seed)
    P = {k: Pi for (k, Pi, _, _) in projs}
    cuts = sorted(P)
    resid, err2, e2 = 0.0, 0.0, np.zeros(len(cuts))
    for _ in range(n_samples):
        Sx = [STEPS[i] for i in r.integers(K, size=2 * N)]
        rho, o = RHOS[r.integers(len(RHOS))], OBS[r.integers(len(OBS))]
        chis = [None] * N                                         # exact futures
        chi = np.outer(o, q_end)
        chis[N - 1] = chi
        for nn in range(N - 1, 0, -1):
            chi = Sx[2 * nn].T @ propagate_adjoint(Sx[2 * nn + 1].T @ chi, pt.M[nn])
            chis[nn - 1] = chi
        psi = np.outer(rho, rE)                                   # exact and hybrid pasts
        psit = psi.copy()
        es = []
        for k in range(N):
            psi = Sx[2 * k + 1] @ propagate(Sx[2 * k] @ psi, pt.M[k])
            psit = Sx[2 * k + 1] @ propagate(Sx[2 * k] @ psit, pt.M[k])
            if k in P:
                es.append(pair(chis[k], psit @ (np.eye(D) - P[k]).T))
                psit = psit @ P[k].T
        y, yt = o @ psi @ q_end, o @ psit @ q_end
        resid = max(resid, abs((y - yt) - sum(es)))
        err2 += abs(y - yt) ** 2 / n_samples
        e2 += np.abs(np.array(es)) ** 2 / n_samples
    return resid, err2, e2


if __name__ == '__main__':
    ok = True
    print(f"step-independent ensemble (K={K} unitaries per half-step), D={D}, N={N}, d={d}")
    print(f"{'m':>2} {'method':>9} | {'exact RMS error':>15} {'sum sqrt(E_aug)':>16} "
          f"{'d*sum sqrt(tail)':>17} | {'MC: max identity resid':>22} {'MC E|e|^2 vs E_aug':>19}")
    for m in (3, 4, 6, 8):
        for mode in ('balanced', 'svd'):
            projs = sweep(m, mode)
            rms = np.sqrt(exact_global_ms_error(projs))
            cert = sum(np.sqrt(ea) for (_, _, _, ea) in projs)
            cert_tail = d * sum(np.sqrt(eb) for (_, _, eb, _) in projs)
            resid, mc_err2, mc_e2 = monte_carlo(projs, 1500, seed=m)
            e_aug = np.array([ea for (_, _, _, ea) in projs])
            rel = np.sum(np.abs(mc_e2 - e_aug)) / np.sum(e_aug)
            print(f"{m:>2} {mode:>9} | {rms:>15.3e} {cert:>16.3e} {cert_tail:>17.3e} | "
                  f"{resid:>22.1e} {rel:>18.1%}")
            ok &= rms <= cert * (1 + 1e-9)                         # (ii) certificate
            ok &= cert <= cert_tail * (1 + 1e-9)                   # (iii) absorption bound
            ok &= resid < 1e-10                                    # (i) identity
            ok &= rel < 0.15                                       # (iii) E|e_k|^2 == E_aug,k (MC)
            ok &= all(ea <= NL * eb * (1 + 1e-9) + 1e-15 for (_, _, eb, ea) in projs)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
