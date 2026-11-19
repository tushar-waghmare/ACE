"""
Study A, Stage 0 proxy (no PT-MPO), reconstructed from the description in docs/studyA_spec.md.

Pure-dephasing Gaussian bath, QUAPI coefficients eta_{kl} (hbar = 1, t in ps, w in ps^-1).
Cross-cut blocks at a cut (Theorem 2 of the Letter):
  flat / HS ensemble : B_HS = sqrt(2) [eta_fp, -conj(eta_fp)]
  pi-pulse family    : B_F  = 4 Sigma_f^{1/2} Re(eta_fp) Sigma_p^{1/2},  Sigma_kl = E[y_k y_l]
     Bernoulli(p)    : Sigma_kl = (1-2p)^{|k-l|}
     coarse(k, p)    : Sigma_kl = (1-2p)^{|blk(k)-blk(l)|}
Linear-order proxy for (rank - 1): r_abs(eps) = min{ r : sum_{i>r} s_i(B)^2 <= eps^2 }.
"""
import argparse
import numpy as np

DT, BETA, WC = 0.1, 1.0, 5.0
PEAK_W, PEAK_H = 0.15, 6.0      # width, height in units of alpha
OOB = [12., 15., 18., 21., 24., 27.]


def J_of(w, alpha, peaks):
    J = alpha * w * np.exp(-w / WC)
    for w0 in peaks:
        J += PEAK_H * alpha * PEAK_W**2 * (w / w0) / ((w - w0)**2 + PEAK_W**2)
    return J


BATHS = {
    "0 out-of-band peaks": [],
    "2 out-of-band peaks": OOB[:2],
    "4 out-of-band peaks": OOB[:4],
    "6 out-of-band peaks": OOB[:6],
    "edge peak at 5": [5.],
    "edge peak at 3": [3.],
    "in-band peak at 0.8": [0.8],
}


def eta_lags(alpha, peaks, nmax, dt=DT, beta=BETA, wmax=400., dw=2e-3):
    """eta(Delta) for Delta = 1..nmax-1:
    (1/pi) int dw J(w) 4 sin^2(w dt/2)/w^2 [coth(beta w/2) cos(w Delta dt) - i sin(w Delta dt)]"""
    w = np.arange(dw / 2, wmax, dw)
    base = J_of(w, alpha, peaks) * 4 * np.sin(w * dt / 2)**2 / w**2 / np.pi * dw
    coth = 1 / np.tanh(beta * w / 2)
    lag = np.arange(1, nmax)
    ph = np.outer(lag, w) * dt
    return (np.cos(ph) * (base * coth)).sum(1) - 1j * (np.sin(ph) * base).sum(1)


def eta_fp(eta, N, cut):
    """rows: future steps k = cut..N-1, cols: past steps l = 0..cut-1; eta_{kl} = eta(k-l)."""
    k = np.arange(cut, N)[:, None]
    l = np.arange(cut)[None, :]
    return eta[k - l - 1]


def psd_sqrt(S):
    L, V = np.linalg.eigh(S)
    return (V * np.sqrt(np.clip(L, 0, None))) @ V.T


def sigma(family, idx):
    kind, par = family
    if kind == "bern":
        q = 1 - 2 * par
        return q ** np.abs(idx[:, None] - idx[None, :])
    kk, p = par
    b = idx // kk
    return (1 - 2 * p) ** np.abs(b[:, None] - b[None, :])


FAMILIES = {"p=.002": ("bern", 0.002), "p=.005": ("bern", 0.005), "p=.01": ("bern", 0.01),
            "coarse10": ("coarse", (10, 0.3)), "coarse20": ("coarse", (20, 0.3))}


def blocks(alpha, peaks, N, cut=None):
    cut = N // 2 if cut is None else cut
    eta = eta_lags(alpha, peaks, N)
    E = eta_fp(eta, N, cut)
    out = {"HS": np.sqrt(2) * np.hstack([E, -E.conj()])}
    fi, pi = np.arange(cut, N), np.arange(cut)
    for name, fam in FAMILIES.items():
        Sf = psd_sqrt(sigma(fam, fi))
        Sp = psd_sqrt(sigma(fam, pi))
        out[name] = 4 * Sf @ E.real @ Sp
    return out


def r_abs(s, eps):
    tail = np.concatenate([np.cumsum((s**2)[::-1])[::-1], [0.]])  # tail[r] = sum_{i>=r} s_i^2
    return int(np.argmax(tail <= eps**2))


def table(alpha, N, epss=(1e-2, 3e-3, 1e-3)):
    print(f"alpha = {alpha}, N = {N}")
    print(f"{'bath':24s}" + "".join(f"{n:>20s}" for n in ["HS"] + list(FAMILIES)))
    for bname, peaks in BATHS.items():
        B = blocks(alpha, peaks, N)
        row = f"{bname:24s}"
        for n in ["HS"] + list(FAMILIES):
            s = np.linalg.svd(B[n], compute_uv=False)
            rs = "/".join(str(r_abs(s, e)) for e in epss)
            row += f"{rs:>12s} [{np.linalg.norm(s):5.2f}]"
        print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, nargs="*", default=[0.05, 0.2])
    ap.add_argument("--N", type=int, default=200)
    a = ap.parse_args()
    for al in a.alpha:
        table(al, a.N)
        print()
    print("Time dependence (alpha = 0.05, 2 out-of-band peaks)")
    for N in (100, 200, 400):
        B = blocks(0.05, BATHS["2 out-of-band peaks"], N)
        print(f"N={N} " + "   ".join(
            f"{n}: " + "/".join(str(r_abs(np.linalg.svd(B[n], compute_uv=False), e)) for e in (1e-2, 3e-3, 1e-3))
            for n in ["HS"] + list(FAMILIES)))
