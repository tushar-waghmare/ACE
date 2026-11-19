"""
Study A bath definitions (docs/studyA_spec.md, Sec. 2) and exact Gaussian-dephasing references.

Conventions
-----------
Stage 0 (tools/studyA/stage0_proxy.py): J_s(w) = alpha w exp(-w/wc) + peaks, QUAPI
  eta_D = (1/pi) int J_s 4 sin^2(w dt/2)/w^2 [coth(beta w/2) cos(w D dt) - i sin(w D dt)] dw,
  sigma_z-type path variables y = +-1, family exponent 4 Re(eta) y_k y_l.
ACE (src/DiagBB.cpp, SysOp |1><1|): K_D = int J_A 2(1-cos w dt)/w^2 [coth cos - i sin] dw (no 1/pi);
  the influence phase of the pair (k,l) is S = -(c_nu_k - c_mu_k)(c_nu_l K - c_mu_l K^*) with
  couplings c in {0,1}; for a pi-pulse path (y = c_nu - c_mu = +-1) the y_k y_l coefficient is -Re K.
  Equal family exponents: Re K = 4 Re eta  <=>  J_A = (4/pi) J_s.  (Verified by the T1 calibration.)
Frequencies in ps^-1, t in ps, beta in ps (hbar = 1 in these units).
"""
import numpy as np

HBAR = 0.6582119569      # meV ps
KB = 8.6173303e-2        # meV / K, as ACE (include/Constants.hpp)
DT, BETA, WC = 0.1, 1.0, 5.0
G, H = 0.15, 6.0         # peak width (ps^-1) and height (units of alpha)
OOB = [12., 15., 18., 21., 24., 27.]
J_MAP = 4.0 / np.pi      # J_ACE = J_MAP * J_stage0
W_MAX = 200.0            # table end = Boson_omega_max
DW_FILE = G / 100.       # table spacing (spec: <= g/20)

BATHS = {
    "ohmic": [], "oob2": OOB[:2], "oob4": OOB[:4], "oob6": OOB[:6],
    "edge3": [3.], "edge5": [5.], "inband0.8": [0.8],
}


def temperature_K(beta=BETA):
    return HBAR / (KB * beta)


def J_stage0(w, alpha, peaks):
    w = np.asarray(w, float)
    J = alpha * w * np.exp(-w / WC)
    for w0 in peaks:
        J = J + H * alpha * G**2 * (w / w0) / ((w - w0)**2 + G**2)
    return J


def J_ace(w, alpha, peaks):
    return J_MAP * J_stage0(w, alpha, peaks)


def write_J_file(fname, alpha, peaks, wmax=W_MAX, dw=DW_FILE):
    w = np.linspace(0., wmax, int(round(wmax / dw)) + 1)
    np.savetxt(fname, np.column_stack([w, J_ace(w, alpha, peaks)]), fmt="%.17g")
    return w


class InterpJ:
    """Piecewise-linear J from a table, exactly as RealFunction_Interpolate (clamped outside)."""
    def __init__(self, w, J):
        self.w, self.J = np.asarray(w), np.asarray(J)

    @classmethod
    def from_file(cls, fname):
        d = np.loadtxt(fname)
        return cls(d[:, 0], d[:, 1])

    def nodes(self, order=10, wmax=None):
        """Gauss-Legendre nodes/weights per table interval on [w_0, wmax]; J is linear there."""
        x, wt = np.polynomial.legendre.leggauss(order)
        a, b = self.w[:-1], self.w[1:]
        if wmax is not None:
            keep = a < wmax
            a, b = a[keep], np.minimum(b[keep], wmax)
        h = (b - a) / 2
        W = (a[:, None] + h[:, None] * (x[None, :] + 1)).ravel()
        Wt = (h[:, None] * wt[None, :]).ravel()
        return W, Wt, np.interp(W, self.w, self.J)


def coth(x):
    return 1. / np.tanh(x)


def phase_integrals(Jint, times, beta=BETA, wmax=W_MAX, chunk=200000):
    """Gamma(t) = int J coth(beta w/2) (1-cos wt)/w^2,  Theta(t) = int J sin(wt)/w^2,
    Lambda = int J/w (polaron shift: Theta_noSub = Theta - Lambda t)."""
    W, Wt, J = Jint.nodes(wmax=wmax)
    c = coth(beta * W / 2)
    times = np.atleast_1d(times)
    Gam = np.zeros(len(times)); The = np.zeros(len(times))
    for s in range(0, len(W), chunk):
        w, f = W[s:s + chunk], (Wt * J)[s:s + chunk] / W[s:s + chunk]**2
        wt = np.outer(times, w)
        Gam += (2 * np.sin(wt / 2)**2 * (f * c[s:s + chunk])).sum(1)
        The += (np.sin(wt) * f).sum(1)
    Lam = (Wt * J / W).sum()
    return Gam, The, Lam


def K_coefficients(Jint, n_max, dt=DT, beta=BETA, wmax=W_MAX, subtract_polaron_shift=True,
                   chunk=200000):
    """ACE's K_n (src/DiagBB.cpp) by Gauss-Legendre quadrature on the tabulated J:
    K_0 = int J/w^2 [coth (1-cos w dt) + i (sin w dt [- w dt if not subtracted])],
    K_n = int J 2(1-cos w dt)/w^2 [coth cos(w n dt) - i sin(w n dt)],  n >= 1."""
    W, Wt, J = Jint.nodes(wmax=wmax)
    c = coth(beta * W / 2)
    K = np.zeros(n_max, complex)
    n = np.arange(1, n_max)
    for s in range(0, len(W), chunk):
        w = W[s:s + chunk]
        f = (Wt * J)[s:s + chunk] / w**2
        cs = c[s:s + chunk]
        im0 = np.sin(w * dt) - (0. if subtract_polaron_shift else w * dt)
        K[0] += (f * cs * 2 * np.sin(w * dt / 2)**2).sum() + 1j * (f * im0).sum()
        g = 2 * f * 2 * np.sin(w * dt / 2)**2
        ph = np.outer(n, w) * dt
        K[1:] += (np.cos(ph) * (g * cs)).sum(1) - 1j * (np.sin(ph) * g).sum(1)
    return K


def write_K_file(fname, K, dt=DT):
    t = dt * np.arange(len(K))
    np.savetxt(fname, np.column_stack([t, K.real, K.imag]), fmt="%.17g")
