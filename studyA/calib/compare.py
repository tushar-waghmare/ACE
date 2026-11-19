"""Compare an ACE coherence output (columns t, Re rho_01, Im rho_01) with the exact Gaussian result.
Usage: python compare.py <bath> <outfile>   (bath: J file studyA/baths/J_<bath>_a0.05.dat)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, baths as B
b, f = sys.argv[1], sys.argv[2]
d = np.loadtxt(f)
t, rho = d[:, 0], d[:, 1] + 1j * d[:, 2]
G, T, L = B.phase_integrals(B.InterpJ.from_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                           '..', 'baths', f'J_{b}_a0.05.dat')), t)
for sgn in (+1, -1):
    ref = 0.5 * np.exp(-G + sgn * 1j * T)
    rel = np.abs(rho - ref) / np.abs(ref)
    print(f"{b} {f} sign {sgn:+d}: max rel err {rel.max():.3e} (t={t[rel.argmax()]})")
