# Study A: deviations from docs/studyA_spec.md

Recorded before the grid is run. Each entry: date, what, why.

- 2026-11-19, Stage 0 proxy. The original `stage0_proxy.py` was not in the repository; it was
  reconstructed from the spec's definitions (QUAPI eta via
  `(1/pi) int J(w) 4 sin^2(w dt/2)/w^2 [coth(beta w/2) cos(w D dt) - i sin(w D dt)] dw`, middle cut
  N//2). It reproduces every entry of the original Stage 0 tables (alpha = 0.05 and 0.2, all 7 baths,
  HS + 5 families, ranks and ||B||) and 17 of 18 time-dependence entries. The exception is N = 100,
  coarse20, eps = 1e-3: 3 instead of 2; that tail is borderline (sqrt(tail_2) = 1.155e-3 vs eps = 1e-3).
  Predictions in Sec. 1 are unchanged.

- 2026-11-19, T1 (bath input). ACE's own J quadrature (`Boson_J_from_file`, FFT in
  `DiagBB::precalc_FFT`) drops the finite w -> 0 value of the integrand of every K_n (rectangle rule
  starting at domega): about 1.5e-7 per K_n, 3.2e-3 relative error of rho_01(20 ps) for the Ohmic
  bath (predicted 3.0e-3 from that endpoint alone). It cannot reach the 1e-6 calibration target.
  Instead the J files are still written as specified, and K_n are computed from them by
  Gauss-Legendre quadrature per table interval (`studyA/baths.py: K_coefficients`, exact for the
  piecewise-linear J that ACE would interpolate) and passed with ACE's existing `Boson_read_K_int`.
  With these K_n, ACE's influence-functional assembly matches the analytic coherence; the remaining
  deviation is PT construction (SVD) error: 2.6e-5 / 2.7e-6 / 1.4e-7 at t = 5 ps for thresholds
  1e-8 / 1e-10 / 1e-12, i.e. <= 1e-6 needs a construction threshold <= 1e-12.
- 2026-11-19, T1 (J convention, result). J_ACE = (4/pi) J_stage0 (ACE has no 1/pi; |1><1| coupling
  gives the y_k y_l coefficient Re K vs 4 Re eta for Stage 0's +-1 path variables), sign
  rho_01 ~ exp(-Gamma + i Theta). alpha_ACE is therefore alpha = 0.05 in Stage 0 units (J files store
  J_ACE), which reproduces the Stage 0 exponent scales ||B_F|| exactly (0.02-0.06 at alpha = 0.05).
- 2026-11-19, PT construction. `use_Gaussian true` (line-by-line) is too slow for the peaked baths:
  +2 peaks at threshold 1e-7 reached line 62/200 (D = 152) after 70 min, at 1e-9 line 19/200
  (D = 335). Divide-and-conquer (`use_Gaussian_log true`, same DiagBB influence functional) needs a
  power of two: PTs are built for 256 steps (25.6 ps) and cut to the first 200 steps with
  `tools/PTB_truncate_time` (element 199 contracted with its closure = trace over the later steps,
  exact by causality). +2 peaks at 1e-7: 500 s, max D 208.
- 2026-11-19, theta. 1e-9 is not feasible on this machine for peaked baths (DnC +2 peaks at 1e-9
  stalled at line 16/256 with D ~ 425). Theta pair to be fixed in the pilot (testing 1e-6/1e-8).
- 2026-11-19, T2(b) runs on the Ohmic bath (N = 30, threshold 1e-12, D = 110), not on a peaked bath:
  a 1e-8 comparison needs an essentially untruncated PT, and the +2-peak bath reached D = 561 after
  7 lines at threshold 1e-12. The test checks pulse placement and the filter-function algebra.
- 2026-11-19, conventions fixed in the implementation (not in the spec):
  * pulse slots s = 0..N-2 at the full-step boundaries t_{s+1} (no pulse after the last step), as
    `apply_Operator_left_right {(s+1)*dt} {X} {X}` places them;
  * readouts at the final time only (Gramians, held-out errors);
  * sigma_y is the standard [[0,-i],[i,0]] (ACE's keyword `sigma_y` is its negative);
  * middle cut = cut 99 (bond between elements 99 and 100: 100 steps of past), as Stage 0 (cut N//2);
  * out-of-family set for coarse(k, p_slot): Bernoulli(4 p_slot / k) (4x the mean pulse rate);
    for Bernoulli(p): Bernoulli(4p) as specified;
  * flat (HS) spectra come from normalized recursions: their sum_s2 is relative only;
  * T4 compares flat and SVD Schmidt values for s_i >= 1e-4 s_1 (all values that enter ranks at
    eps >= 1e-3), max relative deviation 3.7e-12.
