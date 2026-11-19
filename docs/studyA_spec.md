# Study A specification: predictive vs representational memory under Gaussian dephasing

Add `@docs/studyA_spec.md` to `CLAUDE.md`. Work on branch `studyA`. This study decides whether the
paper is a PRL (Outcome A) or a PRX Quantum / PRR paper (Outcome B). **Do not change the
pre-registered rule, predictions or grid after seeing results.** If a change is unavoidable (for
example after calibration), record it in `studyA/CHANGES.md` *before* running the grid and tell the user.

Stage 0 (already done, no PT needed): `tools/studyA/stage0_proxy.py`, output in
`tools/studyA/studyA_stage0_results.txt`. It computes the cross-cut correlation blocks that
control Theorem 2 of the Letter and fixes the predictions below.

---

## 1. Hypothesis, predictions, decision rule

**Hypothesis (Theorem 2).** For path-selecting control families, the operational rank depends on
the bath only through the correlation block projected onto the family. Spectral structure the
family does not probe raises the Schmidt rank chi_HS but not the operational rank chi_op.

**Scaling axis.** The number of narrow out-of-band spectral peaks, n = 0, 2, 4, 6. For an Ohmic bath
at this temperature the memory time is finite, so both ranks saturate in total time T (Stage 0
confirms this). Time dependence is a supplementary panel, not the main axis.

**Pre-registered predictions** (alpha = 0.05, eps = 3e-3, middle cut; linear-order proxy, so
chi ~ 1 + r_abs; tolerance +-2):

| bath | chi_HS | chi_op, Bernoulli p=0.002 | p=0.005 | coarse k=20 | p=0.01 (leaky contrast) |
|---|---|---|---|---|---|
| Ohmic | ~2 | ~2 | ~2 | ~2 | ~3 |
| + 2 out-of-band peaks | ~6 | ~2 | ~2 | ~3 | ~3 (rises to ~6 at eps=1e-3) |
| + 4 out-of-band peaks | ~10 | ~2 | ~2 | ~3 | ~3 (~7 at 1e-3) |
| + 6 out-of-band peaks | ~14 | ~2 | ~2 | ~3 | ~2 (~7 at 1e-3) |
| in-band peak at 0.8 | ~3 | rank +1..2, exponent scale x15-20 | same | same | same |

At alpha = 0.2, the proxy predicts the same Schmidt growth (2 to 14). The narrow families still stay
low at eps <= 3e-3 but leak at 1e-3, and nonlinear terms matter. Treat alpha = 0.2 as secondary.

**Decision rule.** All criteria at alpha = 0.05, eps = 3e-3, middle cut:

- **A1** chi_HS increases by >= 6 from n = 0 to n = 6.
- **A2** For at least one narrow family (p = 0.002, p = 0.005 or coarse k = 20), chi_op increases by
  <= 1 over the same range, and chi_HS / chi_op >= 3 at n = 6.
- **A3** Closing the loop, at n = 6:
  - the protected balanced compression at bond chi_op + 2 reaches a held-out RMS outcome error
    <= eps x RMS(y) on that family;
  - its step-independent certificate is valid;
  - SVD at the same bond dimension has a held-out error >= 3x larger.
- **A4** Controls and convergence:
  - the in-band peak raises chi_op by >= 2, or raises the family exponent scale >= 5x;
  - the flat family reproduces the SVD Schmidt values;
  - ranks change by <= 1 between the construction thresholds theta and theta/100, and between dt and dt/2.
- **A5** A2 holds for two narrow families (refinement of the family).

**Outcome A** if A1–A5 all hold; otherwise **Outcome B**. Report partial results as they are.

---

## 2. Physics setup

**Model.** Pure dephasing, `Boson_SysOp {|1><1|_2}`, stationary Gaussian bath, PT built from the
influence functional (no mode discretization): `use_Gaussian true` (ACE `DiagBB` path), full
memory (`t_mem >= te`). Spectral density from file: `Boson_J_from_file <file> <col_w> <col_J>`.
Check the exact keys in `src/DiagBB.cpp` and `src/SpectralDensity_Selector.cpp`. Keep
`Boson_subtract_polaron_shift` identical across all baths, and record the value.

**Units.** Time in ps, frequencies in ps^-1, hbar = 0.6582119569 meV ps. Stage 0 parameters:
dt = 0.1 ps, N = 200 (T = 20 ps), beta = 1 ps (T_temp = 7.64 K), Ohmic J = alpha w exp(-w/wc) with
wc = 5 ps^-1. Peaks are Lorentzian, `A g^2 (w/w0)/((w-w0)^2+g^2)`, with height A = 6 alpha,
width g = 0.15 ps^-1, at w0 = 12, 15, 18, 21, 24, 27 ps^-1 (all below the grid Nyquist pi/dt = 31.4).
Edge controls sit at w0 = 3 and 5; the in-band control at w0 = 0.8. ACE's J normalization may differ
from Stage 0's; **calibrate (T1)** instead of assuming.

**Families** (step-independent, defined by exact second moments):
- Bernoulli(p): an instantaneous pi pulse (X) after each step with probability p, independently.
  The second moment is Phi(Z) = (1-p) Z + p X Z X^dag. Filter width is about 2p/dt.
  p in {0.002, 0.005, 0.01}.
- Coarse(k, p_slot = 0.3): pulses allowed only after every k-th step, k in {10, 20}.
- Contexts: initial states |+>, |+i> (equal weight); readouts sigma_x, sigma_y (equal weight).
- Flat family (all basis contractions): must reproduce the Schmidt spectrum.

**Ranks.** As in Eq. (4) of the Letter: `chi(eps) = min{m : sum_{i>m} s_i^2 <= eps^2 sum_i s_i^2}`.
Record `sum_i s_i^2` so absolute values can be recovered. eps in {1e-2, 3e-3, 1e-3}. Report both
the middle cut and the maximum over cuts.

---

## 3. Tasks and acceptance tests

**T0 Prerequisites.** Run on Linux, or resolve the Mac discrepancy first (the smoke test must give
0.516043). Put all Study A code in `studyA/` and tools in `src_exec/tools/`.

**T1 Bath inputs and calibration.**
- Generate the J files (Ohmic + n peaks; edge; in-band) on a fine grid (spacing <= g/20) in ACE units.
- *Calibration test:* for a context with no pulses and no H_S, the coherence from ACE must match
  the analytic Gaussian dephasing result rho_01(t) = rho_01(0) exp(-Gamma(t) - i Theta(t)) computed
  from the same J, to <= 1e-6 relative. This fixes the J convention.
- Then choose alpha_ACE such that the family exponent scale ||B_F|| from the Stage 0 proxy (fed
  with ACE's convention) is about 0.05 (primary) and about 0.2 (secondary). Record the mapping.

**T2 Pulse-sequence contexts.**
- Per-step system operations S_k in {I, X} applied at full-step boundaries. Use the same placement
  in the trajectory route, in the second-moment route, and in `Simulation_PT`.
- Sampler for Bernoulli and coarse families; second-moment operator for each.
- Tests:
  - (a) one sampled sequence gives the same outcome through the new context path and through
    `Simulation_PT`;
  - (b) for pure dephasing without H_S, the outcome equals exp(-chi_y), the exact filter-function
    expression for that switching function, to <= 1e-8 relative;
  - (c) Gramians from N_s sampled sequences converge to the second-moment Gramians, with the
    difference scaling like 1/sqrt(N_s).

**T3 Second-moment (route R) Gramians with the system index open.**
- Forward recursion `W <- Q (Phi (x) id)[W] Q^dag`; backward analogue; absorb the system index
  only at the cut (Sec. S1 of the Supplemental Material).
- Test: reproduce `tools/balanced_ref/certificate_iid.py` (exact second moments) on the exported
  toy PT to <= 1e-10. Cost O(d^6 D^3) per step.

**T4 Spectrum export tool `PTB_memory_spectra`.**
- For chosen cuts, write the balanced singular values for each family and for the flat family,
  plus `sum s_i^2`. The flat sums need dictionary multiplicities.
- Tests:
  - the flat spectrum equals the Schmidt values from a canonical SVD sweep of the same PT, to 1e-8
    relative;
  - the spectra are invariant under a random bond gauge.

**T5 Closing the loop.**
- For each bath, compress with the protected balanced projector, using the family's route-R
  Gramians, with eta = 0 and eta = 0.3 (gauge-invariant blend). Also compress with SVD at the same
  bond dimension. Bond dimensions: chi_op(eps) + 2 and a few larger values.
- Evaluate on 200 held-out sequences from the family and 200 out-of-family sequences (p x 4).
- Report RMS outcome errors, the step-independent certificate `d sum_k (tail_k)^{1/2}`, and the
  paired certificate on the held-out set.
- The Bernoulli and coarse families are step-independent, so Theorem 1(iii) applies directly.

**T6 Runs and analysis.**
- Run the grid in Sec. 4, write CSV and spectra, produce the Fig. 2 panels, and compare Stage 0
  predictions with the actual ranks.

---

## 4. Run grid and order

| block | alpha | baths | N, dt | thresholds | purpose |
|---|---|---|---|---|---|
| **Pilot** (stop and report) | 0.05 | Ohmic, +2 peaks | 200, 0.1 | theta, theta/100 | T1–T5 end to end; compare with Stage 0 |
| Primary | 0.05 | Ohmic, +2/4/6 peaks, edge 3, edge 5, in-band | 200, 0.1 | theta, theta/100 | A1–A5 |
| Secondary | 0.2 | same 7 baths | 200, 0.1 | theta | robustness, nonlinear regime |
| Time (SM) | 0.05 | +2 peaks | 100 and 400, 0.1 | theta | saturation in T |
| Step (SM) | 0.05 | +2 peaks | 400, 0.05 | theta | dt convergence |
| Stretch (optional) | 0.05 | sub-Ohmic s = 0.5, beta = 5 | 100/200/400 | theta | long memory |

Choose theta (ACE's relative SVD threshold) so that the flat-family chi_HS at eps = 1e-3 is
converged. Start with 1e-7 and 1e-9. A threshold that is too loose underestimates chi_HS; that bias
works against Outcome A, so note it but do not "correct" it.

**Stop after the pilot** and send the user:
- the calibration result;
- the measured vs predicted ranks for the two pilot baths;
- the closing-the-loop numbers;
- runtimes.

The user decides whether to run the full grid.

---

## 5. Outputs

- `studyA/results/ranks.csv`: bath, alpha, N, dt, theta, cut, family, eps, chi, sum_s2, exponent_scale.
- `studyA/results/spectra/*.txt`: singular values per (bath, family, cut).
- `studyA/results/heldout.csv`: bath, family, method (balanced eta / SVD), bond, set (in/out),
  rms_err, rms_y, cert_stepindep, cert_paired.
- `studyA/figs/fig2_{a,b,c,d}.pdf`:
  - (a) chi vs n at alpha = 0.05, eps = 3e-3, for HS and all families;
  - (b) the same at alpha = 0.2;
  - (c) controls (edge and in-band peaks) with exponent scales;
  - (d) held-out error vs bond dimension for balanced vs SVD at n = 6, with certificates.
- `studyA/REPORT.md`: A1–A5 pass/fail with numbers, plus every deviation from this spec.

Do not edit the paper. Report numbers; the user decides the narrative.

---

## 6. Pitfalls

- **Conventions.** Calibrate J, sign and normalization in T1 before anything else.
- **Dictionary.** Outer indices are compressed; flat sums need multiplicities.
- **Memory cutoff.** The peaks decay as exp(-0.15 t); reaching 1e-4 takes about 60 ps > T, so do
  not truncate memory.
- **Pulse placement.** Placement relative to the symmetric Trotter halves must be identical in
  all three routes.
- **Protected bond.** The protected projector needs total bond chi_op + 2 for the same tail
  (rank-one interlacing). Report both the unprotected chi_op and the protected bond.
- **Peaks near Nyquist.** Do not move peaks above pi/dt. Do not change peak positions or widths
  after the pilot without recording it.
- **Nonlinearity.** Stage 0 is a linear-order proxy. At alpha = 0.2, deviations from the predictions
  are expected and are information, not failure.
