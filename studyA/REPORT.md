# Study A report: predictive vs representational memory under Gaussian dephasing

Spec: `docs/studyA_spec.md`. Deviations: `studyA/CHANGES.md` (all recorded before the grid ran; the theta
choice on 2026-11-23). Numbers: `studyA/results_cirrus/` (Cirrus grid, commit c6822ab), reproduced by
`python studyA/analyze_grid.py` (full printout in `results_cirrus/analysis.txt`). Figures: `studyA/figs/fig2_{a,b,c,d}.pdf`.

Status 2026-11-24: Primary block complete at theta = 1e-6, except oob6 at theta/10 (PT rebuild running). The dt/2
convergence check (Step block) and the Time and Stretch blocks have not been run. The secondary block (alpha = 0.2)
has no n = 4, 6 PTs yet.

## Verdict under the pre-registered rule

All values at alpha = 0.05, eps = 3e-3, middle cut (cut 99), theta = 1e-6.

| criterion | result | key numbers |
|---|---|---|
| A1 | **pass** | chi_HS = 2, 6, 10, 14 for n = 0, 2, 4, 6 (increase 12; >= 6 required) |
| A2 | **pass** | p = 0.002: chi_op 2, 2, 2, 2; p = 0.005: 3, 3, 3, 3; coarse k = 20: 2, 2, 2, 2; chi_HS / chi_op at n = 6: 7.0, 4.7, 7.0 |
| A3 | **fail** | at bond chi_op + 2 no narrow family reaches error <= eps RMS(y) (details below); the certificate and SVD clauses pass |
| A4 | **incomplete; one clause fails** | in-band control passes; theta convergence passes for n = 0, 2, 4, fails for the in-band control, oob6 pending; dt/2 not run |
| A5 | **pass** | A2 holds for 3 narrow families (2 required) |

**Outcome B**: A3 fails, so not all of A1–A5 hold. The A4 data still pending cannot change this.

## A1, A2, A5: ranks versus Stage 0 predictions (Fig. 2a)

| family | n = 0 | 2 | 4 | 6 | Stage 0 prediction (+-2) |
|---|---|---|---|---|---|
| HS (flat) | 2 | 6 | 10 | 14 | 2, 6, 10, 14 |
| Bernoulli p = 0.002 | 2 | 2 | 2 | 2 | 2, 2, 2, 2 |
| Bernoulli p = 0.005 | 3 | 3 | 3 | 3 | 2, 2, 2, 2 |
| coarse k = 20 | 2 | 2 | 2 | 2 | 2, 3, 3, 3 |
| Bernoulli p = 0.01 (leaky contrast) | 3 | 3 | 3 | 3 | 3, 3, 3, 2 |
| coarse k = 10 | 2 | 3 | 3 | 3 | (not tabulated) |

Every measured rank lies within the +-2 tolerance of the predictions. chi_HS matches them exactly.

## A3: closing the loop at n = 6 (Fig. 2d)

Protected balanced compression with the family's route-R Gramians at eta = 0. Errors are held-out RMS outcome
errors on 200 in-family sequences x 2 initial states x 2 readouts, relative to the input PT.

| family | chi_op | bond | balanced error | eps RMS(y) | step-indep. cert. | paired cert. | SVD error | SVD / balanced |
|---|---|---|---|---|---|---|---|---|
| p = 0.002 | 2 | 4 | 1.01e-4 | 4.47e-5 | 7.66e-3 | 1.18e-3 | 5.07e-3 | 50x |
| p = 0.005 | 3 | 5 | 1.73e-4 | 4.76e-5 | 1.35e-2 | 1.76e-3 | 4.68e-3 | 27x |
| coarse k = 20 | 2 | 4 | 7.29e-4 | 5.30e-5 | 1.11e-2 | 3.14e-3 | 7.17e-3 | 10x |

- **Error clause: fails for all three families.** The best relative error is 6.8e-3, for p = 0.002, against the
  target of 3e-3.
- **Certificate clause: passes.** Both certificates bound the error in all 440 balanced rows of the grid (all baths,
  families, bonds and eta, in and out of family).
- **SVD clause: passes,** by 10–50x.

Context, recorded before the grid (CHANGES.md, 2026-11-19): outcomes here are about 0.015–0.02, not O(1),
because Gamma(20 ps) ~ 3.9. The A3 threshold is therefore about 5e-5 in absolute terms.

**Supplementary (bond not pre-registered), n = 6, in-family error by bond, as balanced eta = 0 / eta = 0.3 / SVD:**

| family | bond chi_op+2 | +4 | +8 | 16 |
|---|---|---|---|---|
| p = 0.002 | 1.0e-4 / 4.4e-3 / 5.1e-3 | 6.1e-5 / 8.9e-4 / 4.9e-3 | **3.1e-5** / 3.9e-4 / 5.5e-3 | **2.7e-5** / 4.8e-4 / 2.2e-3 |
| p = 0.005 | 1.7e-4 / 5.6e-3 / 4.7e-3 | 9.4e-5 / 4.8e-3 / 4.9e-3 | 7.9e-5 / 4.7e-4 / 6.2e-3 | 6.4e-5 / 6.3e-4 / 1.8e-3 |
| coarse k = 20 | 7.3e-4 / 6.4e-3 / 7.2e-3 | 2.3e-4 / 6.2e-3 / 6.9e-3 | **1.2e-5** / 4.9e-3 / 7.4e-3 | **4.0e-7** / 3.9e-4 / 9.1e-4 |
| p = 0.01 | 3.8e-4 / 6.4e-3 / 5.3e-3 | 2.0e-4 / 5.7e-3 / 5.5e-3 | 1.4e-4 / 6.1e-4 / 6.7e-3 | 8.7e-5 / 7.5e-4 / 1.4e-3 |

Bold entries meet eps RMS(y).
- p = 0.002 and coarse k = 20 meet the threshold at bond 10 (chi_op + 8), with SVD 180x and 600x worse there.
- p = 0.005 does not meet it up to bond 16.

Observations, not investigated:
- **Bernoulli families level off.** At eta = 0 the error stops falling from bond ~10, at a floor that rises with p
  (3e-5, 6e-5, 9e-5). Coarse k = 20 keeps falling.
- **The eta = 0.3 blend is close to SVD at small bonds** and 10–100x worse than eta = 0 throughout.
- **Trace error has a floor.** Balanced runs have max |tr rho - 1| = 6.8–7.0e-4 at every bond, against 2–4e-2 for
  SVD. With closure protection, this floor is most likely the input PT's own trace error at theta = 1e-6 (as in the
  joined-cavity benchmark). Not checked here.
- One possible reason why chi_op + 2 is not enough: chi_op(eps) bounds a relative tail at one cut, while the outcome
  error accumulates over 199 cuts and is measured relative to a small RMS(y). Untested.

## A4: controls and convergence (Fig. 2c)

**In-band peak at 0.8, versus Ohmic: passes.** chi_op rises by 4 to 11, depending on the family. The exponent scale
rises 12–54x.

| family | chi_op Ohmic -> in-band | exponent scale (Stage 0 proxy) |
|---|---|---|
| p = 0.002 | 2 -> 6 | 0.060 -> 0.72 (x12) |
| p = 0.005 | 3 -> 9 | 0.053 -> 0.86 (x16) |
| coarse k = 20 | 2 -> 11 | 0.038 -> 1.37 (x36) |
| p = 0.01 | 3 -> 12 | 0.044 -> 1.05 (x24) |
| coarse k = 10 | 2 -> 13 | 0.025 -> 1.35 (x54) |

**Edge peaks:** edge 5 and edge 3 raise chi_HS to 4. chi_op rises for some families (up to 6 for coarse k = 10 at
edge 3); see Fig. 2c.

**Flat family reproduces SVD Schmidt values:** covered by the T4 unit test (`tests/test_studyA.cpp`, max relative
deviation 3.7e-12 for s_i >= 1e-4 s_1). It was not repeated for each grid PT.

**Threshold convergence (CHANGES.md: theta vs theta/10; theta/100 and theta/1000 where built).** The table gives the
largest |Delta chi| over all six families at eps = 3e-3, middle cut.

| bath | 1e-7 | 1e-8 | 1e-9 |
|---|---|---|---|
| Ohmic | 1 | 1 | 1 |
| +2 peaks | 1 | 1 | 1 |
| +4 peaks | 1 | not built | not built |
| +6 peaks | **pending** (rebuild running) | not built | not built |
| edge 5, edge 3 | 0 | 0 | 0 |
| in-band 0.8 | **2** (p = 0.01) | **3** (coarse k = 10) | **2** (p = 0.01) |

- In the Ohmic and peak baths, every change of 1 is Bernoulli p = 0.005 (3 at theta = 1e-6, 2 below).
- chi_HS is identical at all four thresholds for every bath.
- The in-band control's family ranks fluctuate non-monotonically with theta. Coarse k = 10 reads 13, 14, 16 and 13
  from theta = 1e-6 to 1e-9. This fails the convergence clause, and would also fail under the spec's original
  theta/100 reading. Whether the clause applies to the control baths is not stated in the spec.

**dt vs dt/2:** not run (Step block).

## Secondary block, alpha = 0.2 (Fig. 2b)

| family | Ohmic | +2 peaks |
|---|---|---|
| HS (flat) | 3 | 21 |
| p = 0.002 | 3 | 3 |
| p = 0.005 | 4 | 5 |
| coarse k = 20 | 3 | 4 |
| p = 0.01 | 4 | 8 |
| coarse k = 10 | 3 | 7 |

- chi_HS at +2 peaks is 21, against the proxy's ~6: the nonlinear regime goes well beyond the linear proxy.
- The narrowest family (p = 0.002) stays at 3.
- n = 4, 6 PTs are being rebuilt (job 727438); at theta = 1e-6 they did not finish in 48 h.
- The in-band ranks at alpha = 0.2 are not converged in theta (for example p = 0.002: 26, 36, 30 and 39 for theta =
  1e-6 to 1e-9).

## Deviations and incidents (details in CHANGES.md)

- **T1:** K_n from Gauss-Legendre quadrature of the J files, passed with `Boson_read_K_int`; ACE's own quadrature
  misses 1e-6. J_ACE = (4/pi) J_stage0.
- **PT construction:** divide-and-conquer over 256 steps, cut to 200. Theta = 1e-6 primary, with theta/10 for A4.
  These builds were not feasible within the time limit: oob6 at theta <= 1e-7; oob4 at <= 1e-8; alpha = 0.2 oob4
  and oob6 at all thresholds; alpha = 0.2 oob2 at <= 1e-8.
- **T2(b):** run on the Ohmic bath.
- **Conventions:** pulse slots, final-time readout, sigma_y sign, middle cut = 99, out-of-family sets.
- **Closing-the-loop runs:** split per bond dimension, with unchanged numerics. The pilot runs used theta = 1e-7.
- **Crash:** one run (in-band 0.8, coarse k = 20, bond 15) aborted on Cirrus with an Eigen 3.4.0 BDCSVD index
  assertion in a diagnostic SVD. It does not reproduce on the Mac. Fix (2026-11-24): diagnostic and spectrum SVDs use
  JacobiSVD up to size 512. This run is missing from heldout.csv until it is rerun with the fixed build.
