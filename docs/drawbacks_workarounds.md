# Drawbacks 1-11: workarounds, and what changes with the global certificate

Evidence labels: **[tested]** on the exact toy PT (`tools/balanced_ref`); **[proved]** follows from
the certificate proposition of the manuscript; **[derived]** short argument,
not yet tested numerically; **[proposed]** design idea; **[rejected]** tested and dropped.

The certificate (`docs/certificate_port.md`) bounds the ensemble RMS outcome error of a whole
sweep by `sum_k sqrt(E|e_k|^2)`, with no amplification factors, for any projectors. It
mainly changes how we *decide and certify*. It does not remove the need for coverage.

| # | Drawback | Recommended workaround | Changed by certificate? |
|---|---|---|---|
| 1 | Accuracy tied to training contexts | Master PT + derived balanced PTs; coverage by design; validation guard | Yes: the guard becomes a rigorous bound |
| 2 | Small gain with few contexts | Constraint-defined ensembles; greedy selection; deploy only if validated | Slightly: paired certificate decides balanced vs SVD |
| 3 | eta is a judgement call | Choose eta by held-out validation, or by an HS budget | Yes: validation gives model selection with a bound |
| 4 | HS error always worse than SVD | HS budget kappa (HS error <= kappa x SVD optimum) | No |
| 5 | Oblique amplification | Keep noise floor + logging; balanced normal form later | Yes: downgraded, `||X|| ||Y||` absent from the certificate |
| 6 | Not in normal form, must be last | Allow later truncations with per-sweep certificates; unified criterion later | Yes: relaxed |
| 7 | Positivity not guaranteed | Measure; mean-square negativity bounded by the certificate | Yes: new average bound [derived] |
| 8 | Only the final sweep is balanced | Over-provision intermediate sweeps; eta schedule | Yes: error budget attribution |
| 9 | Cost grows with contexts | Second moments for broad families; parallel contexts | Slightly: unify past/future lists |
| 10 | Tied to one setup | Uniform PTs next | No |
| 11 | Multiple baths need an exact join | Truncate each PT with the other bath as context | Yes: same guarantee extends [derived] |

## Details

**1. Accuracy tied to training contexts.**
Treat the master PT (SVD or looser threshold) as the product and balanced PTs as cheap derived
products per control family (seconds to build at 30 modes). Build coverage in by design:
(a) constraint-defined ensembles (amplitude <= Omega_max, bandwidth <= B) via the second-moment
route, so that every control within the limits is covered; (b) greedy selection from a
candidate pool, repeatedly adding the context whose past states have the largest residual
outside the kept subspace [proposed].
*Changed:* the out-of-distribution guard is now the paired certificate on a validation set,
computed against the master's exact futures, rather than an ad hoc comparison [proved].
The coverage result (dephasing-only training loses to SVD on pulsed contexts at small m)
remains a genuine limitation and belongs in the paper as a coverage requirement.

**2. Small gain with few contexts.**
With few contexts, the product-ensemble tails are unreliable for Hamiltonian families
(`certificate_paired.py`, m = 6). Decision rule: deploy the balanced PT only if its
validation error or certificate beats SVD's at the same m. Otherwise fall back to eta = 1.

**3. eta is a judgement call.**
Option A: choose eta by held-out validation error, computed exactly by running the validation
contexts through the source and the truncated PT, and report the certificate. Option B:
bisect eta so that the HS error stays <= kappa x SVD's HS optimum (a user-facing guarantee in
the HS norm). Keep 0.3 as the default only until one of these is in place.

**4. HS error always worse than SVD.**
Unchanged: it is a property of the objective. Option B above makes it a controlled trade-off
(kappa). The certificate machinery also applies to SVD sweeps, so both methods can be
reported in the same operational framework.

**5. Oblique projector amplification.**
*Changed (downgraded):* `||X|| ||Y||` does not enter the ensemble certificate [proved]. It still
matters for contexts outside the averaging ensemble, for floating-point conditioning
(Sigma^{-1/2}), and for any later operation on the tensors. Keep the noise-floor rule and the
logging. Structural fix for later: store bonds in balanced normal form, where W_p = W_f = Sigma
and truncation is plain coordinate deletion [proposed].

**6. Output not in normal form; balanced sweep must be last.**
*Changed (relaxed):* a later truncating sweep is acceptable if it reports its own certificate
under the same operational ensemble, relative to its input. Sweep certificates add by the
triangle inequality [proved]. The clean long-term option is still a single criterion
everywhere (blended Gramians with an eta schedule) [proposed].

**7. Positivity of the reduced state not guaranteed.**
First, measure it: the minimum eigenvalue of the Hermitian part of rho~(t) over held-out
contexts and times, for SVD and balanced.
*Changed [derived, test together with the measurement]:* closure protection keeps tr rho~ = 1
exactly, so for a qubit the deviation is traceless. If the observable ensemble is isotropic
(random Pauli directions, independent of H and rho0), then E_O|Delta<O>|^2 = (2/3)||Delta rho||_F^2.
By Weyl's inequality,
`E[ max(0, -lambda_min(rho~))^2 ] <= (3/2) E|y - y~|^2 <= (3/2) * certificate^2`.
Negativity is therefore controlled on average; a worst-case statement is not implied.
Exact Hermiticity via ensemble symmetry (bra-ket involution) remains an idea [proposed].

**8. Only the final sweep is balanced.**
Over-provision the intermediate SVD sweeps (looser threshold or maxk than the final target)
and let the balanced final sweep cut to the target. Research option: eta schedule across
merges, or a two-pass tree with pass-1 subtrees as approximate complements [proposed].
*Changed:* error budget attribution. Total error <= (merge-stage error, measured as master vs
a tighter reference) + (final-sweep certificate). If the certificate is much smaller than the
merge-stage error, effort belongs in the merges, not in the final sweep.

**9. Cost grows with the number of contexts.**
Second-moment (recursion) route for broad families, so that cost is independent of the number
of contexts. Contexts are embarrassingly parallel. Store absorbed factors (rank <= n d^2), not
Gramians.
*Changed:* the paired certificate is free when past and future use the same context list, which
argues for unifying the lists. The D = 1620 probe is still pending.

**10. Tied to one setup (time grid, symmetric Trotter; PT_infinite, uniACE, TEMPO, PTR,
CompressionTree, merge-stage truncation out of scope).**
Unchanged. The next target is uniform PTs: one stationary Gramian equation and one projector
for all bonds. Stability preservation for balanced truncation of generalized Lyapunov
operators is known, but needs the protected closure pair deflated first, because trace
preservation makes the stationary map only marginally stable.

**11. Multiple baths need an exact join (D and disk blow up).**
Truncate each PT separately, with the other bath as context. Propagate with
`ProcessTensorForwardList` (the augmented state spans both bonds), absorb the other PT's bond
together with mu into the labels, and alternate between the PTs. Alternative: join and truncate
element by element, with futures computed through the list [proposed].
*Changed [derived]:* the certificate extends. Insertions on either PT's bond are just further
insertions in the telescoping (insertions at the same cut combine as Pi_1 (x) Pi_2), and each
term pairs the hybrid past of the joint system with its exact future.

## Tested and rejected (do not pursue without new evidence)

* **Haar-random per-step ensemble as a "general-purpose" operational PT** [rejected]: about equal
  to SVD (modest gain at m = 4, identical at m = 6).
* **Protecting the top Schmidt directions, or filling spare slots with HS directions** [rejected]:
  worse than SVD out of distribution. The protected variant blew up at m = 6 because
  noise-dominated directions were amplified.
* **Unregularized ALS refinement as the default** [rejected]: 10x better in distribution, 4x worse
  than SVD out of distribution, with `||XY||` up to 1e5.

## Suggested order

1. Port the certificate. It is cheap, and it underpins the decisions in 1, 2, 3, 6 and 8.
2. Positivity measurement, together with the bound in 7.
3. eta by validation (A) and the HS budget (B).
4. Greedy context selection and constraint-defined ensembles.
5. Complement-as-context for multiple baths.
6. Uniform PTs.
7. Balanced normal form.
