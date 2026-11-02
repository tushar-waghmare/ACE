# Porting the global error certificate into the balanced sweep

Add `@docs/certificate_port.md` to `CLAUDE.md` next to the plan import.

Ground truth: `tools/balanced_ref/certificate_iid.py` (step-independent ensemble, exact second
moments + Monte Carlo) and `tools/balanced_ref/certificate_paired.py` (Hamiltonian-family
contexts, the realistic case). Theory: Proposition "global ensemble certificate" of the
manuscript (Eqs. 26-31 of the edited version; not kept in this repository).

## What the certificate says

For a forward sweep that inserts `Pi_k = X_k Y_k` at cuts `k = 0 .. n_tot-2`:

```
y(c) - y~(c) = sum_k e_k(c)                               (exact, every context c)
e_k(c)       = sum( chi_k(c) .* ( state~_k(c) - state~_k(c) * Y_k^T * X_k^T ) )
sqrt(E|y - y~|^2) <= sum_k sqrt(E |e_k|^2)                (any weights, any projectors)
```

* `state~_k(c)`: augmented state (`NL x D`) of context `c` at cut `k`, propagated through the
  ALREADY TRUNCATED part (cuts `< k` inserted), cut `k` not yet truncated. This is exactly the
  past state the sweep keeps per context.
* `chi_k(c)`: future tester of the SAME context `c` at cut `k`, from the UNTRUNCATED chain
  (the backward precompute of Phase 4).
* `.*` and `sum`: elementwise product and full sum (bilinear pairing, no conjugation).

Only for step-independent ensembles (every step's propagator drawn independently) does
`E|e_k|^2` equal the product-ensemble value, and then `E|e_k|^2 <= d^2 * tail_k` with
`tail_k = sum_{i>r} sigma_i(K)^2` (already logged). For Hamiltonian families (one H for all
steps) the product value is NOT a bound; `certificate_paired.py` shows a failing case (m = 6).

## Implementation

1. **Use one context list for both sides.** The paired terms need `state~_k(c)` and `chi_k(c)`
   for the same `c`. If pasts and futures currently use different lists, either (a) also
   compute backward testers for the past contexts (one extra backward pass per context), or
   (b) report only the product version and label it "valid for step-independent ensembles".
2. **Accumulate during the forward sweep.** At cut `k`, after computing `X_k, Y_k` and before
   updating the stored states with `state <- state * Y_k^T`:
   `e_k(c) = sum(chi_k(c) .* (state_k(c) - (state_k(c) * Y_k^T) * X_k^T))` for every context;
   `paired_k = sum_c w_c |e_k(c)|^2` with normalized weights `w_c`.
3. **Report.**
   * Per cut: existing diagnostics plus `paired_k`.
   * At the end: `certificate = sum_k sqrt(paired_k)`, and for comparison
     `d * sum_k sqrt(tail_k)`.
   * Parameter `balanced_validation_context <file>` (repeatable): propagate those contexts as
     well (hybrid pasts, exact futures) and report their certificate separately. Optionally run
     them through the source and the truncated PT and report the actual RMS error next to it.
4. **Scope notes in the log/output header.** The certificate is relative to the sweep's INPUT
   PT, not to the uncompressed physics. Errors of earlier stages (mode merges) add by the
   triangle inequality. Repeated sweeps: one certificate per sweep, summed.

## Tests to port (C++ test harness)

* **Telescoping identity:** for each training and held-out context,
  `|(y - y~) - sum_k e_k| < 1e-10 * max(1, |y|)`, with `y` from the source PT and `y~` from the
  truncated PT via the normal propagation path. This is the decisive check; an indexing or
  conjugation error breaks it immediately.
* **Certificate:** `RMS(y - y~) <= certificate` on training and on validation contexts.
* **Closure corollary** (if T8 is not already ported): `tr rho~(t_k) = 1` at all cuts for
  held-out trace-preserving contexts, to ~1e-12.
* Optional: reproduce the table of `certificate_paired.py` on the same toy, exporting the
  toy PT from Python if a C++ builder for it is inconvenient.

## Cost

Zero extra propagations when the lists coincide: `e_k(c)` reuses the stored states and
testers. Extra work per cut: one `NL x D` by `D x m` by `m x D` product per context.
