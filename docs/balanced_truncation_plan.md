# Operational balanced truncation for ACE PT-MPOs: implementation plan

Companion to the manuscript *Balanced truncation of process-tensor MPOs* (protected, square-root
balanced truncation with operational Gramians). This plan was written against ACE commit
`49b151e` (2026-10-16). Re-check file/line references if the upstream code has moved.

Ground truth for every formula below: `tools/balanced_ref/balanced_ref.py`
(numpy; `python tools/balanced_ref/balanced_ref.py` runs tests T1–T8 and a small benchmark).
The C++ implementation must reproduce its numbers on the same toy problem.

---

## 0. Goal and scope

ACE truncates PT-MPO bonds by SVD in sweeps, which minimizes the Hilbert–Schmidt (HS) error.
We add an alternative truncation that minimizes the error of a specified ensemble of
**contexts** (system Hamiltonians/pulses, initial states, observables), with the PT closure
protected exactly.

**v1 scope (this plan):** post-processing of a finished PT-MPO, i.e. a balanced *final sweep*.
Deliverable: library code + a standalone tool `PTB_balanced_truncate` (reads `read_PT`, writes
`write_PT`), modelled on `src_exec/tools/PTB_sweep_forward.cpp`.

**Out of scope for v1** (do not touch, guard with clear errors if encountered):
multiple PTs in `ProcessTensorForwardList` (`expand_front/back`), `CompressionTree` (`TTree`),
`PT_infinite`, `PT_uniACE`, `ProcessTensorRepeat`/PTR, TEMPO/CTEMPO paths, and changing the
truncation used *during* `add_modes`/`add_modes_tree`. Existing default behaviour must not change.

---

## 1. Conventions (critical)

ACE stores each PT element as `MPS_Matrix M` with `M(i, d1, d2)`: `d1` = bond to the previous
element, `d2` = bond to the next, `i` = outer index, a dictionary-compressed pair
(output Liouville index, input Liouville index); see `IF_OD_Dictionary` (`beta[i*NL+j]`).

The augmented system–bond state is an `Eigen::MatrixXcd state` of shape `NL x D`
(rows: system Liouville index, cols: bond). `ProcessTensorElementAccessor::propagate` does

```
state2.row(i) += state.row(j) * M_slice(dict(i,j))      // row vector on the bond
```

| Object | ACE | Paper |
|---|---|---|
| bond vector for system index mu | `state.row(mu)` (1 x D) | `p_mu = state.row(mu)^T` (no conjugation) |
| element | `M(beta, d1, d2)`, slice `M_beta` (D1 x D2) | `Q^beta = M_beta^T` |
| future tester | `chi` (NL x D), `y = sum(chi .* state)` | `f_mu = chi.row(mu)^T`, `y = sum_mu f_mu^T p_mu` |
| closure at cut n | `e_n.closure` (on d2 of element n) | `q` (`q^T p` = trace readout) |
| insert projector `Pi = X Y` on bond n | `M_n <- M_n * Y^T`, `M_{n+1} <- X^T * M_{n+1}` | `Q_n <- Y Q_n`, `Q_{n+1} <- Q_{n+1} X` |
| | `M_n.inner_multiply_right(Y^T)`; `M_{n+1}.inner_multiply_left(X^T)` | |
| closure / env ops update | `closure <- X^T closure`; `env_ops.process_forward(PassOn{P=X^T, Pinv=Y^T})` | `q^T <- q^T X` |

Pairings are **bilinear** (no conjugation). Conjugation appears only in Gramians/norms.
`Simulation_PT::propagate_state` uses a symmetric Trotter step by default:
`half system step -> PT element -> half system step`. Contexts must reproduce this exactly.
The cut `n` is the `d2` bond of element `n`, after the trailing half step.

---

## 2. Math to implement

For cut `n` (bond between elements `n` and `n+1`), `D = e_n.M.dim_d2`, `NL = N_sys^2`:

1. **Past factor** `A_n` (`D x (n_p*NL)`): for each past context `a` with weight `w_a`, propagate
   the initial augmented state through elements `0..n` (with that context's system propagator).
   In ACE the initial environment state is absorbed into the first element, so the initial
   state is `vec(rho0_a)` with `cols == e_0.M.dim_d1` (expected 1; copy exactly what
   `Simulation_PT` does when it sets up `state`, and assert the dimension);
   columns are `sqrt(w_a) * state_a.row(mu)^T` for all `mu`. (The system index is *absorbed* into
   the column label.)
2. **Future factor** `F_n` (`(n_f*NL) x D`): for each future context `b`, start from
   `chi = o_b (x) closure_last` (o_b with `o_b . vec(rho) = tr(O_b rho)`) and propagate **backward**
   with the adjoint of each step down to cut `n`; rows are `sqrt(v_b) * chi_b.row(mu)`.
   Adjoint of one element: `chi_in.row(j) += chi_out.row(i) * M_slice(dict(i,j))^T`;
   adjoint of a system step `S` (acting on rows): `chi <- S^T chi`.
3. **Protected pair**: `q = e_n.closure`;
   `rhat = mean_a sum_nu state_a.row((nu,nu))^T` (trace-weighted mean bond state; gives
   `s = q^T rhat = 1` for an exact closure, so always well-defined). Abort if `|s| < tol`.
4. **Deflate**: `P_perp = I - rhat q^T / s`, `A_perp = P_perp A`, `F_perp = F P_perp`,
   `K = F_perp A_perp = U Sigma V^dagger` (thin SVD; `K` is small: `(n_f NL) x (n_p NL)`).
5. **Rank**: `r` from `Sigma` with the usual ACE rules (relative `threshold`, `maxk`, `mink`)
   applied to the core singular values; total new bond dimension `m = r + 1`.
6. **Factors**: `X = [rhat/s, A_perp V_r Sigma_r^{-1/2}]` (`D x m`),
   `Y = [q^T ; Sigma_r^{-1/2} U_r^dagger F_perp]` (`m x D`). Check `Y X = I_m` (tolerance, log).
7. **Apply** as in the table in Section 1; then `clearNF()` on both elements.
8. **Diagnostics to log per cut**: `D -> m`, `sum_{i>r} sigma_i^2 / sum_i sigma_i^2` (local
   operational error), `||X||_2 ||Y||_2` (oblique amplification), `|s - 1|`.

**Sweep schedule** (forward): precompute and store all `F_n` on the untruncated chain
(backward pass), then go forward: at cut `n`, compute `A_n` from trajectories propagated through
the *already truncated* past (keep the per-context augmented states; after truncating cut `n`
update them as `state <- state * Y^T`, then propagate one more element). `F_{n+1}` stays valid
because it depends only on elements `> n+1`, which are untouched.

**Guarantees to test** (all proven in the manuscript, all checked in the reference):
`Pi^2 = Pi`, `q^T Pi = q^T`, `Pi rhat = rhat`, local error = core tail (T3);
`rho_S(t_n)` read out at the cut is exact for *any* past (T4);
a protected sweep keeps `tr rho_S(t) = 1` at all times for any trace-preserving
context (T8). The SVD baseline does not have T4/T8.

**Optional (Phase 5)**: HS Gramians of the bare PT via flat recursions
`Wp <- sum_beta M_beta^T Wp M_beta^*` (forward from `r0 r0^dagger`),
`Wf <- sum_beta M_beta^* Wf M_beta^T` (backward from `q^* q^T`), with **dictionary
multiplicities** (or on `expand_from_dict()` copies), and the gauge-invariant blend
`Wp(eta) = (1-eta) Wp_op/tr(Wp_op Wf_hs) + eta Wp_hs/tr(Wp_hs Wf_hs)`,
`Wf(eta) = (1-eta) Wf_op/tr(Wp_hs Wf_op) + eta Wf_hs/tr(Wp_hs Wf_hs)`.
Factors from Gramians via `SelfAdjointEigenSolver` (`A = V sqrt(L)`, `F = sqrt(L) V^dagger`).

---

## 3. ACE architecture map (what to read, what to touch)

| File | Role | Use |
|---|---|---|
| `include/ProcessTensorElement.hpp`, `src/ProcessTensorElement.cpp` | element: `M`, `closure`, `env_ops`, `forwardNF/backwardNF`; `sweep_forward/backward` (SVD), `calculate_closure`, `expand_from_dict` | template for applying X, Y; read `sweep_forward` first |
| `src/ProcessTensorElementAccessor.cpp` | `propagate(state, M, dim1_front, expand)` with dictionary; `closure_indices()` | add `propagate_adjoint` next to it |
| `include/MPS_Matrix.hpp`, `src/MPS_Matrix.cpp` | `inner_multiply_left/right` | apply X^T / Y^T |
| `include/PassOn.hpp` | `{P, Pinv}` passed between elements | `P = X^T`, `Pinv = Y^T` for `env_ops` |
| `src/ProcessTensorBuffer.cpp` | file-backed chain (`get(n, ForwardPreload)`), `sweep_forward/backward`, `add_modes`, `sweep_intermediate_or_final_start_forward` | add `sweep_forward_balanced(...)`; later hook into the final sweep |
| `src/TruncatedSVD.cpp`, `src/TruncationLayout.cpp` | rank rules; param names (`threshold`, `compress_maxk`, `final_sweep_*`) | reuse `get_truncated_dim` for `r` |
| `src/Simulation_PT.cpp` | `propagate_state` (symmetric Trotter), `propagate_system` | contexts must match this exactly |
| `FreePropagator`, `Parameters`, `TimeGrid` | build system propagators from param files | one `FreePropagator` per context |
| `src_exec/tools/PTB_sweep_forward.cpp`, `PTB_apply_system_propagator.cpp` | standalone PT tools | templates for `PTB_balanced_truncate` |
| `Makefile` (`TOOLS`, `tools:` target) | builds `src_exec/tools/*.cpp` into `tools/` | add the new tool |

---

## 4. Phases (each ends with passing checks; commit per phase)

**Phase 0: Baseline.** Branch `balanced-truncation`. Build (`make -j$(nproc)`; needs Eigen at
`/usr/include/eigen3` or `EIGEN_HOME`). Run `examples/JCP2024/Fig2/Rabi.param` and
`examples/NatPhys2022/02_phonons/03_calculate_phonon_PT.param` + `05_...` to confirm the
toolchain. Quick smoke test (verified on commit `49b151e`, runs in about a minute): in
`benchmarks/balanced/`, run `bin/ACE smoke_PT.param` (10 phonon modes, writes `smoke.pt`)
then `bin/ACE smoke_reuse.param`; the last line of both outputs must show `|1><1|` = 0.516043
at t = 5. `tools/PTB_analyze -read_PT smoke.pt` prints the bond dimensions
(expected `1 4 8 ... 3 1`, max 17): first element `d1 = 1`, last `d2 = 1`. Add a minimal C++ test harness (a `tests` Makefile target building
`tests/*.cpp` against `lib/libACE.so`; plain asserts, no new dependencies).
*Done when:* build + example runs succeed; `make tests` runs an empty test.

**Phase 1: Adjoint propagation.** `ProcessTensorElementAccessor::propagate_adjoint(chi, M, ...)`
handling the dictionary (support `dim1_front == dim1_back == 1` and no `expand` in v1; throw
otherwise). *Test:* for random `state` (NL x D1) and `chi` (NL x D2) on a real PT element,
`sum(chi .* propagate(state)) == sum(propagate_adjoint(chi) .* state)` to 1e-12 relative.

**Phase 2: Contexts and factors.** `BalancedContexts`: a list of contexts, each with a
`FreePropagator`, initial state, observable and weight, read from param files
(`add_balanced_context <file.param>` repeated; each file uses ACE's usual
`add_Hamiltonian`/`add_Pulse`/`initial`/`add_Output` keys; first `add_Output` is the observable).
`BalancedFactors`: forward trajectories (past) and backward testers (future) on a
`ProcessTensorBuffer`, symmetric Trotter as in `Simulation_PT::propagate_state`.
*Test:* `y = sum(chi_n .* state_n)` is identical for every cut `n` and equals the
`Simulation_PT` output for that context (reference test T2).

**Phase 3: Core projector.** `BalancedTruncation::protected_projector(A, F, q, rhat, trunc)`
returning `X, Y, sigma, diagnostics`; also an unprotected variant. *Tests (ported T3):*
`YX = I`, `q^T Pi = q^T`, `Pi rhat = rhat`, error = tail; compare `sigma` and errors against
`balanced_ref.py` on matrices exported from the reference (write them to small text files).

**Phase 4: Balanced sweep + tool.** `ProcessTensorBuffer::sweep_forward_balanced(contexts,
trunc, verbosity)` implementing Section 2's schedule (respect block buffering: use
`get(n, ForwardPreload)` / `BackwardPreload`, don't keep references across block loads;
throw if `TTree` is set). Tool `src_exec/tools/PTB_balanced_truncate.cpp`:
`read_PT`, `write_PT`, `add_balanced_context ...`, `balanced_threshold`, `balanced_maxk`,
`balanced_rhat` (`tw`|`reference`, default `tw`), `balanced_eta` (see Phase 5), `verbosity`.
Closure protection is mandatory (no `balanced_protect none`; see Section 7). Log the per-cut
diagnostics listed in Section 7. *Tests:* T4 and T8 on a small PT;
running a context through the truncated PT with `add_PT` reproduces the in-memory result.

**Phase 5: Blend + HS baseline in the same code path.** Flat HS Gramians (dictionary
multiplicities!), gauge-invariant blend (`balanced_eta`), and `balanced_mode hs` which must
reproduce optimal HS truncation (compare to an ACE SVD sweep at equal bond dimension: should be
equal or better per cut). *Test:* gauge invariance (T6) with a random gauge applied to a bond.
The blend is the production default (`balanced_eta 0.3`); until this phase lands, Phase 4 runs
with eta = 0 (pure operational).

**Phase 6: Benchmarks** (scripts in `benchmarks/balanced/`, outputs as CSV):
(a) QD phonons: `examples/NatPhys2022/02_phonons/03_calculate_phonon_PT.param` with a family
of Gaussian pulses (vary area 1π–5π and detuning) as contexts; train on a subset, evaluate
held-out pulses; compare SVD-truncated vs balanced-truncated PT at equal max bond dimension.
Metrics: RMS error of outputs vs an untruncated (or much tighter-threshold) reference, trace
error, runtime, `||X|| ||Y||` statistics.
(b) Pure dephasing (no pulse / `sigma_z` coupling only), where an exact reference exists.
Report both operational and HS errors (the manuscript discusses the trade-off).

**Later (not v1):** hook `sweep_forward_balanced` into
`sweep_intermediate_or_final_start_forward` when `is_final` (param `balanced_final_sweep`),
multiple PTs, alternating-least-squares refinement of the system-coupled objective.

---

## 5. Gotchas

- **Dictionary:** outer indices are deduplicated. Never index `M` with raw `(i, j)`; go through
  the accessor. For flat (HS) sums, weight by multiplicity or work on `expand_from_dict()` copies.
- **`keep` rescaling** in `sweep_forward`: SVD sweeps rescale `P`, `Pinv` by the largest singular
  value. Do not rescale `X`, `Y` independently; if any scaling is applied, use `X*c`, `Y/c`.
- **Normal forms:** call `clearNF()` on modified elements; balanced output is not in normal form.
- **Last element** has `dim_d2 == 1`; cuts run over `n = 0 .. n_tot-2`.
- **Closure semantics:** ACE closures contract the future with "trace output, input = identity/N"
  (`closure_indices`). That is the paper's `q`. Recompute with `calculate_closures()` on the
  untruncated PT only for checks; during the sweep update `closure_n <- X^T closure_n`.
- **Conjugation:** bilinear pairings; Gramians/norms use dagger. If a test is off by complex
  conjugation, the bug is almost certainly a missing/extra `.conjugate()` in Gramian assembly.
- **Memory:** future factors for all cuts: `n_tot * (n_f*NL) * D` complex numbers; for long PTs
  write them through the existing block-buffer mechanism or recompute in chunks.
- **Conditioning:** `Sigma_r^{-1/2}` with tiny `sigma` blows up `||X|| ||Y||`; respect `threshold`
  and log the amplification.

---

## 6. Definition of done (v1)

`make` and `make tests` pass; T1–T8 analogues pass in C++; `PTB_balanced_truncate` produces a PT
that ACE reads with `add_PT`; benchmark (a) produces a CSV comparing SVD vs balanced at equal bond
dimension on held-out pulses; no change to outputs of existing examples with default parameters.

---

## 7. Research decisions: ask the user, don't decide silently

- Objective: absorbed objective (closed form, current plan) vs. refining the system-coupled
  objective with alternating least squares.
- `rhat`: trace-weighted mean bond state (current default) vs. other choices or `q`-only protection.
- Context ensembles for the benchmarks (which pulse family, train/test split).
- Whether/how to apply operational truncation during mode merges (intermediate sweeps).

### Decisions (2026-10-22)

- **Objective:** closed-form absorbed objective plus the gauge-invariant blend, `balanced_eta`
  as a parameter, default 0.3. This is the object the Proposition guarantees, stays
  well-conditioned, and costs one small SVD per cut. ALS refinement of the system-coupled
  objective is a v2 research item: only with regularization (Tikhonov on X, Y, or a cap on
  `||XY||_2`), judged only out of distribution, never the default.
- **rhat:** default trace-weighted `rhat_tw`; expose `balanced_rhat tw|reference` for
  experiments. No optimizer over rhat.
- **Protection:** closure protection is mandatory (T4 and T8 depend on it). The unprotected
  projector stays in the library only for baselines, bounds and tests.
- **Per-cut diagnostics** (in addition to Section 2, item 8):
  `E_aug(Pi)` on the training pairs (always on the operational factors, also when blending);
  `||Pi||_2`; the unprotected bound `sum_{i>m} sigma_i(F A)^2`; `|s - 1|`; `||Pi_0||_2` with
  `Pi_0 = rhat q^T / s`. The bracket's lower bound `sum_{i>d^2 m} sigma_i(F_aug A_aug)^2` is
  not logged by default: it vanishes unless the number of contexts exceeds `d^2 m`.

- **Reference rhat** (`balanced_rhat reference`): one designated past, never "the first
  context" (order of `add_balanced_context` must not matter). `balanced_rhat_context <file>`
  defines it (context keys; observable ignored); default is the no-intervention past
  (identity system propagation, `rho0 = 1/d`, `rhat_ni`). rhat is the trace-weighted bond
  vector of that single trajectory (s = 1 for an exact closure). It is propagated through the
  already-truncated past like the training pasts (`state <- state Y^T` after each cut) and
  only defines rhat; it enters the Gramians only if also listed as a training context.
- **maxk/mink:** refer to the total bond dimension m (`mink <= m <= maxk`, `r <= maxk-1`), as
  `get_truncated_dim` does for SVD sweeps. `maxk = 1` -> r = 0, Pi = Pi_0. Numerically zero
  core values are never kept (r = 0 allowed). Skip the insertion only if m >= D (a threshold
  may still truncate when D <= maxk). `balanced_threshold` acts on operational core values
  and is not comparable to the SVD threshold: compare at equal bond dimension via maxk.
  Implementation note: the numerical-rank floor is `rtol * max(sigma_1, ||F||_2 ||A||_2)`;
  a floor relative to sigma_1 alone fails when the whole core is rounding noise
  (e.g. a past entirely along rhat), giving ||X|| ||Y|| ~ 1e8.
- **Weights for rhat_tw:** weighted mean with normalized past weights,
  `rhat = sum_a w_a r_a`, `sum_a w_a = 1` (protects the mixture the Gramians are built
  from); past weights only.
- **Phase 5 implementation notes (2026-10-22):** no separate unprotected `balanced_mode hs`
  (protection is mandatory); `balanced_eta 1` is the protected pure-HS case, and the
  reproduction of optimal HS/SVD truncation by unprotected HS balancing is a library test.
  HS Gramians use dictionary multiplicities (ACE's SVD sweeps use multiplicity 1 per stored
  index; both agree on expanded PTs). Gramians are never formed: square-root factors
  `Wp = L L^+`, `Wf = R^+ R` are propagated with QR compression, and the blend uses the exact
  factors `A_b = [sqrt((1-eta)/c_po) A, sqrt(eta/c_ph) L]`, `F_b = [...; ...]`. An
  eigendecomposition-based version lost gauge invariance to ~eps cond(G)^2-3 and inflated
  `||X|| ||Y||` to 1e5-1e7 on the test PT; the square-root version is gauge invariant to
  ~1e-13. When blending, `local_error`/`tail`/`amplification` refer to the blended
  objective, `E_aug` and the unprotected bound to the operational one.
- **Readout times (2026-10-22):** future contexts are read out at subsampled times,
  `balanced_readout_stride s` (default 5; 0 = final time only), cuts
  `{n_tot-1, n_tot-1-s, ...}`, weight `v_b/|R|` per readout. Motivation: with final-time
  testers only, eta=0 collapsed a pure-dephasing PT to m=2 (final readout exact, intermediate
  coherence wrong by ~3e-2). F_n is QR-compressed to <= D rows (Pi depends on F^+F only);
  E_aug is evaluated on the final-time training pairs (block structure kept separately).
- **Benchmarks:** train on a diverse ensemble (mixed initial states/observables); PT inputs
  for 30 modes (local) and 100 modes (HPC), same driver.

### Decisions (2026-10-23): post-v1 scope

- **Mode merges:** balanced truncation only as the *final* sweep of a PT calculation
  (`balanced_final_sweep true` in `bin/ACE`); intermediate sweeps (mode merges) stay SVD.
  The balanced sweep is the last truncating operation (no SVD truncation after it).
  Intermediate sweeps may use a looser `threshold`/`compress_maxk` than the final balanced
  target (`balanced_maxk`/`balanced_threshold`); the SVD target parameters
  (`final_sweep_threshold`, `final_sweep_compress_maxk`) are rejected together with
  `balanced_final_sweep` so the target is never set twice. Benchmark: both arms (SVD final
  sweep to maxk m vs balanced final sweep to m) with identical intermediate settings.
- **ALS:** not now (stays v2).
- **Multiple PTs:** before adding code, check whether `PTB_join` merges two PTs into one
  PT-MPO that the single-PT balanced path can truncate (then: join first, truncate after).
- **Phase 6 completion:** report the HS error ||T - T_trunc||_HS / ||T||_HS (dictionary
  multiplicities, computed on the difference MPO by QR, no cancellation) next to the
  operational errors; pure-dephasing check against an exact reference.

### Implementation notes and results (2026-10-23)

- **Phase 6 completion:** `BalancedGramians::hs_distance` / `tools/PTB_HS_distance` (HS error of a
  truncated PT, difference MPO + QR, no cancellation); `exact_dephasing.py` (independent boson
  model on ACE's discretized modes, validated to 3.4e-7 on a tight 6-mode PT). On the 100-mode
  PT, eta=0.3 reaches the input PT's own model error (vs. exact, M=3) from maxk 8; SVD does not
  within maxk 24. SVD is HS-optimal; eta=0.3 stays within ~3x of its HS error, eta=0 does not
  (HS error up to ~8 for pure dephasing: never use eta=0 alone).
- **Phase 7 (balanced final sweep):** `balanced_final_sweep true` in bin/ACE
  (`BalancedFinalSweep`, called in `ProcessTensorForwardList::setup2` right before the PT is
  written; parameters and context time grids are checked before the calculation). Without the
  flag the calculation is bitwise unchanged. End-to-end benchmark (30 modes, threshold 1e-6,
  final_sweep_n 1, identical intermediate settings, `final_sweep_benchmark.py`): balanced to
  m=8 already reaches the untruncated calculation's own total error (vs. the threshold-3e-7 PT
  and vs. exact), SVD needs m>24 for held-out contexts; the balanced sweep adds 2-3 s to a
  ~30 min calculation. Exception: pure-dephasing OOD (pulsed) contexts at m=12, where SVD is
  slightly better (2.4e-4 vs 4.3e-4).
- **Multiple PTs:** no new code needed. `PTB_join -initial_PT X -add_PT Y -only_join true`
  reproduces `add_PT X add_PT Y` to 1e-14, including the order of non-commuting baths, and the
  result is an ordinary PT-MPO that `PTB_balanced_truncate` / the balanced sweep truncate
  (`tests/test_multi_pt_join.cpp`). Exact joins multiply bond dimensions (e.g. 25 x 35 = 875);
  the balanced sweep handles that directly (diagnostics use BDCSVD/QR, the core keeps JacobiSVD).
  Test: with eta=0, final-time readouts and m = 3 NL + 1 the training outputs are exact on a
  joined PT (D=770) to 2e-15. Accuracy on a joined PT (small.pt x 2-mode JC bath, D=875,
  only 3 training contexts, eta=0.3, held-out max error): maxk 8 balanced 1.5e-2 vs SVD
  1.3e-2, maxk 16 balanced 7.3e-3 vs SVD 1.2e-2 (few contexts: modest gain).
  Observed (not investigated further): PTB_join's default compressed mode (select, threshold
  1e-9) deviated by 1.8e-3 from the two-PT simulation for non-commuting baths, so the exact
  join (`only_join`) is the safe input for balanced truncation.

### Decisions (2026-10-25): coverage study and joined-PT benchmark

- **Coverage (pure-dephasing OOD loss).** Finding: with a pulse-free (dephasing) training ensemble,
  SVD beats eta=0.3 on pulsed held-out contexts for m <= 8 and up to ~1.6x at m = 12, in all 3 seeds
  (bench30_thr1e-6), while balanced is 100-1000x better in distribution; balanced wins from m = 16.
  Interpretation: coverage, not a defect. Follow-up study:
  (a) pulsed test contexts disjoint in parameters from pulsed training contexts
      (train areas {pi, 3pi}, test areas {2pi, 4pi, 5pi}, at different detunings);
  (b) 2/16 and 4/16 pulsed training contexts (rest dephasing), 3 seeds each;
  (c) eta in {0.3, 0.6, 1.0} for the augmented runs;
  (d) singular-value spectra at a few cuts (dephasing family, pulsed family, HS Schmidt spectrum),
      plotted as decay curves (`balanced_spectra` output).
  Guidance recorded regardless: the training ensemble must cover the deployment contexts;
  eta = 1 (protected HS) is the fallback when it cannot.
- **Joined-PT benchmark.** First a one-seed validation: two 15-mode phonon halves of bench30
  (`Boson_skip_mode`), joined with `PTB_join -only_join`, must reproduce bench30 outputs and the
  balanced-vs-SVD comparison within truncation accuracy. Main benchmark: bench30 phonons joined with
  a few-mode lossy cavity bath (single modes with `add_Lindblad` in `.mparam` files), mixed
  ensemble 16/32/32, 3 seeds, contexts including exciton-driving pulses. Radiative continuum
  deferred.

### Results (2026-10-26): coverage study

- Pulse-free training ensemble augmented by K pulsed contexts (areas {pi, 3pi}), OOD test pulses
  disjoint in parameters (areas {2pi, 4pi, 5pi}, detunings {-1, 0, 1} meV), 3 seeds, bench30_thr1e-6
  (`results/coverage_dephasing_p{0,2,4}`, `plot_coverage.py` -> `plots/coverage/`). OOD RMS ratio
  SVD / balanced(eta=0.3), geometric mean over seeds, m = 4, 6, 8, 12, 16, 24:
  K=0: 0.54 0.78 0.81 0.50 1.01 1.62; K=2: 0.67 1.24 1.44 1.05 2.20 3.76;
  K=4: 0.75 2.14 2.12 1.46 3.19 7.46. In distribution, eta=0.3 stays better than SVD for all K
  (3-4x at m = 4, 15-240x at m = 6-8, 4000-10000x at m = 24). Conclusion: the OOD loss
  is a coverage effect; 2-4 pulsed training contexts with different parameters remove it from m = 6
  (m = 4 stays SVD-favoured). eta=1 tracks SVD, as expected.
