# ACE fork: operational balanced truncation of PT-MPOs

ACE (github.com/mcygorek/ACE, MIT) is a C++14/Eigen code for numerically exact open-quantum-system
simulations with process-tensor MPOs (PT-MPOs). In this fork we add *operational balanced
truncation*: bond truncation that minimizes the error for a specified ensemble of system contexts,
with the PT closure protected exactly.

Current task and full plan: @docs/balanced_truncation_plan.md

## Build and run

- Build: `make -j$(nproc)` (Eigen at `/usr/include/eigen3` or set `EIGEN_HOME`; MKL optional via
  `MKLROOT`). Use `OPTIM=1` for faster debug builds. A full build takes about 20 min on one core at `OPTIM=1`.
- Outputs: library `lib/libACE.so`, main binary `bin/ACE`, tools from `src_exec/tools/*.cpp` in `tools/`.
- Run: `bin/ACE file.param`. Examples in `examples/` (start with `examples/JCP2024/Fig2/Rabi.param`).
- Python reference (must always pass): `python tools/balanced_ref/balanced_ref.py`

## Code map (most relevant)

- `ProcessTensorElement` (`M`, `closure`, `env_ops`, normal forms, SVD `sweep_forward/backward`)
- `ProcessTensorElementAccessor::propagate` (augmented state, dictionary-compressed outer index)
- `ProcessTensorBuffer` (file-backed chain; sweeps; `add_modes`; final sweeps)
- `TruncatedSVD`, `TruncationLayout` (rank rules and parameter names)
- `Simulation_PT::propagate_state` (symmetric Trotter: half step, PT element, half step)
- `src_exec/tools/PTB_*.cpp` (standalone PT tools; templates for new tools)

## Index conventions (get these right first)

- Element `M(i, d1, d2)`: `d1` bond to previous element, `d2` to next, `i` outer index via
  `IF_OD_Dictionary` (never index raw (out, in) pairs directly).
- Augmented state: `MatrixXcd state` of shape `NL x D`; `state2.row(i) += state.row(j) * M_slice`.
- Paper to ACE: `p_mu = state.row(mu)^T`, `Q^beta = M_beta^T`, closure = `q`.
  Inserting `Pi = X Y` on bond n: `M_n.inner_multiply_right(Y^T)`,
  `M_{n+1}.inner_multiply_left(X^T)`, `closure_n = X^T closure_n`,
  `env_ops.process_forward(PassOn{P = X^T, Pinv = Y^T})`, then `clearNF()`.
- Pairings are bilinear (no conjugation); only Gramians and norms use the adjoint.

## Working rules

- Work on branch `balanced-truncation`; one commit per plan phase, each with passing tests.
- Do not change default behaviour or outputs of existing algorithms; new features sit behind new
  parameters and new functions/tools.
- Follow existing style: namespace `ACE`, errors via `std::cerr` + `throw DummyException()`,
  parameters via `Parameters::get_as_*`. No new third-party dependencies.
- Every numerical routine gets a test that compares against `tools/balanced_ref/balanced_ref.py`
  or checks an exact identity (adjoint pairing, `YX = I`, closure protection, trace preservation).
- Section 7 of the plan lists research decisions: ask the user instead of choosing silently.
