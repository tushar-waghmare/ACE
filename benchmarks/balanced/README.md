# Balanced truncation benchmarks

Run from the repository root after `make`:

```
# 1. reference PT (10 QD-phonon modes, te=5; ~1 min)
cd benchmarks/balanced && ../../bin/ACE smoke_PT.param && cd ../..
# 2. SVD vs balanced at equal bond dimension
python3 benchmarks/balanced/run_benchmark.py --pt benchmarks/balanced/smoke.pt --te 5 --outdir out_pulses
python3 benchmarks/balanced/run_benchmark.py --pt benchmarks/balanced/smoke.pt --te 5 --family dephasing --outdir out_dephasing
```

`run_benchmark.py --help` documents the context families, splits (train / test from the same
family / ood from the other family) and the CSV columns. Errors are measured against the input
PT; `max_population_drift` is exactly zero for pure dephasing (only meaningful for that family).

Full benchmark (30 modes locally; 100 modes on HPC, see `slurm_bench100.sh`):

```
benchmarks/balanced/run_all.sh 30          # builds bench30.pt if missing, then all families
benchmarks/balanced/run_all.sh 100 32      # 100-mode PT (plan's NatPhys setup), 32 parallel jobs
```

Families: `mixed` (training diversity: single pulses, half random pure initial states,
observables |1><1|, sigma_x, sigma_y), `pulses` (narrow: ground state, |1><1|), `dephasing`
(no pulse, coherence); OOD sets are structurally different (two-pulse sequences / single pulses).
Settings can be overridden through environment variables (`SEEDS`, `MAXK`, `ETA`, `N_TRAIN`,
`N_TEST`, `N_OOD`, `STRIDE`, `FAMILIES`).

`results/pilot_*.csv` are **pilot** runs on `smoke.pt` (8 train, 16 test, 16 ood contexts, seed 1,
maxk 4/6/8/12, eta 0/0.3/1, rhat tw, final-time readouts only).

## Figures and runtime profile

```
python3 benchmarks/balanced/profile_timing.py --outdir benchmarks/balanced/results/timing   # ~15 min, sequential
~/.venvs/ace/bin/python benchmarks/balanced/plot_accuracy.py   # needs matplotlib
~/.venvs/ace/bin/python benchmarks/balanced/plot_timing.py
```

`profile_timing.py` uses `PTB_balanced_truncate -balanced_timing <file.csv>` (wall-clock seconds per
phase: contexts, read_PT, setup, future, hs_future, past, factors, noise_floor, core_svd, proj_diag,
diagnostics, apply, write_PT, total). Plots go to `plots/<topic>/` (PNG + PDF): `plots/accuracy/`, `plots/timing/`; new plot scripts add their own subfolder.

## HS errors and exact pure-dephasing reference

```
python3 benchmarks/balanced/hs_errors.py --jobs 8              # <dir>/hs_errors.csv (regenerates truncated PTs)
~/.venvs/ace/bin/python benchmarks/balanced/exact_dephasing.py # <dir>/exact_errors.csv for bench*_dephasing
```

`tools/PTB_HS_distance -read_PT A -compare_PT B` prints ||A - B||_HS / ||A||_HS (dictionary multiplicities,
difference MPO orthogonalized by QR: no cancellation). New `run_benchmark.py` runs record it as `hs_rel`.
`exact_dephasing.py`: independent boson model on the same discretized modes as the PT (from
`Boson_print_E_g`), Fock cutoff M=3 (the Hamiltonian the PT represents) and M=20; validated against a
6-mode PT built with threshold 1e-10 (max deviation 3.4e-7 for M=3).

## Balanced final sweep in bin/ACE (Phase 7)

```
# in a PT calculation's parameter file:
balanced_final_sweep  true
balanced_maxk         16            # target; 'threshold'/'compress_maxk' apply to the calculation itself
balanced_eta          0.3
add_balanced_context  ctx1.param    # repeat; same time grid as the calculation

~/.venvs/ace/bin/python benchmarks/balanced/final_sweep_benchmark.py --jobs 9   # 13 builds (~1 h on 10 cores)
~/.venvs/ace/bin/python benchmarks/balanced/plot_final_sweep.py                 # plots/final_sweep/
```

Multiple baths: join first (`tools/PTB_join -initial_PT X.pt -add_PT Y.pt -only_join true`, same
order as `add_PT X add_PT Y`), then truncate the joined PT with `PTB_balanced_truncate`.
