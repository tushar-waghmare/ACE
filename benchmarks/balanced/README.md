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

`results/pilot_*.csv` are **pilot** runs on `smoke.pt` (8 train, 16 test, 16 ood contexts, seed 1,
maxk 4/6/8/12, eta 0/0.3/1, rhat tw). The context ensemble for the final benchmarks is a
research decision (docs/balanced_truncation_plan.md, Section 7) and still open.
