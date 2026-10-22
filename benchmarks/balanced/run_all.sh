#!/usr/bin/env bash
# Full benchmark (Phase 6): SVD vs protected balanced truncation at equal bond dimension.
#
#   benchmarks/balanced/run_all.sh [MODES] [JOBS]
#
# MODES: 30 (default; builds in minutes on a laptop) or 100 (plan's NatPhys PT; hours on
#        one core -> HPC, ideally with MKL: make MKLROOT=...). Any N works if
#        bench${N}_PT.param exists.
# JOBS:  parallel ACE simulations (default: number of cores).
# Environment overrides: SEEDS, MAXK, ETA, N_TRAIN, N_TEST, N_OOD, STRIDE, FAMILIES.
# Results: benchmarks/balanced/results/bench${MODES}_<family>/results.csv
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MODES="${1:-30}"
JOBS="${2:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1) )}"
SEEDS="${SEEDS:-1 2 3}"
MAXK="${MAXK:-4 6 8 12 16 24}"
ETA="${ETA:-0 0.1 0.3 0.5 1}"
N_TRAIN="${N_TRAIN:-16}"; N_TEST="${N_TEST:-32}"; N_OOD="${N_OOD:-32}"
STRIDE="${STRIDE:-5}"
FAMILIES="${FAMILIES:-mixed pulses dephasing}"
PYTHON="${PYTHON:-python3}"

for b in bin/ACE tools/PTB_balanced_truncate tools/PTB_sweep_forward tools/PTB_sweep_backward tools/PTB_analyze; do
  [ -x "$ROOT/$b" ] || { echo "missing $ROOT/$b: run 'make' first" >&2; exit 1; }
done

PT="$HERE/bench${MODES}.pt"
if [ ! -f "$PT" ]; then
  [ -f "$HERE/bench${MODES}_PT.param" ] || { echo "missing $HERE/bench${MODES}_PT.param" >&2; exit 1; }
  echo "building $PT ..."
  (cd "$HERE" && /usr/bin/time -p "$ROOT/bin/ACE" "bench${MODES}_PT.param" > "bench${MODES}_build.log" 2>&1)
fi
TE=$(awk '$1=="te"{print $2}' "$HERE/bench${MODES}_PT.param")
DT=$(awk '$1=="dt"{print $2}' "$HERE/bench${MODES}_PT.param")

for FAM in $FAMILIES; do
  echo "=== family $FAM (PT: $PT, te=$TE)"
  "$PYTHON" "$HERE/run_benchmark.py" --pt "$PT" --te "$TE" --dt "$DT" --family "$FAM" \
    --seeds $SEEDS --maxk $MAXK --eta $ETA --n_train "$N_TRAIN" --n_test "$N_TEST" --n_ood "$N_OOD" \
    --readout_stride "$STRIDE" --jobs "$JOBS" --outdir "$HERE/results/bench${MODES}_${FAM}"
done
