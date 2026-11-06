#!/bin/bash --login
# Validation of the grid pipeline before using it (one job): seed 1, maxk 4 with the new code path
# (batched reference and simulations, cached futures and HS norm, streaming balanced truncation,
# shared SVD PT) into $OUT_V, compared row by row with the earlier run's
# $OUT/seed_1/maxk_4/results.csv (compare_results.py, rtol 1e-9) and the reference outputs byte by
# byte. The caches it writes ($CACHE) are reused by the grid. Prints REPRODUCED / NOT REPRODUCED.
#   sbatch joined_grid_validate.slurm
#SBATCH --job-name=ACE-bal-valid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-valid-%j.out
#SBATCH --error=ACE-bal-valid-%j.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
source "$ACE_BAL_HOME/benchmarks/balanced/hpc/grid_env.sh"
OUT_V="${OUT_V:-$B/results/joined_cavity_validate}"
S=1; K=4
header "validation seed $S maxk $K"
OLD="$OUT/seed_$S/maxk_$K/results.csv"
[[ -f "$OLD" ]] || { echo "ERROR: $OLD missing" >&2; exit 1; }
[[ -f "$OUT/svd/svd_$K.pt.done" ]] || { echo "ERROR: $OUT/svd/svd_$K.pt.done missing (SVD arm would need a 68 GB copy)" >&2; exit 1; }
require_quota_gb 30
d="$OUT_V/seed_$S/maxk_$K"
mkdir -p "$d" "$OUT_V/seed_$S/prep"
run_stage() {  # $1 = stage, rest = extra arguments
  local st="$1"; shift
  echo "$st: start $(date)"
  srun --hint=nomultithread --cpu-bind=none "$PYTHON" -u "$B/run_benchmark.py" $(common_args "$S" "$OUT_V") --maxk "$K" \
    --stage "$st" "$@" > "$OUT_V/run_$st.log" 2>&1
  echo "$st: finished $(date)"
}
set_threads "${THREADS:-16}"
run_stage prep --jobs "${REF_JOBS:-8}" --outdir "$OUT_V/seed_$S/prep"
ETA_JOBS="${ETA_JOBS:-1}"
set_threads "${THREADS_BAL:-$((120 / ETA_JOBS))}"
run_stage balanced --eta $ETA --stream --eta_jobs "$ETA_JOBS" --jobs 4 --outdir "$d"
set_threads "${THREADS:-16}"
run_stage svd --svd_dir "$OUT/svd" --jobs 8 --outdir "$d"

echo "== reference outputs (old: bin/ACE per context; new: PTB_simulate_contexts)"
n=0; nd=0
for f in "$OUT/seed_$S/ref"/ref_*.out; do
  n=$((n + 1)); cmp -s "$f" "$OUT_V/seed_$S/ref/$(basename "$f")" || { nd=$((nd + 1)); echo "differs: $(basename "$f")"; }
done
echo "reference files: $n compared, $nd differ"
echo "== results"
"$PYTHON" "$B/compare_results.py" "$OLD" "$d"
echo "== timings (seconds per phase)"; cat "$d/timings.csv" "$OUT_V/seed_$S/prep/timings.csv"
echo "finished: $(date)"
