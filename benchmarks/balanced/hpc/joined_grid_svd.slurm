#!/bin/bash --login
# Joined-PT grid, SVD arm (one task per maxk, --array = indices into MAXK "4 6 8 12 16 24", i.e. 0-5).
# The SVD-truncated PT does not depend on the seed: computed once in $OUT/svd (kept, svd_<k>.pt.done),
# then every seed's contexts are simulated on it and scored -> seed_<s>/maxk_<k>/results_svd.csv.
# Computing it needs one full 68 GB working copy of the input PT (backward sweep in place), so run
# these tasks one at a time (%1) under the 200 GB quota. Seeds/maxk already done are skipped.
#   sbatch --array=0-5%1 --dependency=afterok:<prep job> joined_grid_svd.slurm   (or: bash submit_grid.sh)
#SBATCH --job-name=ACE-bal-svd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-svd-%A_%a.out
#SBATCH --error=ACE-bal-svd-%A_%a.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
source "$ACE_BAL_HOME/benchmarks/balanced/hpc/grid_env.sh"
K="${MAXKS[${SLURM_ARRAY_TASK_ID:?submit with --array=0-5%1}]}"
header "svd maxk $K"
if [[ ! -f "$OUT/svd/svd_$K.pt.done" ]]; then
  rm -f "$OUT/svd/svd_$K.pt" "$OUT/svd/svd_$K.pt"_*   # partial copy of an interrupted run
  require_quota_gb "$(awk -v p="$(pt_gb)" 'BEGIN{printf "%.0f", p+10}')"
fi
set_threads "${THREADS:-16}"
for S in "${SEEDS[@]}"; do
  d="$OUT/seed_$S/maxk_$K"
  if [[ -f "$d/results.csv" || -f "$d/results_svd.csv" ]]; then echo "seed $S maxk $K svd: done, skipping"; continue; fi
  mkdir -p "$d"
  echo "seed $S maxk $K svd: start $(date)"
  srun --hint=nomultithread --cpu-bind=none "$PYTHON" -u "$B/run_benchmark.py" $(common_args "$S") --maxk "$K" \
    --stage svd --svd_dir "$OUT/svd" --jobs "${JOBS:-8}" --outdir "$d" > "$d/run_svd.log" 2>&1
  echo "seed $S maxk $K svd: finished $(date)"
done
"$PYTHON" "$B/hpc/merge_results.py" "$OUT"
