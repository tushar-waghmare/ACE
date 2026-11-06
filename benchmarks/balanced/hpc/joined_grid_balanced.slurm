#!/bin/bash --login
# Joined-PT grid, balanced arm: one task per (seed, maxk); --array index t -> seed SEEDS[t / nK],
# maxk MAXK[t % nK] (default 3 seeds x 6 maxk: --array=0-17). Streaming truncation (no PT copy on
# disk), cached futures; the ETA_JOBS etas run in parallel (memory: ETA_JOBS x one
# PTB_balanced_truncate, see joined_cavity_probe.slurm). Done (seed, maxk) are skipped.
#   sbatch --array=0-17%6 --dependency=afterok:<prep job> joined_grid_balanced.slurm  (or: submit_grid.sh)
#SBATCH --job-name=ACE-bal-bal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-bal-%A_%a.out
#SBATCH --error=ACE-bal-bal-%A_%a.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
source "$ACE_BAL_HOME/benchmarks/balanced/hpc/grid_env.sh"
T="${SLURM_ARRAY_TASK_ID:?submit with --array=0-17}"
nK=${#MAXKS[@]}
S="${SEEDS[$((T / nK))]}"; K="${MAXKS[$((T % nK))]}"
header "balanced seed $S maxk $K"
d="$OUT/seed_$S/maxk_$K"
if [[ -f "$d/results.csv" || -f "$d/results_balanced.csv" ]]; then echo "seed $S maxk $K balanced: done, skipping"; exit 0; fi
mkdir -p "$d"
rm -f "$d"/balanced_*.pt "$d"/balanced_*.pt_*   # outputs of an interrupted run
ETA_JOBS="${ETA_JOBS:-1}"
set_threads "${THREADS:-$((120 / ETA_JOBS))}"
echo "seed $S maxk $K balanced: start $(date) (eta_jobs $ETA_JOBS, threads $OMP_NUM_THREADS)"
srun --hint=nomultithread --cpu-bind=none "$PYTHON" -u "$B/run_benchmark.py" $(common_args "$S") --maxk "$K" \
  --eta $ETA --stage balanced --stream --eta_jobs "$ETA_JOBS" --jobs "${JOBS:-4}" --outdir "$d" > "$d/run_balanced.log" 2>&1
echo "seed $S maxk $K balanced: finished $(date)"
"$PYTHON" "$B/hpc/merge_results.py" "$OUT"
