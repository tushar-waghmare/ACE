#!/bin/bash --login
# Joined-PT grid, stage 1 (one task per seed, --array = seeds): context files, reference outputs
# (PTB_simulate_contexts: each process reads the 68 GB PT once for its chunk of contexts) and the
# maxk/eta-independent caches in $CACHE (computed once, reused by all later stages and seeds):
#   log ||T||_HS and maxdim of the input PT, HS future factors (shared by all seeds; locked),
#   operational future factors of this seed's training contexts.
# Finished parts are skipped on resubmission (existing reference outputs and matching caches).
#   sbatch --array=1-3 joined_grid_prep.slurm        (or: bash submit_grid.sh)
#SBATCH --job-name=ACE-bal-prep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-prep-%A_%a.out
#SBATCH --error=ACE-bal-prep-%A_%a.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
source "$ACE_BAL_HOME/benchmarks/balanced/hpc/grid_env.sh"
[[ -f "$PT.done" ]] || { echo "ERROR: $PT not ready; run joined_cavity_setup.slurm first" >&2; exit 1; }
SEED="${SLURM_ARRAY_TASK_ID:?submit with --array=<seeds>}"
header "prep seed $SEED"
require_quota_gb 30   # caches: 2 x <= 9 GB (HS shared, future per seed) + margin
# reference: REF_JOBS processes x THREADS; caches: 3 processes in parallel with THREADS each
set_threads "${THREADS:-16}"
mkdir -p "$OUT/seed_$SEED/prep"
srun --hint=nomultithread --cpu-bind=none "$PYTHON" -u "$B/run_benchmark.py" $(common_args "$SEED") --maxk 4 \
  --stage prep --jobs "${REF_JOBS:-8}" --outdir "$OUT/seed_$SEED/prep"
ls -l "$CACHE"
echo "finished: $(date)"
