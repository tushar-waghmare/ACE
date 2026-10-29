#!/bin/bash --login
# Joined-PT benchmark, step 2 of 2 (ARCHER2): one node per seed (array index = seed), SVD vs
# balanced (eta 0, 0.1, 0.3, 0.5, 1) at maxk 4 6 8 12 16 24 on the exact join
# cavity x bench30 (D <= 1620, 68 GB on /work). At most MAX_PARALLEL maxk processes run at once:
# each holds one balanced truncation in memory, and the first seed's run of a maxk also a 68 GB
# copy of the PT on /work (SVD arm, shared by all seeds in results/.../svd). Set MAX_PARALLEL from
# the peak memory measured by joined_cavity_probe.slurm (node: 256 GB). Run the seeds one after the
# other (%1). Restartable: resubmit the same array after a timeout, finished maxk are skipped.
#
#   sbatch --array=1-3%1 joined_cavity_seed.slurm
#   results: $ACE_BAL_HOME/benchmarks/balanced/results/joined_cavity_mixed/results.csv (merged)
#
#SBATCH --job-name=ACE-bal-joined
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-joined-%A_%a.out
#SBATCH --error=ACE-bal-joined-%A_%a.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
B="$ACE_BAL_HOME/benchmarks/balanced"
PT="$B/joined/cavity_bench30.pt"
[[ -f "$PT.done" ]] || { echo "ERROR: $PT not ready; run joined_cavity_setup.slurm first" >&2; exit 1; }
export SEED="${SLURM_ARRAY_TASK_ID:?submit with --array=1-3}"
# 2 maxk processes x 6 parallel simulations x 10 threads = 120 of 128 cores
export PT OUT="$B/results/joined_cavity_mixed" MAX_PARALLEL="${MAX_PARALLEL:-2}" JOBS=6 THREADS=10 BLOCKSIZE=2
export REF_JOBS=120
export OMP_PROC_BIND=false OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
echo "ACE-bal joined seed $SEED: $(date) on $(hostname), commit $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD)"
srun --hint=nomultithread --cpu-bind=none bash "$B/hpc/run_seed_parallel.sh"
echo "finished: $(date)"
