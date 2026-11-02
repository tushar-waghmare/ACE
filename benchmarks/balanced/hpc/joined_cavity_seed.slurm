#!/bin/bash --login
# Joined-PT benchmark, step 2 of 2 (ARCHER2): one node per seed (array index = seed), SVD vs
# balanced (eta 0, 0.1, 0.3, 0.5, 1) at maxk 4 6 8 12 16 24 on the exact join
# cavity x bench30 (D <= 1620, 68 GB on /work). At most MAX_PARALLEL maxk processes run at once.
# Disk: every running maxk process temporarily writes one full 68 GB copy of the PT on /work (the
# SVD arm copies and sweeps it in place; PTB_balanced_truncate copies it before truncating), which
# shrinks once that sweep is done. Peak on /work: the input PT + MAX_PARALLEL x 68 GB + a few GB.
# run_seed_parallel.sh checks the free user quota first (lustrequota) and stops if it is too small.
# Memory: one balanced truncation per process (see joined_cavity_probe.slurm; node: 256 GB).
#
#   Option A (default, 200 GB user quota): one maxk at a time, 12 simulations x 10 threads
#     sbatch --array=1-3%1 joined_cavity_seed.slurm
#   Option B (user quota >= ~300 GB): two maxk at a time, 2 x 6 simulations x 10 threads
#     sbatch --export=ALL,MAX_PARALLEL=2 --array=1-3%1 joined_cavity_seed.slurm
#
# Seeds run one after the other (%1). Restartable: resubmit the same array after a timeout (or
# switch from A to B in between); finished maxk are skipped, references and SVD PTs are reused.
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
# MAX_PARALLEL maxk processes x JOBS parallel simulations x 10 threads = 120 of 128 cores
MAX_PARALLEL="${MAX_PARALLEL:-1}"
export PT OUT="$B/results/joined_cavity_mixed" MAX_PARALLEL JOBS="${JOBS:-$((12 / MAX_PARALLEL))}" THREADS=10 BLOCKSIZE=2
export REF_JOBS=120
export OMP_PROC_BIND=false OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
echo "ACE-bal joined seed $SEED: $(date) on $(hostname), commit $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD), MAX_PARALLEL=$MAX_PARALLEL JOBS=$JOBS"
srun --hint=nomultithread --cpu-bind=none bash "$B/hpc/run_seed_parallel.sh"
echo "finished: $(date)"
