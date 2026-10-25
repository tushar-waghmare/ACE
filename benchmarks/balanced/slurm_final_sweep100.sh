#!/usr/bin/env bash
# SLURM template: end-to-end balanced final sweep vs SVD final sweep on the 100-mode PT
# (final_sweep_benchmark.py with bench100_PT.param). 13 independent PT builds (base, svd_m,
# balanced_mixed_m, balanced_dephasing_m for m in 8 12 16 24), one core each, then evaluation.
# Adapt partition/modules to your cluster. Builds already finished are skipped on resubmission.
#SBATCH --job-name=ace_final_sweep100
#SBATCH --cpus-per-task=13
#SBATCH --mem=64G
#SBATCH --time=72:00:00
##SBATCH --partition=<partition>
# module load <compiler> <mkl> <python3>   # python needs numpy (exact dephasing reference)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
# make -j"${SLURM_CPUS_PER_TASK}" MKLROOT="${MKLROOT:-}" EIGEN_HOME=<path>   # once
export OMP_NUM_THREADS=1
PYTHON="${PYTHON:-python3}"
B=benchmarks/balanced
for FAM in mixed dephasing; do   # same seeded contexts as run_all.sh (results/bench100_<family>)
  [ -d "$B/results/bench100_$FAM/seed_1/ctx" ] ||
    "$PYTHON" $B/run_benchmark.py --pt $B/bench100.pt --te 20 --dt 0.1 --family $FAM \
      --n_train 16 --n_test 32 --n_ood 32 --seeds 1 2 3 --contexts_only --outdir $B/results/bench100_$FAM
done
"$PYTHON" $B/final_sweep_benchmark.py --template $B/bench100_PT.param \
  --ctxdir "$B/results/bench100_{family}/seed_{seed}/ctx" --tight '' \
  --outdir $B/results/final_sweep100 --jobs "${SLURM_CPUS_PER_TASK:-13}"
