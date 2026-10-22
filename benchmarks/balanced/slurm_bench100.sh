#!/usr/bin/env bash
# SLURM template for the 100-mode benchmark. Adapt partition/modules to your cluster.
#SBATCH --job-name=ace_balanced_bench100
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
##SBATCH --partition=<partition>
# module load <compiler> <mkl> <python3>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
# make -j"${SLURM_CPUS_PER_TASK}" MKLROOT="${MKLROOT:-}" EIGEN_HOME=<path>   # once
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
benchmarks/balanced/run_all.sh 100 "${SLURM_CPUS_PER_TASK:-16}"
