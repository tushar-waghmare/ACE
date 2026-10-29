#!/bin/bash --login
# Joined-PT benchmark, step 1 of 2 (ARCHER2): build bench30 phonons + lossy cavity PTs, join them
# exactly (D up to 45*36 = 1620, 68 GB on /work), validate against the two-PT simulation, and
# write the context files for seeds 1-3. Finished steps are skipped on resubmission.
#
#   cd $ACE_BAL_HOME/benchmarks/balanced/hpc
#   sbatch joined_cavity_setup.slurm
#   sbatch --dependency=afterok:<jobid> joined_cavity_probe.slurm   (peak memory -> MAX_PARALLEL)
#   sbatch --array=1-3%1 joined_cavity_seed.slurm
#
#SBATCH --job-name=ACE-bal-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-setup-%j.out
#SBATCH --error=ACE-bal-setup-%j.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
B="$ACE_BAL_HOME/benchmarks/balanced"
ACE="$ACE_BAL_HOME/bin/ACE"
echo "ACE-bal setup: $(date) on $(hostname), commit $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD)"

cd "$B"
[[ -f bench30_thr1e-6.pt ]] || srun --hint=nomultithread "$ACE" bench30_thr1e-6_PT.param > bench30_thr1e-6_build.log 2>&1
"$ACE_BAL_HOME/tools/PTB_analyze" -read_PT bench30_thr1e-6.pt | grep -E '^Maxdim'
cd "$B/joined"
[[ -f cavity_b2.pt ]] || srun --hint=nomultithread "$ACE" cavity_b2_PT.param > cavity_b2_build.log 2>&1
"$ACE_BAL_HOME/tools/PTB_analyze" -read_PT cavity_b2.pt | grep -E '^Maxdim'

FIRST_PT="$B/joined/cavity_b2.pt" SECOND_PT="$B/bench30_thr1e-6.pt" \
JOINED_PT="$B/joined/cavity_bench30.pt" CTX_OUT="$B/results/joined_cavity_mixed" \
  srun --hint=nomultithread bash "$B/hpc/prepare_joined.sh"
echo "setup finished: $(date)"
