#!/bin/bash --login
# Joined-PT benchmark, resource probe (ARCHER2): one balanced truncation of the 68 GB joined PT
# (seed 1 training contexts, maxk 24, eta 0.3, same threads as joined_cavity_seed.slurm) under
# GNU time. Prints peak memory and wall time; choose MAX_PARALLEL for the seed jobs from it
# (MAX_PARALLEL * peak + ~10 GB for the simulations must fit into 256 GB).
#
#   sbatch joined_cavity_probe.slurm
#
#SBATCH --job-name=ACE-bal-probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-bal-probe-%j.out
#SBATCH --error=ACE-bal-probe-%j.err
set -euo pipefail
source "${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}/benchmarks/balanced/hpc/archer2_env.sh"
B="$ACE_BAL_HOME/benchmarks/balanced"
PT="$B/joined/cavity_bench30.pt"
OUT="$B/results/joined_cavity_mixed"
[[ -f "$PT.done" ]] || { echo "ERROR: $PT not ready; run joined_cavity_setup.slurm first" >&2; exit 1; }
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OMP_PROC_BIND=false OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
P="$OUT/probe"
mkdir -p "$P"
cmd=("$ACE_BAL_HOME/tools/PTB_balanced_truncate" -read_PT "$PT" -write_PT "$P/balanced_24_0.3.pt"
     -balanced_maxk 24 -balanced_eta 0.3 -balanced_rhat tw -balanced_readout_stride 5
     -balanced_log "$P/balanced_24_0.3.csv" -buffer_blocksize 2 -verbosity 1)
for c in "$OUT"/seed_1/ctx/train_*.param; do cmd+=(-add_balanced_context "$c"); done
echo "ACE-bal probe: $(date) on $(hostname), commit $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD)"
srun --hint=nomultithread --cpu-bind=none /usr/bin/time -v "${cmd[@]}" > "$P/truncate.log" 2> "$P/time.log"
grep -E 'Maximum resident set size|Elapsed \(wall clock\)|User time|Exit status' "$P/time.log"
"$ACE_BAL_HOME/tools/PTB_analyze" -read_PT "$P/balanced_24_0.3.pt" | grep -E '^Maxdim'
du -ch "$P"/balanced_24_0.3.pt* | tail -1
rm -f "$P"/balanced_24_0.3.pt "$P"/balanced_24_0.3.pt_*
echo "probe finished: $(date)"
