#!/bin/bash --login
# Study A grid on Cirrus: spectra/ranks (T4) and closing the loop (T5) on the finished PT builds
# (docs/studyA_spec.md Sec. 3-4; deviations, including the theta choice, in studyA/CHANGES.md).
# The tools are single-threaded (no OpenMP, no MKL), so parallelism = independent tool runs.
#
#   STAGE=spectra, --array=0: alpha 0.05, all 7 baths, theta 1e-6..1e-9 (whatever is built), then ranks
#   STAGE=spectra, --array=1: alpha 0.2,  same
#   STAGE=loop:               alpha 0.05, theta 1e-6 (primary), all built baths, families
#                             bernoulli 0.002/0.005/0.01 and coarse 20; one run per (bath, family, maxk),
#                             maxk = chi_op(3e-3)+{2,4,8} and 16. Needs the alpha 0.05 spectra.
#
#   cd $STUDYA_BASE/studyA
#   J=$(sbatch --parsable --export=ALL,STAGE=spectra --array=0-1 $STUDYA_HOME/studyA/hpc/studyA_grid_cirrus.slurm)
#   sbatch --export=ALL,STAGE=loop --dependency=afterok:${J}_0 $STUDYA_HOME/studyA/hpc/studyA_grid_cirrus.slurm
#
# Finished outputs are skipped on resubmission (spectra CSVs, heldout_*_m<maxk>.csv), so a job killed at
# the time limit can simply be resubmitted. Outputs: $STUDYA_RESULTS (default $STUDYA_BASE/studyA/results):
# ranks.csv, spectra/, heldout.csv (merged), runtimes_*.csv, and per-run logs.
# PTs that finish later (rebuilds on the long QOS) are picked up by resubmitting the same stage.
#SBATCH --job-name=ACE-studyA-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=2-00:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-studyA-grid-%x-%j.out
#SBATCH --error=ACE-studyA-grid-%x-%j.err
set -euo pipefail
source "${STUDYA_HOME:-${STUDYA_BASE:-/epccfs/e89/e89/$USER}/software/ACE-studyA}/studyA/hpc/cirrus_env.sh"
export STUDYA_PTS="$STUDYA_OUT"
export STUDYA_RESULTS="${STUDYA_RESULTS:-$STUDYA_BASE/studyA/results}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$STUDYA_RESULTS"
JOBS="${STUDYA_JOBS:-${SLURM_CPUS_PER_TASK:-256}}"
BATHS="oob6 oob4 oob2 inband0.8 edge3 edge5 ohmic"     # largest first
DRIVER="$STUDYA_HOME/studyA/run_pilot.py"
for b in tools/PTB_memory_spectra tools/PTB_family_truncate; do
  [[ -x "$STUDYA_HOME/$b" ]] || { echo "ERROR: missing $STUDYA_HOME/$b (build first)" >&2; exit 1; }
done
[[ -f "$STUDYA_PTS/builds.csv" ]] || { echo "ERROR: no builds.csv in $STUDYA_PTS" >&2; exit 1; }
echo "stage ${STAGE:?export STAGE=spectra|loop} commit $(git -C "$STUDYA_HOME" rev-parse --short HEAD) jobs $JOBS $(date)"

case "$STAGE" in
  spectra)
    case "${SLURM_ARRAY_TASK_ID:?submit with --array=0-1}" in
      0) ALPHA=0.05 ;;
      1) ALPHA=0.2 ;;
      *) echo "unknown array index" >&2; exit 1 ;;
    esac
    $PYTHON "$DRIVER" spectra --alpha $ALPHA --baths $BATHS --thetas 1e-6 1e-7 1e-8 1e-9 --available --jobs "$JOBS"
    # ranks.csv is shared by both array tasks: serialize the read-modify-write
    flock "$STUDYA_RESULTS/.ranks.lock" \
      $PYTHON "$DRIVER" ranks --alpha $ALPHA --baths $BATHS --thetas 1e-6 1e-7 1e-8 1e-9
    ;;
  loop)
    $PYTHON "$DRIVER" loop --alpha 0.05 --baths $BATHS --theta 1e-6 --available --split_maxk --jobs "$JOBS" \
      --families bernoulli:0.002 bernoulli:0.005 coarse:20:0.3 bernoulli:0.01
    ;;
  *) echo "unknown STAGE $STAGE" >&2; exit 1 ;;
esac
echo "finished: $(date)"
