#!/bin/bash --login
# Build the Study A clone on a Cirrus compute node (a dropped login session cannot kill it).
# First, on the login node:  git clone -b studyA https://github.com/tushar-waghmare/ACE.git $STUDYA_HOME
# Then:                       sbatch $STUDYA_HOME/studyA/hpc/build_cirrus.slurm
# Afterwards: make tests (from $STUDYA_HOME, see README) and the smoke test.
#SBATCH --job-name=ACE-studyA-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-studyA-build-%j.out
#SBATCH --error=ACE-studyA-build-%j.err
set -euo pipefail
source "${STUDYA_HOME:-${STUDYA_BASE:-/epccfs/e89/e89/$USER}/software/ACE-studyA}/studyA/hpc/cirrus_env.sh"
[[ -n "${EIGEN_HOME:-}" && -f "$EIGEN_HOME/Eigen/Eigen" ]] || { echo "ERROR: Eigen headers not found; set EIGEN_HOME" >&2; exit 1; }
cd "$STUDYA_HOME"
echo "commit $(git rev-parse --short HEAD), $(date)"
rm -f lib/*.o include/PCH.hpp.gch   # objects from another machine/compiler must not be reused
make -j 64 EIGEN_HOME="$EIGEN_HOME" MKLROOT="${MKLROOT:-}"
for b in bin/ACE tools/PTB_truncate_time tools/PTB_memory_spectra tools/PTB_family_truncate; do
  [[ -x "$b" ]] || { echo "ERROR: missing $b" >&2; exit 1; }
done
echo "build OK $(date)"
