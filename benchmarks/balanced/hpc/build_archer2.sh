#!/bin/bash --login
# Clone (or update) the balanced-truncation branch on ARCHER2 and build it. Run on a LOGIN node:
#   bash build_archer2.sh            (first time: copy this file to the login node, or curl it)
# Separate from the ACE-PD build in $ACE_HOME, which stays untouched.
set -euo pipefail
source "$(dirname "$0")/archer2_env.sh" 2>/dev/null || {
  # first run: the clone does not exist yet; the env file is fetched with it below
  module load PrgEnv-gnu gcc/11.2.0 eigen/3.4.0 mkl/2023.0.0 cray-python
  export ACE_BAL_HOME="${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}"
}
REPO="${REPO:-https://github.com/tushar-waghmare/ACE.git}"
BRANCH="${BRANCH:-balanced-truncation}"

if [[ -d "$ACE_BAL_HOME/.git" ]]; then
  git -C "$ACE_BAL_HOME" fetch origin "$BRANCH"
  git -C "$ACE_BAL_HOME" checkout "$BRANCH"
  git -C "$ACE_BAL_HOME" merge --ff-only "origin/$BRANCH"
else
  git clone -b "$BRANCH" "$REPO" "$ACE_BAL_HOME"
fi
source "$ACE_BAL_HOME/benchmarks/balanced/hpc/archer2_env.sh"
if [[ -z "${EIGEN_HOME:-}" || ! -f "$EIGEN_HOME/Eigen/Eigen" ]]; then
  echo "ERROR: Eigen headers not found; set EIGEN_HOME (dir containing Eigen/Eigen)" >&2; exit 1
fi
echo "commit:     $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD)"
echo "EIGEN_HOME: $EIGEN_HOME"
echo "MKLROOT:    ${MKLROOT:-<unset: building without MKL>}"
cd "$ACE_BAL_HOME"
make -j 8 EIGEN_HOME="$EIGEN_HOME" MKLROOT="${MKLROOT:-}"   # 'all' = library, bin/ACE and tools
for b in bin/ACE tools/PTB_join tools/PTB_balanced_truncate tools/PTB_sweep_forward tools/PTB_sweep_backward \
         tools/PTB_analyze tools/PTB_HS_distance; do
  [[ -x "$b" ]] || { echo "ERROR: missing $b" >&2; exit 1; }
done
echo "build OK"
