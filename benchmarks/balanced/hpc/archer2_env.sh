# Shared ARCHER2 environment for the balanced-truncation HPC jobs (sourced, not executed).
# Same toolchain as the ACE-PD jobs: PrgEnv-gnu, gcc 11.2, Eigen 3.4, MKL 2023.
# Override ACE_BAL_HOME if the balanced-truncation clone lives elsewhere.
module load PrgEnv-gnu
module load gcc/11.2.0
module load eigen/3.4.0
module load mkl/2023.0.0
module load cray-python            # drivers need only the Python standard library (>= 3.8)

export ACE_BAL_HOME="${ACE_BAL_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-balanced}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python3}"

# Eigen include dir: EIGEN_HOME if set, else the usual module variables
if [[ -z "${EIGEN_HOME:-}" ]]; then
  for d in "${EIGEN_ROOT:-}" "${EIGEN_DIR:-}" "${EIGEN3_ROOT:-}"; do
    for sub in include/eigen3 include ""; do
      if [[ -n "$d" && -f "$d/$sub/Eigen/Eigen" ]]; then export EIGEN_HOME="$d/$sub"; break 2; fi
    done
  done
fi
