# Study A environment on Cirrus (EPCC, AMD EPYC 9745, HPE Cray PE 25.03). Sourced, not executed.
# Override STUDYA_HOME if the clone lives elsewhere.
module load PrgEnv-gnu 2>/dev/null || true
module load gcc-native/14.2
module load eigen/3.4.0
module load intel-mkl/2025.0
module load cray-python 2>/dev/null || module load python/3.11.7

export STUDYA_HOME="${STUDYA_HOME:-/work/e89/e89/tusharwaghmare/software/ACE-studyA}"
export ACE_BAL_HOME="$STUDYA_HOME"          # name used by the shared ARCHER2 helpers
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python3}"

# Eigen include dir: EIGEN_HOME if set, else module variables / CPATH entries
if [[ -z "${EIGEN_HOME:-}" ]]; then
  for d in "${EIGEN_ROOT:-}" "${EIGEN_DIR:-}" "${EIGEN3_ROOT:-}" "${EIGEN_HOME_DIR:-}" $(echo "${CPATH:-}" | tr ':' ' '); do
    for sub in include/eigen3 include ""; do
      if [[ -n "$d" && -f "$d/$sub/Eigen/Eigen" ]]; then export EIGEN_HOME="$d/$sub"; break 2; fi
    done
  done
fi
# MKL: the module should set MKLROOT; otherwise build without MKL (Eigen's own linear algebra)
if [[ -z "${MKLROOT:-}" ]]; then
  for d in "${MKL_ROOT:-}" "${MKL_HOME:-}"; do [[ -n "$d" && -d "$d/include" ]] && export MKLROOT="$d"; done
fi
echo "studyA env: $(command -v g++) ($(g++ -dumpfullversion 2>/dev/null)), EIGEN_HOME=${EIGEN_HOME:-<unset>}, MKLROOT=${MKLROOT:-<unset>}"
