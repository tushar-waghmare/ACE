# Study A environment on Cirrus (EPCC, AMD EPYC 9745, HPE Cray PE 25.03). Sourced, not executed.
# Project space on Cirrus is /epccfs (NFS); /work is Lustre for shared software only (no e89 there).
# STUDYA_BASE: your project directory; the clone is $STUDYA_BASE/software/ACE-studyA (override with
# STUDYA_HOME), PT outputs go to $STUDYA_BASE/studyA/pts (override with STUDYA_OUT).
module load PrgEnv-gnu 2>/dev/null || true
module load gcc-native/14.2
module load eigen/3.4.0
module load intel-mkl/2025.0
module load cray-python 2>/dev/null || module load python/3.11.7

export STUDYA_BASE="${STUDYA_BASE:-/epccfs/e89/e89/$USER}"
[[ -d "$STUDYA_BASE" ]] || echo "WARNING: STUDYA_BASE=$STUDYA_BASE does not exist; export STUDYA_BASE=<your /epccfs project dir>" >&2
export STUDYA_HOME="${STUDYA_HOME:-$STUDYA_BASE/software/ACE-studyA}"
export STUDYA_OUT="${STUDYA_OUT:-$STUDYA_BASE/studyA/pts}"
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
