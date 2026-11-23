#!/bin/bash --login
# Study A PT builds on Cirrus (AMD EPYC 9745, 256 cores/node; ARCHER2 version: studyA_pt_build.slurm) (docs/studyA_spec.md; deviations in studyA/CHANGES.md).
# One node per theta group, all builds of the group run in parallel on that node:
#   --array=0: theta 1e-6 and 1e-7    (28 builds,  8 threads each)
#   --array=1: theta 1e-8             (14 builds, 16 threads each)
#   --array=2: theta 1e-9, alpha 0.05 ( 7 builds, 32 threads each; split for memory)
#   --array=3: theta 1e-9, alpha 0.2  ( 7 builds, 32 threads each)
# Each build: bin/ACE (DnC, 256 steps, exact K_n from studyA/hpc/K) -> PTB_truncate_time (200 steps)
# -> $OUT/<bath>_a<alpha>_N200_thr<theta>.pt and a .done marker; finished builds are skipped on
# resubmission. Runtime and max bond dimension go to $OUT/builds.csv. A build still running at the
# time limit is killed (no .done): that itself is the result ("not feasible within 24 h").
#   sbatch --array=0-1 studyA_pt_build_cirrus.slurm                                   (48 h, standard)
#   sbatch --array=2-3 --qos=long --time=4-00:00:00 studyA_pt_build_cirrus.slurm      (theta 1e-9)
# Targeted rebuilds (no --array): STUDYA_ONLY="bath:alpha:theta ..." runs exactly those builds, all in
# parallel, e.g. the builds killed at 48 h in job 703919 that the grid needs (studyA/CHANGES.md, 2026-11-23):
#   sbatch --export=ALL,STUDYA_ONLY="oob6:0.05:1e-7 oob4:0.2:1e-6 oob6:0.2:1e-6" \
#          --qos=long --time=4-00:00:00 studyA_pt_build_cirrus.slurm
# Note: ACE is built without OpenMP and (on Cirrus) without MKL, so every build runs on ONE core;
# THREADS below only sets how many builds share the node (NPAR = 256 / THREADS), not their speed.
#SBATCH --job-name=ACE-studyA-pt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=2-00:00:00
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --output=ACE-studyA-pt-%A_%a.out
#SBATCH --error=ACE-studyA-pt-%A_%a.err
set -euo pipefail
source "${STUDYA_HOME:-${STUDYA_BASE:-/epccfs/e89/e89/$USER}/software/ACE-studyA}/studyA/hpc/cirrus_env.sh"
OUT="$STUDYA_OUT"
mkdir -p "$OUT"
cd "$OUT"
GROUP="${SLURM_ARRAY_TASK_ID:-only}"
[[ "$GROUP" == only && -z "${STUDYA_ONLY:-}" ]] && { echo "submit with --array=0-3 or STUDYA_ONLY" >&2; exit 1; }
case "$GROUP" in
  only) THETAS=""; ALPHAS=""; THREADS=8 ;;
  0) THETAS="1e-6 1e-7"; ALPHAS="0.2 0.05"; THREADS=8 ;;
  1) THETAS="1e-8"; ALPHAS="0.2 0.05"; THREADS=16 ;;
  2) THETAS="1e-9"; ALPHAS="0.05"; THREADS=32 ;;
  3) THETAS="1e-9"; ALPHAS="0.2"; THREADS=32 ;;
  *) echo "unknown group $GROUP" >&2; exit 1 ;;
esac
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS
echo "group $GROUP thetas $THETAS only ${STUDYA_ONLY:-} threads $THREADS commit $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD) $(date)"
[[ -x "$ACE_BAL_HOME/bin/ACE" && -x "$ACE_BAL_HOME/tools/PTB_truncate_time" ]] || {
  echo "ERROR: build missing; sbatch $ACE_BAL_HOME/studyA/hpc/build_cirrus.slurm" >&2; exit 1; }

build() {  # $1 bath, $2 alpha, $3 theta
  local name="dnc256_$1_a$2_thr$3" pt="$1_a$2_N200_thr$3.pt"
  [[ -f "$pt.done" ]] && { echo "skip $pt (done)"; return 0; }
  sed -e "s|@THETA@|$3|; s|@NAME@|$name|g; s|@KFILE@|$ACE_BAL_HOME/studyA/hpc/K/K_$1_a$2_dt0.1_N512.dat|" \
      "$ACE_BAL_HOME/studyA/hpc/pt_template.param" > "$name.param"
  rm -f "$name.pt" "$name.pt_"*          # partial output of a build killed at the time limit
  local t0=$(date +%s)
  if "$ACE_BAL_HOME/bin/ACE" "$name.param" > "$name.log" 2>&1 \
     && "$ACE_BAL_HOME/tools/PTB_truncate_time" -read_PT "$name.pt" -write_PT "$pt" -n_tot 200 >> "$name.log" 2>&1; then
    local t1=$(date +%s)
    local maxd=$(grep -o 'Maxdim at n=[0-9]*: [0-9,]* -> [0-9]*' "$name.log" | awk '{print $NF}' | sort -n | tail -1)
    echo "$1,$2,$3,$((t1-t0)),${maxd:-NA},$(date -Iseconds)" >> builds.csv
    rm -f "$name.pt" "$name.pt_"*          # the 256-step PT is not needed any more
    touch "$pt.done"
    echo "done $pt: $((t1-t0)) s, max D ${maxd:-NA}"
  else
    echo "FAILED $pt (see $OUT/$name.log)"
  fi
}
export -f build
export ACE_BAL_HOME OUT
[[ -f builds.csv ]] || echo "bath,alpha,theta,seconds,maxdim,finished" > builds.csv
# largest baths first, so they get the most wall time
TASKS=""
for theta in $THETAS; do
  for alpha in $ALPHAS; do
    for bath in oob6 oob4 oob2 inband0.8 edge3 edge5 ohmic; do TASKS+="$bath $alpha $theta"$'\n'; done
  done
done
if [[ -n "${STUDYA_ONLY:-}" ]]; then
  TASKS=$(for t in $STUDYA_ONLY; do echo "${t//:/ }"; done)$'\n'
fi
NPAR=$(( ${SLURM_CPUS_PER_TASK:-256} / THREADS ))
printf "%s" "$TASKS" | xargs -P "$NPAR" -L 1 bash -c 'build "$0" "$1" "$2"'
echo "finished: $(date)"
