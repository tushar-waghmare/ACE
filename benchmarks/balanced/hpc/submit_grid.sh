#!/usr/bin/env bash
# Submit the joined-PT grid (run on a login node, after joined_grid_validate.slurm printed REPRODUCED):
#   prep (one task per seed) -> SVD arm (one task per maxk, one at a time: 68 GB working copy)
#                            -> balanced arm (one task per (seed, maxk), at most PAR_BAL at once)
# Done parts are skipped, so the same command resumes an interrupted grid. Environment overrides:
# SEEDS, MAXK, ETA, ETA_JOBS (parallel etas per balanced task), PAR_BAL (default 6), OUT, CACHE.
#
# Every array task counts against the QOS submit limit (MaxSubmitPU). The script checks the limit
# first and submits nothing if the grid does not fit; then submit one seed at a time, e.g.
#   SEEDS=1 bash submit_grid.sh      (later: SEEDS=2, SEEDS=3; SVD PTs are reused across seeds)
# If an sbatch fails part-way, the jobs this call already submitted are cancelled again.
set -euo pipefail
cd "$(dirname "$0")"
source ./archer2_env.sh > /dev/null 2>&1 || true
source ./grid_env.sh
nS=${#SEEDS[@]}; nK=${#MAXKS[@]}
seeds_csv=$(IFS=,; echo "${SEEDS[*]}")
need=$((nS + nK + nS * nK))

qos=$(sed -n 's/^#SBATCH --qos=//p' joined_grid_balanced.slurm | head -1)
limit=$(sacctmgr -nP show qos "${qos:-standard}" format=MaxSubmitPU 2> /dev/null | head -1 || true)
have=$( (squeue -h -r -u "$USER" 2> /dev/null || true) | wc -l | tr -d ' ')
if [[ "$limit" =~ ^[0-9]+$ ]]; then
  echo "QOS ${qos:-standard}: submit limit $limit jobs per user, $have already queued/running, this grid needs $need"
  if (( have + need > limit )); then
    echo "ERROR: grid does not fit the submit limit; nothing submitted. Submit fewer seeds per call" \
         "(e.g. SEEDS=1 bash submit_grid.sh) or wait for queued jobs to finish." >&2
    exit 1
  fi
else
  echo "submit limit unknown (sacctmgr gave '${limit}'); $have jobs already queued/running, this grid needs $need"
fi

submitted=()
rollback() { if (( ${#submitted[@]} )); then echo "sbatch failed: cancelling ${submitted[*]}" >&2; scancel "${submitted[@]}"; fi; }
trap rollback ERR
prep=$(sbatch --parsable --export=ALL --array="$seeds_csv" joined_grid_prep.slurm); submitted+=("$prep")
svd=$(sbatch --parsable --export=ALL --array="0-$((nK - 1))%1" --dependency="afterok:$prep" joined_grid_svd.slurm); submitted+=("$svd")
bal=$(sbatch --parsable --export=ALL --array="0-$((nS * nK - 1))%${PAR_BAL:-6}" --dependency="afterok:$prep" joined_grid_balanced.slurm); submitted+=("$bal")
trap - ERR
echo "prep $prep (seeds ${SEEDS[*]}), svd $svd (maxk ${MAXKS[*]}), balanced $bal ($((nS * nK)) tasks)"
echo "progress: python3 $(cd .. && pwd)/hpc/merge_results.py $OUT"
