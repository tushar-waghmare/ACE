#!/usr/bin/env bash
# Submit the joined-PT grid (run on a login node, after joined_grid_validate.slurm printed REPRODUCED):
#   prep (one task per seed) -> SVD arm (one task per maxk, one at a time: 68 GB working copy)
#                            -> balanced arm (one task per (seed, maxk), at most PAR_BAL at once)
# Done parts are skipped, so the same command resumes an interrupted grid. Environment overrides:
# SEEDS, MAXK, ETA, ETA_JOBS (parallel etas per balanced task), PAR_BAL (default 6), OUT, CACHE.
set -euo pipefail
cd "$(dirname "$0")"
source ./archer2_env.sh > /dev/null 2>&1 || true
source ./grid_env.sh
nS=${#SEEDS[@]}; nK=${#MAXKS[@]}
seeds_csv=$(IFS=,; echo "${SEEDS[*]}")
prep=$(sbatch --parsable --export=ALL --array="$seeds_csv" joined_grid_prep.slurm)
svd=$(sbatch --parsable --export=ALL --array="0-$((nK - 1))%1" --dependency="afterok:$prep" joined_grid_svd.slurm)
bal=$(sbatch --parsable --export=ALL --array="0-$((nS * nK - 1))%${PAR_BAL:-6}" --dependency="afterok:$prep" joined_grid_balanced.slurm)
echo "prep $prep (seeds ${SEEDS[*]}), svd $svd (maxk ${MAXKS[*]}), balanced $bal ($((nS * nK)) tasks)"
echo "progress: python3 $(cd .. && pwd)/hpc/merge_results.py $OUT"
