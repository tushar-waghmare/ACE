#!/usr/bin/env bash
# One seed of the SVD-vs-balanced benchmark on a large PT, one run_benchmark.py process per maxk,
# at most MAX_PARALLEL at a time (portable; called by joined_cavity_seed.slurm). Restartable: a maxk
# whose results.csv exists is skipped, so a job that hit the time limit can simply be resubmitted.
# The reference simulations run once per seed (OUT/seed_<s>/ref) and the SVD-truncated PTs once
# for all seeds (OUT/svd); both are reused by the maxk runs and on resubmission.
#   PT          input PT (blocked)            OUT     results root (seed_<s>/maxk_<k>/ below it)
#   SEED        seed                          MAXK    default "4 6 8 12 16 24"
#   ETA         default "0 0.1 0.3 0.5 1"     JOBS    parallel simulations per process (default 2)
#   MAX_PARALLEL  concurrent maxk processes (default: all); each holds one balanced truncation,
#               and the first run of a maxk also one copy of PT (SVD arm)
#   REF_JOBS    parallel reference simulations (default JOBS * MAX_PARALLEL)
#   BLOCKSIZE   blocks of the balanced output PTs (default 2)
#   THREADS     OMP/MKL threads per process (default 1)
#   DISK_MARGIN_GB  extra free space required besides MAX_PARALLEL copies of PT (default 10)
# Disk: each running maxk process temporarily writes one full copy of PT. If 'lustrequota'
# (ARCHER2) is available, the free user quota is checked before any maxk run starts.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
PYTHON="${PYTHON:-python3}"
: "${PT:?}" "${OUT:?}" "${SEED:?}"
MAXK="${MAXK:-4 6 8 12 16 24}"
ETA="${ETA:-0 0.1 0.3 0.5 1}"
JOBS="${JOBS:-2}"
BLOCKSIZE="${BLOCKSIZE:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-$(echo $MAXK | wc -w)}"
REF_JOBS="${REF_JOBS:-$((JOBS * MAX_PARALLEL))}"
export OMP_NUM_THREADS="${THREADS:-1}" MKL_NUM_THREADS="${THREADS:-1}" OPENBLAS_NUM_THREADS="${THREADS:-1}"

quota_gb() {  # '200G', '81.31G', '78.03T', '400k' -> GB
  awk -v v="$1" 'BEGIN{n=v+0; u=substr(v, length(v)); f=1;
    if(u=="k"||u=="K")f=1/1048576; else if(u=="M")f=1/1024; else if(u=="T")f=1024; else if(u=="P")f=1048576;
    printf "%.1f", n*f}'
}
check_disk() {  # free user quota on /work vs MAX_PARALLEL copies of PT; skipped without lustrequota
  command -v lustrequota > /dev/null 2>&1 || { echo "disk check skipped (no lustrequota)"; return 0; }
  local line quota use pt_gb need free
  line="$(lustrequota 2> /dev/null | grep -E '^[[:space:]]*User \(' | head -1)" || true
  if [[ -z "$line" ]]; then echo "disk check skipped (cannot parse lustrequota)"; return 0; fi
  quota="$(quota_gb "$(echo "$line" | awk '{print $3}')")"
  use="$(quota_gb "$(echo "$line" | awk '{print $4}')")"
  pt_gb="$({ du -ck "$PT" "$PT"_* 2> /dev/null || true; } | tail -1 | awk '{printf "%.1f", $1/1048576}')"  # no blocks: glob unmatched
  need="$(awk -v n="$MAX_PARALLEL" -v p="$pt_gb" -v m="${DISK_MARGIN_GB:-10}" 'BEGIN{printf "%.1f", n*p+m}')"
  free="$(awk -v q="$quota" -v u="$use" 'BEGIN{printf "%.1f", q-u}')"
  echo "disk check: user quota ${quota} GB, used ${use} GB, free ${free} GB; need ${need} GB" \
       "(MAX_PARALLEL=$MAX_PARALLEL x PT ${pt_gb} GB + margin)"
  if awk -v f="$free" -v n="$need" 'BEGIN{exit !(f < n)}'; then
    echo "ERROR: not enough quota on /work. Lower MAX_PARALLEL, remove leftover PT copies" \
         "(svd_*.pt*, balanced_*.pt* below $OUT), or request a larger quota." >&2
    exit 1
  fi
}
check_disk

common=(--pt "$PT" --te 20 --dt 0.1 --family mixed --n_train 16 --n_test 32 --n_ood 32 --seed "$SEED"
        --readout_stride 5 --ref_dir "$OUT/seed_$SEED/ref")

ref="$OUT/seed_$SEED/ref"
if [[ ! -f "$ref/done" ]]; then
  mkdir -p "$ref"
  echo "seed $SEED reference: start $(date) ($REF_JOBS parallel)"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    "$PYTHON" -u "$B/run_benchmark.py" "${common[@]}" --reference_only --jobs "$REF_JOBS" \
    --outdir "$ref/run" > "$ref/run.log" 2>&1
  touch "$ref/done"
  echo "seed $SEED reference: finished $(date)"
fi

running() {  # number of still-running maxk processes (bash 3.2 has no 'wait -n')
  local n=0 pk
  for pk in ${pids[@]+"${pids[@]}"}; do kill -0 "${pk%%:*}" 2> /dev/null && n=$((n + 1)); done
  echo $n
}
pids=()
for k in $MAXK; do
  d="$OUT/seed_$SEED/maxk_$k"
  if [[ -f "$d/results.csv" ]]; then echo "seed $SEED maxk $k: done, skipping"; continue; fi
  while (( $(running) >= MAX_PARALLEL )); do sleep 30; done
  mkdir -p "$d"
  echo "seed $SEED maxk $k: start $(date)"
  "$PYTHON" -u "$B/run_benchmark.py" "${common[@]}" --maxk "$k" --eta $ETA --svd_dir "$OUT/svd" \
    --jobs "$JOBS" --buffer_blocksize "$BLOCKSIZE" --outdir "$d" > "$d/run.log" 2>&1 &
  pids+=("$!:$k")
done
fail=0
for pk in ${pids[@]+"${pids[@]}"}; do
  if wait "${pk%%:*}"; then echo "seed $SEED maxk ${pk##*:}: finished $(date)"
  else echo "seed $SEED maxk ${pk##*:}: FAILED (see $OUT/seed_$SEED/maxk_${pk##*:}/run.log)" >&2; fail=1; fi
done
"$PYTHON" "$HERE/merge_results.py" "$OUT"
exit $fail
