# Shared settings of the (seed, maxk) grid for the joined cavity x bench30 benchmark (sourced by the
# joined_grid_*.slurm scripts after archer2_env.sh). The context/run arguments are those of
# run_seed_parallel.sh, so the grid continues (and must reproduce) the earlier per-seed runs.
B="$ACE_BAL_HOME/benchmarks/balanced"
PT="${PT:-$B/joined/cavity_bench30.pt}"
OUT="${OUT:-$B/results/joined_cavity_mixed}"
CACHE="${CACHE:-$B/results/joined_cavity_cache}"   # per input PT: shared by OUT and validation runs
SEEDS=(${SEEDS:-1 2 3})
MAXKS=(${MAXK:-4 6 8 12 16 24})
ETA="${ETA:-0 0.1 0.3 0.5 1}"
BLOCKSIZE="${BLOCKSIZE:-2}"
PYTHON="${PYTHON:-python3}"
export OMP_PROC_BIND=false OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE

common_args() {  # $1 = seed, $2 = output root (default OUT)
  local root="${2:-$OUT}"
  echo --pt "$PT" --te 20 --dt 0.1 --family mixed --n_train 16 --n_test 32 --n_ood 32 --seed "$1" \
       --readout_stride 5 --ref_dir "$root/seed_$1/ref" --cache_dir "$CACHE" --batched --buffer_blocksize "$BLOCKSIZE"
}

set_threads() { export OMP_NUM_THREADS="$1" MKL_NUM_THREADS="$1" OPENBLAS_NUM_THREADS="$1"; }

quota_gb() {  # '200G', '81.31G', '78.03T', '400k' -> GB
  awk -v v="$1" 'BEGIN{n=v+0; u=substr(v, length(v)); f=1;
    if(u=="k"||u=="K")f=1/1048576; else if(u=="M")f=1/1024; else if(u=="T")f=1024; else if(u=="P")f=1048576;
    printf "%.1f", n*f}'
}
require_quota_gb() {  # $1 = GB that must be free in the /work user quota; skipped without lustrequota
  command -v lustrequota > /dev/null 2>&1 || { echo "disk check skipped (no lustrequota)"; return 0; }
  local line quota use free
  line="$(lustrequota 2> /dev/null | grep -E '^[[:space:]]*User \(' | head -1)" || true
  [[ -n "$line" ]] || { echo "disk check skipped (cannot parse lustrequota)"; return 0; }
  quota="$(quota_gb "$(echo "$line" | awk '{print $3}')")"
  use="$(quota_gb "$(echo "$line" | awk '{print $4}')")"
  free="$(awk -v q="$quota" -v u="$use" 'BEGIN{printf "%.1f", q-u}')"
  echo "disk check: user quota ${quota} GB, used ${use} GB, free ${free} GB; need $1 GB"
  if awk -v f="$free" -v n="$1" 'BEGIN{exit !(f < n)}'; then
    echo "ERROR: not enough quota on /work (free $free GB < $1 GB). Remove leftover PT copies (svd_*.pt*" \
         "without .done, balanced_*.pt*) below $OUT or request a larger quota." >&2
    return 1
  fi
}

pt_gb() { { du -ck "$PT" "$PT"_* 2> /dev/null || true; } | tail -1 | awk '{printf "%.1f", $1/1048576}'; }

header() {  # $1 = job kind
  echo "ACE-bal $1: $(date) on $(hostname), checkout $(git -C "$ACE_BAL_HOME" rev-parse --short HEAD)," \
       "tool build $("$ACE_BAL_HOME/tools/PTB_balanced_truncate" -build_info true 2>/dev/null || echo '<unstamped: rebuild>')"
}
