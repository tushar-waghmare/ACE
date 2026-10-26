#!/usr/bin/env bash
# Exact join of two PTs and its validation (portable; called by joined_cavity_setup.slurm).
#   FIRST_PT   initial PT (blocked; it is COPIED first: PTB_join writes into initial_PT in place)
#   SECOND_PT  PT joined into it; the result reproduces '-add_PT FIRST_PT -add_PT SECOND_PT'
#   JOINED_PT  output PT (header + block files)
#   CTX_OUT    run_benchmark.py outdir for the context files (seed_<s>/ctx), mixed family
#   SEEDS      default "1 2 3";   VAL_TOL  max allowed |joined - two-PT| (default 1e-8)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
B="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$B/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
SEEDS="${SEEDS:-1 2 3}"
VAL_TOL="${VAL_TOL:-1e-8}"
: "${FIRST_PT:?}" "${SECOND_PT:?}" "${JOINED_PT:?}" "${CTX_OUT:?}"

copy_pt() {  # header + block files <name>_<n>
  cp "$1" "$2"
  local f
  for f in "$1"_*; do
    [[ -e "$f" && "${f##*_}" =~ ^[0-9]+$ ]] && cp "$f" "$2_${f##*_}"
  done
  return 0
}

if [[ -f "$JOINED_PT" && -f "$JOINED_PT.done" ]]; then
  echo "join exists: $JOINED_PT"
else
  rm -f "$JOINED_PT" "$JOINED_PT"_*
  copy_pt "$FIRST_PT" "$JOINED_PT"
  echo "joining $SECOND_PT into a copy of $FIRST_PT -> $JOINED_PT  ($(date))"
  "$ROOT/tools/PTB_join" -initial_PT "$JOINED_PT" -add_PT "$SECOND_PT" -only_join true -verbosity 0 \
    > "$JOINED_PT.join.log" 2>&1
  touch "$JOINED_PT.done"
fi
"$ROOT/tools/PTB_analyze" -read_PT "$JOINED_PT" | grep -E '^Maxdim'
du -ch "$JOINED_PT"* | tail -1

# contexts (deterministic per seed; run_benchmark.py regenerates the same files in each run)
"$PYTHON" "$B/run_benchmark.py" --pt "$JOINED_PT" --te 20 --dt 0.1 --family mixed \
  --n_train 16 --n_test 32 --n_ood 32 --seeds $SEEDS --contexts_only --outdir "$CTX_OUT"

# validation: joined PT vs simulating with both PTs (exact join => agreement to rounding)
first_seed="${SEEDS%% *}"
VAL="$CTX_OUT/validation"
mkdir -p "$VAL"
for c in train_00 test_00 ood_00 ood_01; do
  ctx="$CTX_OUT/seed_$first_seed/ctx/$c.param"
  "$ROOT/bin/ACE" "$ctx" -add_PT "$JOINED_PT" -outfile "$VAL/$c.joined.out" -set_precision 14 > /dev/null
  "$ROOT/bin/ACE" "$ctx" -add_PT "$FIRST_PT" -add_PT "$SECOND_PT" -outfile "$VAL/$c.twoPT.out" \
    -set_precision 14 > /dev/null
done
"$PYTHON" - "$VAL" "$VAL_TOL" <<'PY'
import glob, sys
val, tol = sys.argv[1], float(sys.argv[2])
worst = 0.0
for fj in sorted(glob.glob(f'{val}/*.joined.out')):
    rows = lambda p: [list(map(float, l.split())) for l in open(p) if l.strip() and not l.startswith('#')]
    a, b = rows(fj), rows(fj.replace('.joined.', '.twoPT.'))
    assert len(a) == len(b) and len(a) > 0, fj
    d = max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra[1:], rb[1:]))
    worst = max(worst, d)
    print(f'  {fj.split("/")[-1]:24s} max |joined - two-PT| = {d:.2e}')
print(f'validation: max deviation {worst:.2e} (tolerance {tol:.0e})')
sys.exit(0 if worst <= tol else 1)
PY
