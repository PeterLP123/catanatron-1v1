#!/usr/bin/env bash
# Reconstruct the hybrid-BC parent used by the DAgger F pilot.
#
# The original w0030 checkpoint is not in Git. This script regenerates the
# locked hard_state_v2 corpora and the hybrid objective (weight 0.003) so the
# existing DAgger command can run on a clean machine.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
PARENT_GAMES=${PARENT_GAMES:-2000}
PARENT_SEED=${PARENT_SEED:-101}
PARENT_SHARD_GAMES=${PARENT_SHARD_GAMES:-100}
BC_EPOCHS=${PARENT_BC_EPOCHS:-10}
BC_SEED=${PARENT_BC_SEED:-101}
PROMOTION_GAMES=${PARENT_PROMOTION_GAMES:-50}

DATA_F_F=${DATA_F_F:-$ROOT/data/hard_state_v2/F_F}
DATA_VP_F=${DATA_VP_F:-$ROOT/data/hard_state_v2/VP_F}
OUTPUT_ROOT=${1:-$ROOT/runs/bc-hybrid-sweep/w0030}
BC_CHECKPOINT=${BC_CHECKPOINT:-$OUTPUT_ROOT/bc.pt}
BC_META=${BC_META:-$OUTPUT_ROOT/bc.meta.json}
REPORT=${REPORT:-$OUTPUT_ROOT/promotion_benchmark.json}
STATUS_FILE="$OUTPUT_ROOT/parent_status.txt"

mkdir -p "$OUTPUT_ROOT/logs" "$ROOT/data/hard_state_v2"

write_status() {
  local state=$1
  local detail=${2:-none}
  {
    echo "state=$state"
    echo "detail=$detail"
    echo "updated=$(date --iso-8601=seconds)"
  } >"$STATUS_FILE"
}

dataset_complete() {
  local dir=$1
  local expected=$2
  [[ -f "$dir/dataset_meta.json" ]] || return 1
  "$PYTHON_BIN" - "$dir/dataset_meta.json" "$expected" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
expected = int(sys.argv[2])
ok = (
    meta.get("status") == "complete"
    and int(meta.get("completed_games", 0)) == expected
)
raise SystemExit(0 if ok else 1)
PY
}

generate_dataset() {
  local teachers=$1
  local output=$2
  local log=$3
  if dataset_complete "$output" "$PARENT_GAMES"; then
    echo "Reusing complete dataset: $output"
    return 0
  fi
  local extra=()
  if [[ -f "$output/dataset_meta.json" ]]; then
    extra+=(--resume)
  fi
  "$PYTHON_BIN" examples/colonist_1v1_generate_data.py \
    --num "$PARENT_GAMES" \
    --teachers "$teachers" \
    --score-candidates \
    --choices-only \
    --seed "$PARENT_SEED" \
    --shard-games "$PARENT_SHARD_GAMES" \
    --feature-profile raw \
    --output "$output" \
    "${extra[@]}" \
    2>&1 | tee "$log"
}

write_status collecting parent-data
generate_dataset F,F "$DATA_F_F" "$OUTPUT_ROOT/logs/generate_ff.log" &
pid_ff=$!
generate_dataset VP,F "$DATA_VP_F" "$OUTPUT_ROOT/logs/generate_vpf.log" &
pid_vpf=$!
ff_rc=0
vpf_rc=0
wait "$pid_ff" || ff_rc=$?
wait "$pid_vpf" || vpf_rc=$?
if [[ "$ff_rc" -ne 0 ]]; then
  write_status blocked generate-ff
  echo "F,F dataset generation failed" >&2
  exit 3
fi
if [[ "$vpf_rc" -ne 0 ]]; then
  write_status blocked generate-vpf
  echo "VP,F dataset generation failed" >&2
  exit 3
fi

write_status training hybrid-bc
if [[ -f "$BC_CHECKPOINT" && -f "$BC_META" ]]; then
  echo "Reusing completed BC checkpoint: $BC_CHECKPOINT"
else
  "$PYTHON_BIN" examples/colonist_1v1_bc.py \
    --data-dir "$DATA_F_F" "$DATA_VP_F" \
    --loss hybrid \
    --hybrid-listwise-weight 0.003 \
    --listwise-temperature 0.02 \
    --lr 0.001 \
    --epochs "$BC_EPOCHS" \
    --val-fraction 0.1 \
    --test-fraction 0.1 \
    --split-seed "$BC_SEED" \
    --seed "$BC_SEED" \
    --device auto \
    --feature-profile raw \
    --out "$BC_CHECKPOINT" \
    --run-dir "$OUTPUT_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/bc.log"
fi

write_status evaluating promotion
if [[ -f "$REPORT" ]]; then
  echo "Reusing promotion report: $REPORT"
else
  "$PYTHON_BIN" examples/colonist_1v1_evaluate.py \
    --agent "T:$BC_CHECKPOINT" \
    --benchmark \
    --protocol fast \
    --num-games "$PROMOTION_GAMES" \
    --eval-kind promotion \
    --gate-mode point \
    --checkpoint-label hybrid-bc-w0030 \
    --report "$REPORT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/evaluate.log"
fi

write_status complete "$REPORT"
echo "Hybrid-BC parent complete: $BC_CHECKPOINT"
echo "Promotion report: $REPORT"
