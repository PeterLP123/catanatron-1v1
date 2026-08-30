#!/usr/bin/env bash
# Collect the next DAgger iteration from a stronger student, retrain hybrid BC
# on the frozen base corpora plus all verified distillation iterations, and
# evaluate on the same promotion protocol as the parent.
set -euo pipefail
export PYTHONUNBUFFERED=1

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <student-bc.pt> <distill-root> <base-f-f-dir> <base-vp-f-dir> <output-root>" >&2
  exit 2
fi

STUDENT_CHECKPOINT=$1
DISTILL_ROOT=$2
BASE_F_F=$3
BASE_VP_F=$4
OUTPUT_ROOT=$5
PYTHON_BIN=${PYTHON_BIN:-python}
DAGGER_GAMES=${DAGGER_GAMES:-100}
DAGGER_SEED=${DAGGER_SEED:-20260721}
DAGGER_SHARD_GAMES=${DAGGER_SHARD_GAMES:-10}
DAGGER_AUGMENTATION_WEIGHT=${DAGGER_AUGMENTATION_WEIGHT:-4}
BC_EPOCHS=${DAGGER_BC_EPOCHS:-10}
BC_SEED=${DAGGER_BC_SEED:-101}
PROMOTION_GAMES=${DAGGER_PROMOTION_GAMES:-50}

for required_file in "$STUDENT_CHECKPOINT"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $required_file" >&2
    exit 2
  fi
done
for required_dir in "$DISTILL_ROOT" "$BASE_F_F" "$BASE_VP_F"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Required directory not found: $required_dir" >&2
    exit 2
  fi
done

BC_RUN="$OUTPUT_ROOT/bc"
BC_CHECKPOINT="$BC_RUN/bc.pt"
BC_META="$BC_RUN/bc.meta.json"
REPORT="$OUTPUT_ROOT/promotion_benchmark.json"
STATUS_FILE="$OUTPUT_ROOT/pilot_status.txt"
mkdir -p "$OUTPUT_ROOT/logs"

write_status() {
  local state=$1
  local detail=${2:-none}
  {
    echo "state=$state"
    echo "detail=$detail"
    echo "updated=$(date --iso-8601=seconds)"
  } >"$STATUS_FILE"
}

NEXT_ITERATION=$("$PYTHON_BIN" - "$DISTILL_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
seen = {int(item["iteration"]) for item in manifest.get("iterations", [])}
print(0 if not seen else max(seen) + 1)
PY
)

write_status collecting "dagger-f-iteration-$NEXT_ITERATION"
if [[ -d "$DISTILL_ROOT/iteration-$(printf '%04d' "$NEXT_ITERATION")" ]]; then
  "$PYTHON_BIN" examples/colonist_1v1_distill.py --output "$DISTILL_ROOT" --verify
  echo "Reusing verified DAgger iteration $NEXT_ITERATION in $DISTILL_ROOT"
else
  "$PYTHON_BIN" examples/colonist_1v1_distill.py \
    --student "T:$STUDENT_CHECKPOINT" \
    --teacher F \
    --opponent F \
    --iteration "$NEXT_ITERATION" \
    --games "$DAGGER_GAMES" \
    --seed "$DAGGER_SEED" \
    --shard-games "$DAGGER_SHARD_GAMES" \
    --feature-profile raw \
    --output "$DISTILL_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/distill.log"
  "$PYTHON_BIN" examples/colonist_1v1_distill.py \
    --output "$DISTILL_ROOT" --verify
fi

write_status training hybrid-bc
if [[ -f "$BC_CHECKPOINT" && -f "$BC_META" ]]; then
  echo "Reusing completed BC checkpoint: $BC_CHECKPOINT"
else
  "$PYTHON_BIN" examples/colonist_1v1_bc.py \
    --data-dir "$BASE_F_F" "$BASE_VP_F" \
    --augmentation-data-dir "$DISTILL_ROOT" \
    --augmentation-weight "$DAGGER_AUGMENTATION_WEIGHT" \
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
    --run-dir "$BC_RUN" \
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
    --checkpoint-label "dagger-f-iteration-$NEXT_ITERATION" \
    --report "$REPORT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/evaluate.log"
fi

write_status complete "$REPORT"
echo "DAgger F iteration $NEXT_ITERATION complete: $REPORT"
