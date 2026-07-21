#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <student-bc.pt> <base-f-f-dir> <base-vp-f-dir> <output-root>" >&2
  exit 2
fi

STUDENT_CHECKPOINT=$1
BASE_F_F=$2
BASE_VP_F=$3
OUTPUT_ROOT=$4
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
for required_dir in "$BASE_F_F" "$BASE_VP_F"; do
  if [[ ! -d "$required_dir" ]]; then
    echo "Required directory not found: $required_dir" >&2
    exit 2
  fi
done

DATA_ROOT="$OUTPUT_ROOT/data"
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

write_status collecting dagger-f
if [[ -f "$DATA_ROOT/manifest.json" ]]; then
  "$PYTHON_BIN" examples/colonist_1v1_distill.py \
    --output "$DATA_ROOT" --verify
  echo "Reusing verified DAgger data: $DATA_ROOT"
elif [[ -d "$DATA_ROOT" && -n "$(find "$DATA_ROOT" -mindepth 1 -print -quit)" ]]; then
  write_status blocked partial-data
  echo "Refusing to overwrite partial immutable DAgger data: $DATA_ROOT" >&2
  exit 3
else
  "$PYTHON_BIN" examples/colonist_1v1_distill.py \
    --student "T:$STUDENT_CHECKPOINT" \
    --teacher F \
    --opponent F \
    --iteration 0 \
    --games "$DAGGER_GAMES" \
    --seed "$DAGGER_SEED" \
    --shard-games "$DAGGER_SHARD_GAMES" \
    --feature-profile raw \
    --output "$DATA_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/distill.log"
  "$PYTHON_BIN" examples/colonist_1v1_distill.py \
    --output "$DATA_ROOT" --verify
fi

write_status training hybrid-bc
if [[ -f "$BC_CHECKPOINT" && -f "$BC_META" ]]; then
  echo "Reusing completed BC checkpoint: $BC_CHECKPOINT"
elif [[ -d "$BC_RUN" && -n "$(find "$BC_RUN" -mindepth 1 -print -quit)" ]]; then
  write_status blocked partial-bc
  echo "Refusing to overwrite partial BC run: $BC_RUN" >&2
  exit 3
else
  "$PYTHON_BIN" examples/colonist_1v1_bc.py \
    --data-dir "$BASE_F_F" "$BASE_VP_F" \
    --augmentation-data-dir "$DATA_ROOT" \
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
    --checkpoint-label dagger-f-iteration-0 \
    --report "$REPORT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/evaluate.log"
fi

write_status complete "$REPORT"
echo "DAgger F pilot complete: $REPORT"
