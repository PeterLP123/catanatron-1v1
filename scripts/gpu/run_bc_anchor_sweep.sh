#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <bc-checkpoint.pt> <output-root>" >&2
  exit 2
fi

BC_CHECKPOINT=$1
OUTPUT_ROOT=$2
PYTHON_BIN=${PYTHON_BIN:-python}
COEFFICIENTS=${BC_ANCHOR_COEFFICIENTS:-"0 0.01 0.03 0.10"}
TIMESTEPS=${BC_ANCHOR_TIMESTEPS:-50000}
SEED=${BC_ANCHOR_SEED:-101}
N_ENVS=${BC_ANCHOR_N_ENVS:-4}
EVAL_FREQ=${BC_ANCHOR_EVAL_FREQ:-10000}
EVAL_GAMES=${BC_ANCHOR_EVAL_GAMES:-20}
FINAL_EVAL_GAMES=${BC_ANCHOR_FINAL_EVAL_GAMES:-20}

if [[ ! -f "$BC_CHECKPOINT" ]]; then
  echo "BC checkpoint not found: $BC_CHECKPOINT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT/logs"
STATUS_FILE="$OUTPUT_ROOT/sweep_status.txt"
TOTAL=$(wc -w <<<"$COEFFICIENTS" | tr -d ' ')
INDEX=0

write_status() {
  local state=$1
  local candidate=${2:-none}
  local coefficient=${3:-none}
  local exit_code=${4:-none}
  {
    echo "state=$state"
    echo "candidate=$candidate"
    echo "coefficient=$coefficient"
    echo "index=$INDEX/$TOTAL"
    echo "exit_code=$exit_code"
    echo "updated=$(date --iso-8601=seconds)"
  } >"$STATUS_FILE"
}

write_status starting
for coefficient in $COEFFICIENTS; do
  INDEX=$((INDEX + 1))
  candidate=$(
    "$PYTHON_BIN" -c \
      'import sys; print(f"coef-{round(float(sys.argv[1]) * 100):03d}")' \
      "$coefficient"
  )
  run_dir="$OUTPUT_ROOT/$candidate"
  log_file="$OUTPUT_ROOT/logs/$candidate.log"

  if [[ -e "$run_dir" ]]; then
    phase=$(
      "$PYTHON_BIN" -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("phase", "unknown"))' \
        "$run_dir/run_manifest.json" 2>/dev/null || true
    )
    if [[ "$phase" == "done" ]]; then
      echo "Skipping completed candidate $candidate"
      continue
    fi
    write_status blocked "$candidate" "$coefficient"
    echo "Refusing to overwrite incomplete candidate directory: $run_dir" >&2
    exit 3
  fi

  write_status running "$candidate" "$coefficient"
  echo "Starting $candidate (coefficient=$coefficient)"
  set +e
  "$PYTHON_BIN" examples/colonist_1v1_train.py \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --n-envs "$N_ENVS" \
    --run-dir "$run_dir" \
    --save-freq "$EVAL_FREQ" \
    --eval-freq "$EVAL_FREQ" \
    --eval-games "$EVAL_GAMES" \
    --eval-protocol fast \
    --final-eval-protocol fast \
    --final-eval-games "$FINAL_EVAL_GAMES" \
    --final-gate-mode point \
    --bc-checkpoint "$BC_CHECKPOINT" \
    --bc-anchor-coef "$coefficient" \
    --learning-rate 3e-5 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 3 \
    --clip-range 0.1 \
    --curriculum balanced \
    --mixed-league \
    --retention-min-f-win-rate 0.10 \
    --retention-require-weak-gates \
    2>&1 | tee "$log_file"
  exit_code=${PIPESTATUS[0]}
  set -e

  if [[ $exit_code -ne 0 ]]; then
    write_status failed "$candidate" "$coefficient" "$exit_code"
    exit "$exit_code"
  fi
  write_status candidate_complete "$candidate" "$coefficient" 0
done

INDEX=$TOTAL
write_status complete all all 0
echo "BC anchor sweep complete: $OUTPUT_ROOT"
