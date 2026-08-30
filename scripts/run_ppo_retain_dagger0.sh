#!/usr/bin/env bash
# Retention-gated PPO warm-start from the kept DAgger iteration 0 parent.
#
# One-variable treatment: the same anchored PPO recipe that forgot the weaker
# hybrid-BC parent, now starting from DAgger-0 (promotion F=36%). Coefficient
# 10 was the least-bad prior (1/20 vs F at the 10k stop). Stop if a 20-game
# development eval falls below F 10% or loses an R/W/VP point gate.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
BC_CHECKPOINT=${BC_CHECKPOINT:-$ROOT/runs/28-dagger-f-s101/bc/bc.pt}
OUTPUT_ROOT=${1:-$ROOT/runs/31-ppo-retain-dagger0}
PPO_RUN=${PPO_RUN:-$OUTPUT_ROOT/ppo}
ANCHOR_COEF=${PPO_ANCHOR_COEF:-10}
TIMESTEPS=${PPO_TIMESTEPS:-50000}
SEED=${PPO_SEED:-101}
N_ENVS=${PPO_N_ENVS:-4}
EVAL_FREQ=${PPO_EVAL_FREQ:-10000}
EVAL_GAMES=${PPO_EVAL_GAMES:-20}
PROMOTION_GAMES=${PPO_PROMOTION_GAMES:-50}
FINAL_ZIP=${FINAL_ZIP:-$PPO_RUN/colonist_maskable_ppo.zip}
REPORT=${REPORT:-$OUTPUT_ROOT/promotion_benchmark.json}
STATUS_FILE="$OUTPUT_ROOT/pilot_status.txt"

if [[ ! -f "$BC_CHECKPOINT" ]]; then
  echo "DAgger-0 BC checkpoint not found: $BC_CHECKPOINT" >&2
  exit 2
fi

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

write_status training "anchor-$ANCHOR_COEF"
if [[ -f "$PPO_RUN/run_manifest.json" ]]; then
  phase=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase",""))' "$PPO_RUN/run_manifest.json")
  if [[ "$phase" == "done" || "$phase" == "training_complete" ]]; then
    echo "Reusing completed PPO run: $PPO_RUN"
  else
    write_status blocked "incomplete-ppo-run"
    echo "Refusing to overwrite an incomplete PPO run: $PPO_RUN (phase=$phase)" >&2
    exit 3
  fi
else
  "$PYTHON_BIN" examples/colonist_1v1_train.py \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    --n-envs "$N_ENVS" \
    --run-dir "$PPO_RUN" \
    --save-freq "$EVAL_FREQ" \
    --eval-freq "$EVAL_FREQ" \
    --eval-games "$EVAL_GAMES" \
    --eval-protocol fast \
    --final-eval-protocol fast \
    --final-eval-games "$EVAL_GAMES" \
    --final-gate-mode point \
    --bc-checkpoint "$BC_CHECKPOINT" \
    --bc-anchor-coef "$ANCHOR_COEF" \
    --learning-rate 3e-5 \
    --n-steps 2048 \
    --batch-size 64 \
    --n-epochs 3 \
    --clip-range 0.1 \
    --curriculum balanced \
    --mixed-league \
    --feature-profile raw \
    --retention-min-f-win-rate 0.10 \
    --retention-require-weak-gates \
    2>&1 | tee "$OUTPUT_ROOT/logs/train.log"
fi

if [[ ! -f "$FINAL_ZIP" ]]; then
  write_status blocked "missing-final-zip"
  echo "PPO finished without $FINAL_ZIP" >&2
  exit 3
fi

write_status evaluating promotion
if [[ -f "$REPORT" ]]; then
  echo "Reusing promotion report: $REPORT"
else
  "$PYTHON_BIN" examples/colonist_1v1_evaluate.py \
    --agent "L:$FINAL_ZIP" \
    --benchmark \
    --protocol fast \
    --num-games "$PROMOTION_GAMES" \
    --eval-kind promotion \
    --gate-mode point \
    --checkpoint-label ppo-retain-dagger0-c10 \
    --report "$REPORT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/evaluate.log"
fi

write_status complete "$REPORT"
echo "Retention-gated PPO complete: $FINAL_ZIP"
echo "Promotion report: $REPORT"
