#!/usr/bin/env bash
# Full Colonist 1v1 pipeline (run inside tmux on EC2).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-runs/c1_${RUN_TAG}}"
DATA_ROOT="${DATA_ROOT:-data/c1_${RUN_TAG}}"
NUM_GAMES="${NUM_GAMES:-2000}"
PPO_STEPS="${PPO_STEPS:-500000}"
TRAIN_PRESET="${TRAIN_PRESET:-custom}"
CURRICULUM="${CURRICULUM:-strong}"
TEACHER_SPECS="${TEACHER_SPECS:-F,F VP,F}"
N_ENVS="${N_ENVS:-4}"
FAST_EVAL_GAMES="${FAST_EVAL_GAMES:-50}"
MILESTONE_EVAL_GAMES="${MILESTONE_EVAL_GAMES:-100}"
FINAL_EVAL_PROTOCOL="${FINAL_EVAL_PROTOCOL:-fast}"
SKIP_PHASE1="${SKIP_PHASE1:-0}"

mkdir -p "$RUN_DIR"

# Fresh log per invocation (keep prior logs as train.log.<tag>)
LOG_FILE="$RUN_DIR/train_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
ln -sf "$(basename "$LOG_FILE")" "$RUN_DIR/train.log"

DATA_DIRS=()
if [[ "$SKIP_PHASE1" != "1" ]]; then
  echo "=== $(date -Is) Phase 1: teacher data ($NUM_GAMES games/spec) -> $DATA_ROOT ==="
  for spec in $TEACHER_SPECS; do
    safe_spec="${spec//[: ,]/_}"
    out="$DATA_ROOT/$safe_spec"
    DATA_DIRS+=("$out")
    mkdir -p "$out"
    echo "--- teachers=$spec output=$out ---"
    python examples/colonist_1v1_generate_data.py --num "$NUM_GAMES" --teachers "$spec" --output "$out"
  done
else
  echo "=== $(date -Is) Phase 1: skipped (SKIP_PHASE1=1), using DATA_DIRS=${DATA_DIRS:-$DATA_ROOT/*} ==="
  # shellcheck disable=SC2206
  DATA_DIRS=(${DATA_DIRS:-$DATA_ROOT/*})
fi

echo "=== $(date -Is) Phase 2: behavioral cloning ==="
python examples/colonist_1v1_bc.py --data-dir "${DATA_DIRS[@]}" --epochs 5 --out "$RUN_DIR/bc.pt" \
  --tensorboard "$RUN_DIR/tb_bc" --run-dir "$RUN_DIR"

echo "=== $(date -Is) Phase 3: PPO + TensorBoard ==="
python examples/colonist_1v1_train.py \
  --preset "$TRAIN_PRESET" \
  --timesteps "$PPO_STEPS" \
  --n-envs "$N_ENVS" \
  --bc-checkpoint "$RUN_DIR/bc.pt" \
  --run-dir "$RUN_DIR" \
  --tensorboard \
  --mixed-league \
  --curriculum "$CURRICULUM" \
  --eval-protocol fast \
  --final-eval-protocol "$FINAL_EVAL_PROTOCOL" \
  --eval-freq 25000 \
  --eval-games "$FAST_EVAL_GAMES" \
  --save-freq 50000

echo "=== $(date -Is) Phase 4: milestone benchmark ==="
python examples/colonist_1v1_benchmark_report.py \
  --agent "L:$RUN_DIR/colonist_maskable_ppo.zip" \
  --protocol milestone \
  --num-games "$MILESTONE_EVAL_GAMES" \
  --gates \
  --run-dir "$RUN_DIR" \
  --checkpoint-label final \
  --training-timesteps "$PPO_STEPS" \
  --registry "$RUN_DIR/models_index.jsonl" \
  --output "$RUN_DIR/milestone_benchmark.json" || true

echo "=== $(date -Is) DONE ==="
