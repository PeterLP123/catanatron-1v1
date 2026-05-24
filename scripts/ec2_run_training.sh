#!/usr/bin/env bash
# Full Colonist 1v1 pipeline (run inside tmux on EC2).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUN_DIR="${RUN_DIR:-runs/c1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-data/run_${RUN_TAG}}"
NUM_GAMES="${NUM_GAMES:-2000}"
PPO_STEPS="${PPO_STEPS:-500000}"
SKIP_PHASE1="${SKIP_PHASE1:-0}"

mkdir -p "$RUN_DIR"

# Fresh log per invocation (keep prior logs as train.log.<tag>)
LOG_FILE="$RUN_DIR/train_${RUN_TAG}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
ln -sf "$(basename "$LOG_FILE")" "$RUN_DIR/train.log"

if [[ "$SKIP_PHASE1" != "1" ]]; then
  echo "=== $(date -Is) Phase 1: teacher data ($NUM_GAMES games) -> $DATA_DIR ==="
  mkdir -p "$DATA_DIR"
  python examples/colonist_1v1_generate_data.py --num "$NUM_GAMES" --teachers F,F --output "$DATA_DIR"
else
  echo "=== $(date -Is) Phase 1: skipped (SKIP_PHASE1=1), using $DATA_DIR ==="
fi

echo "=== $(date -Is) Phase 2: behavioral cloning ==="
python examples/colonist_1v1_bc.py --data-dir "$DATA_DIR" --epochs 5 --out "$RUN_DIR/bc.pt" \
  --tensorboard "$RUN_DIR/tb_bc"

echo "=== $(date -Is) Phase 3: PPO + TensorBoard ==="
python examples/colonist_1v1_train.py \
  --timesteps "$PPO_STEPS" \
  --n-envs 4 \
  --bc-checkpoint "$RUN_DIR/bc.pt" \
  --run-dir "$RUN_DIR" \
  --tensorboard \
  --eval-freq 25000 \
  --eval-games 50 \
  --save-freq 50000

echo "=== $(date -Is) DONE ==="
