#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-"$HOME/.venvs/catanatron-1v1"}
GPU_ID=${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}
TRAIN_PRESET=${TRAIN_PRESET:-standard}
RUN_NAME=${RUN_NAME:-"ucl_cs_${TRAIN_PRESET}_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/$RUN_NAME"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
BC_CHECKPOINT=${BC_CHECKPOINT:-}

[[ -f "$VENV/bin/activate" ]] || {
  echo "Missing environment: $VENV. Run scripts/ucl_cs/setup_env.sh first." >&2
  exit 2
}

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is unavailable; run this on a CS GPU host." >&2
  exit 2
}

PROCESSES=$(nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader 2>/dev/null || true)
if [[ -n "$PROCESSES" && "${ALLOW_BUSY_GPU:-0}" != "1" ]]; then
  echo "Refusing to start because a GPU compute process is already running:" >&2
  echo "$PROCESSES" >&2
  echo "Choose another host. Set ALLOW_BUSY_GPU=1 only when you know the process is yours." >&2
  exit 3
fi

source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$ROOT/catanatron"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR"
git -C "$ROOT" rev-parse HEAD > "$RUN_DIR/git_commit.txt"
nvidia-smi > "$RUN_DIR/nvidia_smi_start.txt"

ARGS=(
  --preset "$TRAIN_PRESET"
  --run-dir "$RUN_DIR"
  --mixed-league
  --tensorboard
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
if [[ -n "$BC_CHECKPOINT" ]]; then
  ARGS+=(--bc-checkpoint "$BC_CHECKPOINT")
fi

python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
echo "host=$(hostname) gpu=$GPU_ID preset=$TRAIN_PRESET run_dir=$RUN_DIR"
python "$ROOT/examples/colonist_1v1_train.py" "${ARGS[@]}" 2>&1 | tee "$RUN_DIR/console.log"

nvidia-smi > "$RUN_DIR/nvidia_smi_end.txt"
echo "Training artifacts: $RUN_DIR"
