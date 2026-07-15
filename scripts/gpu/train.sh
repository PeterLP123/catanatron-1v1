#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-"$HOME/.venvs/catanatron-1v1"}
GPU_ID=${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}
TRAIN_PRESET=${TRAIN_PRESET:-standard}
RUN_NAME=${RUN_NAME:-"gpu_${TRAIN_PRESET}_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/$RUN_NAME"}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
BC_CHECKPOINT=${BC_CHECKPOINT:-}
SEED=${SEED:-0}
EVAL_PROTOCOL=${EVAL_PROTOCOL:-fast}
FINAL_EVAL_PROTOCOL=${FINAL_EVAL_PROTOCOL:-fast}
FINAL_EVAL_GAMES=${FINAL_EVAL_GAMES:-}
FINAL_GATE_MODE=${FINAL_GATE_MODE:-lower_bound}
PROMOTION_EVAL_FREQ=${PROMOTION_EVAL_FREQ:-}
PROMOTION_EVAL_GAMES=${PROMOTION_EVAL_GAMES:-}
PROMOTION_EVAL_PROTOCOL=${PROMOTION_EVAL_PROTOCOL:-milestone}
FEATURE_PROFILE=${FEATURE_PROFILE:-raw}
HUMAN_VISIBLE_OBS=${HUMAN_VISIBLE_OBS:-0}
VISIBLE_VP_REWARD=${VISIBLE_VP_REWARD:-0}
SKIP_FINAL_EVAL=${SKIP_FINAL_EVAL:-0}
NO_RANDOMIZE_SEATS=${NO_RANDOMIZE_SEATS:-0}
TEACHER_CODES=${TEACHER_CODES:-}
CURRICULUM=${CURRICULUM:-}
TIMESTEPS=${TIMESTEPS:-}
SAVE_FREQ=${SAVE_FREQ:-}
EVAL_FREQ=${EVAL_FREQ:-}
EVAL_GAMES=${EVAL_GAMES:-}
N_ENVS=${N_ENVS:-}
EXPERIMENT_ID=${EXPERIMENT_ID:-}
VEC_ENV=${VEC_ENV:-auto}
VEC_START_METHOD=${VEC_START_METHOD:-auto}
LEARNING_RATE=${LEARNING_RATE:-}
GAMMA=${GAMMA:-}
GAE_LAMBDA=${GAE_LAMBDA:-}
N_STEPS=${N_STEPS:-}
BATCH_SIZE=${BATCH_SIZE:-}
N_EPOCHS=${N_EPOCHS:-}
ENT_COEF=${ENT_COEF:-}
CLIP_RANGE=${CLIP_RANGE:-}
VF_COEF=${VF_COEF:-}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-}

[[ -f "$VENV/bin/activate" ]] || {
  echo "Missing environment: $VENV. Run scripts/gpu/setup_env.sh first." >&2
  exit 2
}

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is unavailable; run this on an NVIDIA GPU host." >&2
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
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_DIR"
TEMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/catanatron-gpu.XXXXXX")
trap 'echo "Temporary launch logs retained at: $TEMP_DIR" >&2' EXIT
git -C "$ROOT" rev-parse HEAD > "$TEMP_DIR/git_commit.txt"
nvidia-smi > "$TEMP_DIR/nvidia_smi_start.txt"

ARGS=(
  --preset "$TRAIN_PRESET"
  --run-dir "$RUN_DIR"
  --mixed-league
  --tensorboard
  --seed "$SEED"
  --eval-protocol "$EVAL_PROTOCOL"
  --final-eval-protocol "$FINAL_EVAL_PROTOCOL"
  --final-gate-mode "$FINAL_GATE_MODE"
  --promotion-eval-protocol "$PROMOTION_EVAL_PROTOCOL"
  --feature-profile "$FEATURE_PROFILE"
  --vec-env "$VEC_ENV"
  --vec-start-method "$VEC_START_METHOD"
)
if [[ -n "$RESUME_CHECKPOINT" ]]; then
  ARGS+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi
if [[ -n "$BC_CHECKPOINT" ]]; then
  ARGS+=(--bc-checkpoint "$BC_CHECKPOINT")
fi
if [[ -n "$FINAL_EVAL_GAMES" ]]; then
  ARGS+=(--final-eval-games "$FINAL_EVAL_GAMES")
fi
if [[ -n "$PROMOTION_EVAL_FREQ" ]]; then
  ARGS+=(--promotion-eval-freq "$PROMOTION_EVAL_FREQ")
fi
if [[ -n "$PROMOTION_EVAL_GAMES" ]]; then
  ARGS+=(--promotion-eval-games "$PROMOTION_EVAL_GAMES")
fi
if [[ "$HUMAN_VISIBLE_OBS" == "1" ]]; then
  ARGS+=(--human-visible-obs)
fi
if [[ "$VISIBLE_VP_REWARD" == "1" ]]; then
  ARGS+=(--visible-vp-reward)
fi
if [[ "$SKIP_FINAL_EVAL" == "1" ]]; then
  ARGS+=(--skip-final-eval)
fi
if [[ "$NO_RANDOMIZE_SEATS" == "1" ]]; then
  ARGS+=(--no-randomize-seats)
fi
if [[ -n "$TEACHER_CODES" ]]; then
  IFS=',' read -r -a TEACHER_CODE_ARGS <<< "$TEACHER_CODES"
  ARGS+=(--teacher-codes "${TEACHER_CODE_ARGS[@]}")
fi
if [[ -n "$CURRICULUM" ]]; then
  ARGS+=(--curriculum "$CURRICULUM")
fi
if [[ "$TRAIN_PRESET" == "custom" ]]; then
  [[ -n "$TIMESTEPS" ]] && ARGS+=(--timesteps "$TIMESTEPS")
  [[ -n "$SAVE_FREQ" ]] && ARGS+=(--save-freq "$SAVE_FREQ")
  [[ -n "$EVAL_FREQ" ]] && ARGS+=(--eval-freq "$EVAL_FREQ")
  [[ -n "$EVAL_GAMES" ]] && ARGS+=(--eval-games "$EVAL_GAMES")
  [[ -n "$N_ENVS" ]] && ARGS+=(--n-envs "$N_ENVS")
fi
if [[ -n "$EXPERIMENT_ID" ]]; then
  ARGS+=(--run-id "$EXPERIMENT_ID")
fi
[[ -n "$LEARNING_RATE" ]] && ARGS+=(--learning-rate "$LEARNING_RATE")
[[ -n "$GAMMA" ]] && ARGS+=(--gamma "$GAMMA")
[[ -n "$GAE_LAMBDA" ]] && ARGS+=(--gae-lambda "$GAE_LAMBDA")
[[ -n "$N_STEPS" ]] && ARGS+=(--n-steps "$N_STEPS")
[[ -n "$BATCH_SIZE" ]] && ARGS+=(--batch-size "$BATCH_SIZE")
[[ -n "$N_EPOCHS" ]] && ARGS+=(--n-epochs "$N_EPOCHS")
[[ -n "$ENT_COEF" ]] && ARGS+=(--ent-coef "$ENT_COEF")
[[ -n "$CLIP_RANGE" ]] && ARGS+=(--clip-range "$CLIP_RANGE")
[[ -n "$VF_COEF" ]] && ARGS+=(--vf-coef "$VF_COEF")
[[ -n "$MAX_GRAD_NORM" ]] && ARGS+=(--max-grad-norm "$MAX_GRAD_NORM")

python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
echo "host=$(hostname) gpu=$GPU_ID preset=$TRAIN_PRESET experiment=${EXPERIMENT_ID:-none} run_dir=$RUN_DIR"
set +e
python "$ROOT/examples/colonist_1v1_train.py" "${ARGS[@]}" 2>&1 | tee "$TEMP_DIR/console.log"
TRAIN_STATUS=${PIPESTATUS[0]}
set -e

nvidia-smi > "$TEMP_DIR/nvidia_smi_end.txt"
mv "$TEMP_DIR/git_commit.txt" "$RUN_DIR/git_commit.txt"
mv "$TEMP_DIR/nvidia_smi_start.txt" "$RUN_DIR/nvidia_smi_start.txt"
mv "$TEMP_DIR/nvidia_smi_end.txt" "$RUN_DIR/nvidia_smi_end.txt"
mv "$TEMP_DIR/console.log" "$RUN_DIR/console.log"
rmdir "$TEMP_DIR"
trap - EXIT
if [[ "$TRAIN_STATUS" -ne 0 ]]; then
  echo "Training failed with status $TRAIN_STATUS; inspect $RUN_DIR/console.log" >&2
  exit "$TRAIN_STATUS"
fi
echo "Training artifacts: $RUN_DIR"
