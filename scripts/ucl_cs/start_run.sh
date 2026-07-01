#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-"$HOME/.venvs/catanatron-1v1"}
TRAIN_PRESET=${TRAIN_PRESET:-standard}
RUN_NAME=${RUN_NAME:-"ucl_cs_${TRAIN_PRESET}_$(date +%Y%m%d_%H%M%S)"}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/$RUN_NAME"}
SESSION=${SESSION:-"catan-${RUN_NAME}"}
SESSION=$(printf '%s' "$SESSION" | tr -cs '[:alnum:]_-' '-')

command -v tmux >/dev/null 2>&1 || {
  echo "tmux is unavailable on this host." >&2
  exit 2
}
[[ -f "$VENV/bin/activate" ]] || {
  echo "Missing environment: $VENV. Run scripts/ucl_cs/setup_env.sh first." >&2
  exit 2
}
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 3
fi

bash "$ROOT/scripts/ucl_cs/gpu_check.sh"
mkdir -p "$RUN_DIR"

printf -v Q_ROOT '%q' "$ROOT"
printf -v Q_VENV '%q' "$VENV"
printf -v Q_PRESET '%q' "$TRAIN_PRESET"
printf -v Q_NAME '%q' "$RUN_NAME"
printf -v Q_RUN_DIR '%q' "$RUN_DIR"

TRAIN_COMMAND="cd $Q_ROOT && VENV=$Q_VENV TRAIN_PRESET=$Q_PRESET RUN_NAME=$Q_NAME RUN_DIR=$Q_RUN_DIR bash scripts/ucl_cs/train.sh; exec bash"
DASH_COMMAND="cd $Q_ROOT && sleep 2 && VENV=$Q_VENV RUN_DIR=$Q_RUN_DIR bash scripts/ucl_cs/dashboard.sh; exec bash"
GPU_COMMAND="watch -n 3 nvidia-smi; exec bash"

tmux new-session -d -s "$SESSION" -n training "$TRAIN_COMMAND"
tmux new-window -t "$SESSION" -n dashboard "$DASH_COMMAND"
tmux new-window -t "$SESSION" -n gpu "$GPU_COMMAND"

tmux set-option -t "$SESSION" status on
tmux set-option -t "$SESSION" status-interval 2
tmux set-option -t "$SESSION" status-style 'bg=#171713,fg=#f2efe6'
tmux set-option -t "$SESSION" status-left '#[fg=#f26a2e,bold] CATAN #[fg=#aaa797]'
tmux set-option -t "$SESSION" status-left-length 24
tmux set-option -t "$SESSION" status-right '#[fg=#9acb76]#H #[fg=#aaa797]%H:%M '
tmux set-window-option -t "$SESSION" window-status-current-style 'fg=#171713,bg=#f26a2e,bold'
tmux select-window -t "$SESSION:dashboard"

echo "Started tmux session: $SESSION"
echo "Run directory: $RUN_DIR"
echo "Detach: Ctrl-B then D"
echo "Reattach: tmux attach -t $SESSION"
exec tmux attach -t "$SESSION"
