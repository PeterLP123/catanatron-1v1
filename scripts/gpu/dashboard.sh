#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-"$HOME/.venvs/catanatron-1v1"}
RUN_DIR=${RUN_DIR:-"$ROOT/runs/colonist_1v1"}
REFRESH_SECONDS=${REFRESH_SECONDS:-2}

[[ -f "$VENV/bin/activate" ]] || {
  echo "Missing environment: $VENV. Run scripts/gpu/setup_env.sh first." >&2
  exit 2
}

source "$VENV/bin/activate"

ARGS=(
  --run-dir "$RUN_DIR"
  --runs-root "$ROOT/runs"
  --refresh "$REFRESH_SECONDS"
)
if [[ "${SNAPSHOT:-0}" == "1" ]]; then
  ARGS+=(--once)
fi

exec python "$ROOT/examples/colonist_1v1_tui.py" "${ARGS[@]}"
