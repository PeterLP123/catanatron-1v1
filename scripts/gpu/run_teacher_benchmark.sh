#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-root>" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUTPUT_ROOT=$1
PYTHON_BIN=${PYTHON_BIN:-python}
CANDIDATES=${TEACHER_CANDIDATES:-"AB:2,M:200,M:800,M:2000"}
OPPONENTS=${TEACHER_OPPONENTS:-"R,W,VP,F,G:25,M:200,AB:2"}
NUM_GAMES=${TEACHER_NUM_GAMES:-4}
SEED=${TEACHER_SEED:-20260717}
PROFILE_SAMPLES=${TEACHER_PROFILE_SAMPLES:-1}
PROFILE_SEED=${TEACHER_PROFILE_SEED:-20260617}
REPORT="$OUTPUT_ROOT/report.json"

mkdir -p "$OUTPUT_ROOT"
ARGS=(
  --candidates "$CANDIDATES"
  --opponents "$OPPONENTS"
  --num-games "$NUM_GAMES"
  --seed "$SEED"
  --profile-samples "$PROFILE_SAMPLES"
  --profile-seed "$PROFILE_SEED"
  --report "$REPORT"
)
if [[ -f "$REPORT" ]]; then
  ARGS+=(--resume)
fi
if [[ -n "${TEACHER_MAX_CELLS:-}" ]]; then
  ARGS+=(--max-cells "$TEACHER_MAX_CELLS")
fi
if [[ "${TEACHER_PROFILE_ONLY:-0}" == "1" ]]; then
  ARGS+=(--profile-only)
fi

cd "$ROOT"
exec "$PYTHON_BIN" examples/colonist_1v1_teacher_benchmark.py "${ARGS[@]}"
