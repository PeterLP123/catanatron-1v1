#!/usr/bin/env bash
# Reproducible local teacher-data, BC, PPO, and benchmark pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-runs/local_strength_eval}"
PPO_RUN_DIR="${PPO_RUN_DIR:-$RUN_DIR/ppo}"
DATA_ROOT="${DATA_ROOT:-data/local_strength_eval}"
NUM_GAMES="${NUM_GAMES:-150}"
PPO_STEPS="${PPO_STEPS:-200000}"
EVAL_GAMES="${EVAL_GAMES:-100}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-milestone}"
TEACHER_SPECS="${TEACHER_SPECS:-F,F VP,F}"
SEED="${SEED:-0}"
FEATURE_PROFILE="${FEATURE_PROFILE:-public_derived}"
RESUME_DATA="${RESUME_DATA:-0}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "venv/bin/python" ]]; then
    PYTHON="venv/bin/python"
  else
    PYTHON=$(command -v python3)
  fi
fi

mkdir -p "$RUN_DIR" "$DATA_ROOT"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

echo "=== $(date -Is) Phase 1: teacher data ($NUM_GAMES games/spec) ==="
DATA_DIRS=()
spec_index=0
for spec in $TEACHER_SPECS; do
  safe_spec="${spec//[: ,]/_}"
  out="$DATA_ROOT/$safe_spec"
  DATA_DIRS+=("$out")
  DATA_ARGS=(
    --num "$NUM_GAMES"
    --seed "$((SEED + spec_index * NUM_GAMES))"
    --teachers "$spec"
    --output "$out"
    --feature-profile "$FEATURE_PROFILE"
  )
  [[ "$RESUME_DATA" == "1" ]] && DATA_ARGS+=(--resume)
  "$PYTHON" examples/colonist_1v1_generate_data.py "${DATA_ARGS[@]}"
  spec_index=$((spec_index + 1))
done

echo "=== $(date -Is) Phase 2: behavioral cloning ==="
"$PYTHON" examples/colonist_1v1_bc.py \
  --data-dir "${DATA_DIRS[@]}" \
  --feature-profile "$FEATURE_PROFILE" \
  --seed "$SEED" \
  --epochs 5 \
  --out "$RUN_DIR/bc.pt" \
  --run-dir "$RUN_DIR"

echo "=== $(date -Is) Phase 3: PPO ($PPO_STEPS steps) ==="
"$PYTHON" examples/colonist_1v1_train.py \
  --run-dir "$PPO_RUN_DIR" \
  --timesteps "$PPO_STEPS" \
  --seed "$SEED" \
  --n-envs 4 \
  --bc-checkpoint "$RUN_DIR/bc.pt" \
  --feature-profile "$FEATURE_PROFILE" \
  --mixed-league \
  --curriculum balanced \
  --eval-freq 50000 \
  --eval-protocol fast \
  --eval-games 30 \
  --skip-final-eval

MODEL="$PPO_RUN_DIR/colonist_maskable_ppo.zip"
AGENT="L:$MODEL"

echo "=== $(date -Is) Phase 4: locked benchmark ==="
BENCHMARK_STATUS=0
"$PYTHON" examples/colonist_1v1_evaluate.py \
  --agent "$AGENT" \
  --benchmark \
  --gates \
  --protocol "$EVAL_PROTOCOL" \
  --num-games "$EVAL_GAMES" \
  --eval-kind final \
  --seed-suite final \
  --gate-mode lower_bound \
  --run-dir "$PPO_RUN_DIR" \
  --checkpoint-label local-strength \
  --training-timesteps "$PPO_STEPS" \
  --report "$RUN_DIR/final_benchmark.json" || BENCHMARK_STATUS=$?

"$PYTHON" - <<PY
import json
from pathlib import Path

run = Path("$RUN_DIR")
d = json.loads((run / "final_benchmark.json").read_text())
rows = [
    (m["opponent"], m["wins"], m["games"], m["win_rate"], m.get("wilson_low"), m.get("wilson_high"))
    for m in d["matchups"]
]
print("\n=== BENCHMARK SUMMARY ===")
for opp, w, n, wr, lo, hi in rows:
    print(f"  vs {opp:3s}: {w}/{n} = {wr*100:.1f}%  (95% CI {lo*100:.1f}–{hi*100:.1f}%)")
(run / "benchmark_summary.txt").write_text(
    "\n".join(f"{o}\t{w}\t{n}\t{wr}\t{lo}\t{hi}" for o, w, n, wr, lo, hi in rows) + "\n"
)
print(f"Wrote {run / 'benchmark_summary.txt'}")
PY

echo "=== $(date -Is) DONE ==="
exit "$BENCHMARK_STATUS"
