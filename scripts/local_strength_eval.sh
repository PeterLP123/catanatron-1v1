#!/usr/bin/env bash
# Train + benchmark when EC2 checkpoint is unavailable (same recipe as ec2_run_training.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${RUN_DIR:-runs/local_strength_eval}"
DATA_DIR="${DATA_DIR:-data/local_strength_eval}"
NUM_GAMES="${NUM_GAMES:-150}"
PPO_STEPS="${PPO_STEPS:-200000}"
EVAL_GAMES="${EVAL_GAMES:-100}"

mkdir -p "$RUN_DIR" "$DATA_DIR"
exec > >(tee -a "$RUN_DIR/pipeline.log") 2>&1

echo "=== $(date -Is) Phase 1: teacher data ($NUM_GAMES games, F,F) ==="
python3 examples/colonist_1v1_generate_data.py --num "$NUM_GAMES" --teachers F,F --output "$DATA_DIR"

echo "=== $(date -Is) Phase 2: behavioral cloning ==="
python3 examples/colonist_1v1_bc.py --data-dir "$DATA_DIR" --epochs 5 --out "$RUN_DIR/bc.pt"

echo "=== $(date -Is) Phase 3: PPO ($PPO_STEPS steps) ==="
python3 examples/colonist_1v1_train.py \
  --run-dir "$RUN_DIR" \
  --timesteps "$PPO_STEPS" \
  --n-envs 4 \
  --bc-checkpoint "$RUN_DIR/bc.pt" \
  --eval-freq 50000 \
  --eval-games 30 \
  --skip-final-eval

MODEL="$RUN_DIR/colonist_maskable_ppo.zip"
AGENT="L:$MODEL"

echo "=== $(date -Is) Phase 4: benchmark (R, W, VP, F only) ==="
for opp in R W VP F; do
  echo "--- vs $opp ($EVAL_GAMES games) ---"
  python3 examples/colonist_1v1_evaluate.py \
    --agent "$AGENT" \
    --opponent "$opp" \
    --num-games "$EVAL_GAMES" \
    --report "$RUN_DIR/vs_${opp}.json"
done

python3 - <<PY
import json
from pathlib import Path

run = Path("$RUN_DIR")
rows = []
for p in sorted(run.glob("vs_*.json")):
    d = json.loads(p.read_text())
    m = d["matchups"][0]
    rows.append((m["opponent"], m["wins"], m["games"], m["win_rate"], m.get("ci_low"), m.get("ci_high")))
print("\n=== BENCHMARK SUMMARY ===")
for opp, w, n, wr, lo, hi in rows:
    print(f"  vs {opp:3s}: {w}/{n} = {wr*100:.1f}%  (95% CI {lo*100:.1f}–{hi*100:.1f}%)")
(run / "benchmark_summary.txt").write_text(
    "\n".join(f"{o}\t{w}\t{n}\t{wr}\t{lo}\t{hi}" for o, w, n, wr, lo, hi in rows) + "\n"
)
print(f"Wrote {run / 'benchmark_summary.txt'}")
PY

echo "=== $(date -Is) DONE ==="
