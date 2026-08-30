#!/usr/bin/env bash
# Evidence-first path to a stronger bot: reconstruct the hybrid-BC parent,
# then run one bounded DAgger F iteration against that parent.
set -euo pipefail
export PYTHONUNBUFFERED=1

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
PARENT_ROOT=${PARENT_ROOT:-$ROOT/runs/bc-hybrid-sweep/w0030}
DAGGER_ROOT=${DAGGER_ROOT:-$ROOT/runs/28-dagger-f-s101}
PARENT_CHECKPOINT=${PARENT_CHECKPOINT:-$PARENT_ROOT/bc.pt}
DATA_F_F=${DATA_F_F:-$ROOT/data/hard_state_v2/F_F}
DATA_VP_F=${DATA_VP_F:-$ROOT/data/hard_state_v2/VP_F}

export PYTHON_BIN
"$ROOT/scripts/run_hybrid_bc_parent.sh" "$PARENT_ROOT"
"$ROOT/scripts/gpu/run_dagger_f_pilot.sh" \
  "$PARENT_CHECKPOINT" \
  "$DATA_F_F" \
  "$DATA_VP_F" \
  "$DAGGER_ROOT"

echo
echo "Parent report:  $PARENT_ROOT/promotion_benchmark.json"
echo "DAgger report:  $DAGGER_ROOT/promotion_benchmark.json"
echo "Keep the DAgger child only if F win rate or VP margin improved and R/W/VP held."
