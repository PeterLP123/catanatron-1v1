#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-root>" >&2
  exit 2
fi

OUTPUT_ROOT=$1
PYTHON_BIN=${PYTHON_BIN:-python}

echo "Hybrid-BC parent reconstruction"
date --iso-8601=seconds
echo

if [[ -f "$OUTPUT_ROOT/parent_status.txt" ]]; then
  cat "$OUTPUT_ROOT/parent_status.txt"
else
  echo "Waiting for parent_status.txt"
fi

for meta in data/hard_state_v2/F_F/dataset_meta.json data/hard_state_v2/VP_F/dataset_meta.json; do
  if [[ -f "$meta" ]]; then
    echo
    "$PYTHON_BIN" - "$meta" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print(
    f"{sys.argv[1]}: status={meta.get('status')} "
    f"games={meta.get('completed_games')}/{meta.get('requested_games')} "
    f"rows={meta.get('rows')}"
)
PY
  fi
done

if [[ -f "$OUTPUT_ROOT/bc.meta.json" ]]; then
  echo
  "$PYTHON_BIN" - "$OUTPUT_ROOT/bc.meta.json" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1]))
print(
    f"bc: loss={meta.get('loss_name')} epoch={meta.get('best_epoch')} "
    f"val_loss={meta.get('val_loss')} selection={meta.get('selection_metric')}"
)
PY
fi

for log_file in generate_ff.log generate_vpf.log bc.log evaluate.log; do
  path="$OUTPUT_ROOT/logs/$log_file"
  if [[ -f "$path" ]]; then
    echo
    echo "Latest $log_file"
    tail -n 8 "$path"
  fi
done
