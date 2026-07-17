#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-root>" >&2
  exit 2
fi

OUTPUT_ROOT=$1
PYTHON_BIN=${PYTHON_BIN:-python}

echo "BC anchor sweep"
date --iso-8601=seconds
echo
nvidia-smi \
  --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
echo

if [[ -f "$OUTPUT_ROOT/sweep_status.txt" ]]; then
  cat "$OUTPUT_ROOT/sweep_status.txt"
else
  echo "Waiting for sweep_status.txt"
fi

shopt -s nullglob
for manifest in "$OUTPUT_ROOT"/coef-*/run_manifest.json; do
  echo
  "$PYTHON_BIN" -c '
import json, pathlib, sys
manifest = json.load(open(sys.argv[1]))
training = manifest.get("training", {})
stop = training.get("retention_stop")
name = pathlib.Path(sys.argv[1]).parent.name
phase = manifest.get("phase")
anchor = training.get("bc_anchor_coef")
stop_text = stop or "none"
print(f"{name}: phase={phase} anchor={anchor} stop={stop_text}")
' "$manifest"
  events="$(dirname "$manifest")/training_events.jsonl"
  if [[ -f "$events" ]]; then
    tail -n 2 "$events"
  fi
done

candidate=$(sed -n 's/^candidate=//p' "$OUTPUT_ROOT/sweep_status.txt" 2>/dev/null || true)
log_file="$OUTPUT_ROOT/logs/$candidate.log"
if [[ -f "$log_file" ]]; then
  echo
  echo "Latest log: $log_file"
  tail -n 12 "$log_file"
fi
