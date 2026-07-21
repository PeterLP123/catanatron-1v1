#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-root>" >&2
  exit 2
fi

OUTPUT_ROOT=$1
PYTHON_BIN=${PYTHON_BIN:-python}

echo "DAgger F pilot"
date --iso-8601=seconds
echo
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader
  echo
fi

if [[ -f "$OUTPUT_ROOT/pilot_status.txt" ]]; then
  cat "$OUTPUT_ROOT/pilot_status.txt"
else
  echo "Waiting for pilot_status.txt"
fi

if [[ -f "$OUTPUT_ROOT/data/manifest.json" ]]; then
  echo
  "$PYTHON_BIN" -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
print(f"dagger: games={manifest.get('"'"'games'"'"')} rows={manifest.get('"'"'rows'"'"')} iterations={len(manifest.get('"'"'iterations'"'"', []))}")
' "$OUTPUT_ROOT/data/manifest.json"
fi

if [[ -f "$OUTPUT_ROOT/bc/run_manifest.json" ]]; then
  echo
  "$PYTHON_BIN" -c '
import json, sys
manifest = json.load(open(sys.argv[1]))
print(f"bc: phase={manifest.get('"'"'phase'"'"')} checkpoint={manifest.get('"'"'bc_checkpoint'"'"')}")
' "$OUTPUT_ROOT/bc/run_manifest.json"
fi

for log_file in distill.log bc.log evaluate.log; do
  path="$OUTPUT_ROOT/logs/$log_file"
  if [[ -f "$path" ]]; then
    echo
    echo "Latest $log_file"
    tail -n 8 "$path"
  fi
done
