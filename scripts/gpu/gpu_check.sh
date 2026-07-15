#!/usr/bin/env bash
set -euo pipefail

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is unavailable; this does not appear to be a GPU host." >&2
  exit 2
}

hostname
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv

PROCESSES=$(nvidia-smi \
  --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader 2>/dev/null || true)

if [[ -n "$PROCESSES" ]]; then
  echo
  echo "GPU compute processes are already running:"
  echo "$PROCESSES"
  exit 1
fi

echo
echo "No active GPU compute process was reported. Recheck immediately before training."
