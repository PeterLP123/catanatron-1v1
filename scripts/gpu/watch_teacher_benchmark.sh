#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <output-root>" >&2
  exit 2
fi

OUTPUT_ROOT=$1
PYTHON_BIN=${PYTHON_BIN:-python}
REPORT="$OUTPUT_ROOT/report.json"
EVENTS="$OUTPUT_ROOT/report.events.jsonl"

echo "Teacher population benchmark"
date --iso-8601=seconds
uptime
echo
nvidia-smi \
  --query-gpu=name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader
echo
echo "Benchmark process"
ps -u "$USER" -o pid=,etime=,%cpu=,%mem=,command= \
  | grep '[c]olonist_1v1_teacher_benchmark.py' || true

if [[ -f "$REPORT" ]]; then
  echo
  "$PYTHON_BIN" -c '
import json, sys
report = json.load(open(sys.argv[1]))
status = report.get("status", {})
profiles_completed = status.get("profiles_completed", 0)
profiles_expected = status.get("profiles_expected", 0)
cells_completed = status.get("cells_completed", 0)
cells_expected = status.get("cells_expected", 0)
complete = status.get("complete", False)
print(
    f"profiles={profiles_completed}/{profiles_expected} "
    f"cells={cells_completed}/{cells_expected} complete={complete}"
)
for row in report.get("summaries", []):
    profile = row.get("profile") or {}
    candidate = row.get("candidate")
    completed = row.get("cells_completed")
    expected = row.get("cells_expected")
    p95 = profile.get("p95_latency_ms", 0.0)
    common = row.get("common_population", {}).get("weighted_score", 0.0)
    strength = row.get("strength_population", {}).get("weighted_score", 0.0)
    rates = row.get("win_rates", {})
    print(
        f"{candidate}: cells={completed}/{expected} p95={p95:.1f}ms "
        f"common={common:.3f} strength={strength:.3f} rates={rates}"
    )
' "$REPORT"
else
  echo
  echo "Waiting for $REPORT"
fi

if [[ -f "$EVENTS" ]]; then
  echo
  echo "Latest events"
  tail -n 8 "$EVENTS"
fi
