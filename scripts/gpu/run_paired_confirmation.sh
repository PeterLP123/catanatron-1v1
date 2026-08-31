#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <candidate-bc.pt> <baseline-bc.pt> <output-root>" >&2
  exit 2
fi

CANDIDATE_CHECKPOINT=$1
BASELINE_CHECKPOINT=$2
OUTPUT_ROOT=$3
CONFIRM_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$CONFIRM_REPO_ROOT/catanatron${PYTHONPATH:+:$PYTHONPATH}"
cd "$CONFIRM_REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIRM_GAMES=${PAIRED_CONFIRM_GAMES:-800}
CONFIRM_SEED_ROUND=${PAIRED_CONFIRM_SEED_ROUND:-3}
PAIRED_DIR="$OUTPUT_ROOT/paired-promotion-f"
STATUS_FILE="$OUTPUT_ROOT/status.txt"

for checkpoint in "$CANDIDATE_CHECKPOINT" "$BASELINE_CHECKPOINT"; do
  [[ -f "$checkpoint" ]] || {
    echo "Required checkpoint not found: $checkpoint" >&2
    exit 2
  }
  [[ -f "${checkpoint%.pt}.meta.json" ]] || {
    echo "Required checkpoint metadata not found: ${checkpoint%.pt}.meta.json" >&2
    exit 2
  }
  [[ -f "${checkpoint%.pt}.schema.json" ]] || {
    echo "Required checkpoint schema not found: ${checkpoint%.pt}.schema.json" >&2
    exit 2
  }
done
if [[ "$CONFIRM_GAMES" -le 0 || "$CONFIRM_SEED_ROUND" -lt 0 ]]; then
  echo "Confirmation games must be positive and seed round non-negative" >&2
  exit 2
fi

SOURCE_DIFF_SHA256=$(git diff HEAD --binary | sha256sum | cut -d ' ' -f 1)
SOURCE_STATUS_SHA256=$(git status --porcelain=v1 --untracked-files=all | sha256sum | cut -d ' ' -f 1)
SOURCE_UNTRACKED_SHA256=$(
  git ls-files --others --exclude-standard -- catanatron examples scripts \
    | sort \
    | while IFS= read -r path; do sha256sum "$path"; done \
    | sha256sum \
    | cut -d ' ' -f 1
)
RUN_IDENTITY_FILE="$OUTPUT_ROOT/run_identity.sha256"
CURRENT_RUN_IDENTITY=$(
  {
    echo "candidate_checkpoint_sha256=$(sha256sum "$CANDIDATE_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "candidate_metadata_sha256=$(sha256sum "${CANDIDATE_CHECKPOINT%.pt}.meta.json" | cut -d ' ' -f 1)"
    echo "candidate_schema_sha256=$(sha256sum "${CANDIDATE_CHECKPOINT%.pt}.schema.json" | cut -d ' ' -f 1)"
    echo "baseline_checkpoint_sha256=$(sha256sum "$BASELINE_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "baseline_metadata_sha256=$(sha256sum "${BASELINE_CHECKPOINT%.pt}.meta.json" | cut -d ' ' -f 1)"
    echo "baseline_schema_sha256=$(sha256sum "${BASELINE_CHECKPOINT%.pt}.schema.json" | cut -d ' ' -f 1)"
    echo "games=$CONFIRM_GAMES"
    echo "seed_round=$CONFIRM_SEED_ROUND"
    echo "source_diff_sha256=$SOURCE_DIFF_SHA256"
    echo "source_status_sha256=$SOURCE_STATUS_SHA256"
    echo "source_untracked_sha256=$SOURCE_UNTRACKED_SHA256"
  } | sha256sum | cut -d ' ' -f 1
)
if [[ -f "$RUN_IDENTITY_FILE" ]]; then
  if [[ "$(tr -d '[:space:]' <"$RUN_IDENTITY_FILE")" != "$CURRENT_RUN_IDENTITY" ]]; then
    echo "Run configuration changed; choose a fresh output root: $OUTPUT_ROOT" >&2
    exit 3
  fi
elif [[ -d "$OUTPUT_ROOT" && -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]]; then
  echo "Existing output lacks an immutable run identity; choose a fresh output root: $OUTPUT_ROOT" >&2
  exit 3
else
  mkdir -p "$OUTPUT_ROOT"
  printf '%s\n' "$CURRENT_RUN_IDENTITY" >"$RUN_IDENTITY_FILE"
fi
mkdir -p "$OUTPUT_ROOT/logs"
write_status() {
  local state=$1
  local detail=${2:-none}
  {
    echo "state=$state"
    echo "detail=$detail"
    echo "updated=$(date -Iseconds)"
  } >"$STATUS_FILE"
}

if [[ ! -f "$OUTPUT_ROOT/run_record.txt" ]]; then
  {
    echo "base_commit=$(git rev-parse HEAD)"
    echo "source_diff_sha256=$SOURCE_DIFF_SHA256"
    echo "source_status_sha256=$SOURCE_STATUS_SHA256"
    echo "source_untracked_sha256=$SOURCE_UNTRACKED_SHA256"
    echo "run_identity_sha256=$CURRENT_RUN_IDENTITY"
    echo "paired_cli_sha256=$(sha256sum examples/colonist_1v1_paired_evaluate.py | cut -d ' ' -f 1)"
    echo "launcher_sha256=$(sha256sum scripts/gpu/run_paired_confirmation.sh | cut -d ' ' -f 1)"
    echo "candidate_checkpoint_sha256=$(sha256sum "$CANDIDATE_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "candidate_metadata_sha256=$(sha256sum "${CANDIDATE_CHECKPOINT%.pt}.meta.json" | cut -d ' ' -f 1)"
    echo "candidate_schema_sha256=$(sha256sum "${CANDIDATE_CHECKPOINT%.pt}.schema.json" | cut -d ' ' -f 1)"
    echo "baseline_checkpoint_sha256=$(sha256sum "$BASELINE_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "baseline_metadata_sha256=$(sha256sum "${BASELINE_CHECKPOINT%.pt}.meta.json" | cut -d ' ' -f 1)"
    echo "baseline_schema_sha256=$(sha256sum "${BASELINE_CHECKPOINT%.pt}.schema.json" | cut -d ' ' -f 1)"
    echo "host=$(hostname -f)"
    echo "started=$(date -Iseconds)"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    echo "opponent=F"
    echo "games=$CONFIRM_GAMES"
    echo "seed_suite=promotion"
    echo "seed_round=$CONFIRM_SEED_ROUND"
    echo "gate=paired_bootstrap_95pct_lower_bound_above_zero"
  } >"$OUTPUT_ROOT/run_record.txt"
fi

write_status evaluating-paired "games=$CONFIRM_GAMES seed_round=$CONFIRM_SEED_ROUND"
if [[ -f "$PAIRED_DIR/paired_comparison.json" \
  && -f "$PAIRED_DIR/candidate_report.json" \
  && -f "$PAIRED_DIR/baseline_report.json" ]]; then
  write_status blocked existing-paired-evidence
  echo "Refusing to reuse evaluation evidence; choose a fresh output root: $PAIRED_DIR" >&2
  exit 3
elif [[ -d "$PAIRED_DIR" && -n "$(find "$PAIRED_DIR" -mindepth 1 -print -quit)" ]]; then
  write_status blocked partial-paired-evaluation
  echo "Refusing to overwrite partial paired confirmation: $PAIRED_DIR" >&2
  exit 3
else
  set +e
  "$PYTHON_BIN" examples/colonist_1v1_paired_evaluate.py \
    --candidate "T:$CANDIDATE_CHECKPOINT" \
    --baseline "T:$BASELINE_CHECKPOINT" \
    --opponents F \
    --num-games "$CONFIRM_GAMES" \
    --seed-suite promotion \
    --seed-round "$CONFIRM_SEED_ROUND" \
    --minimum-delta 0 \
    --output-dir "$PAIRED_DIR" \
    2>&1 | tee "$OUTPUT_ROOT/logs/paired-promotion-f.log"
  PAIRED_GATE_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$PAIRED_GATE_RC" -gt 1 ]]; then
    write_status blocked "paired-evaluation-rc-$PAIRED_GATE_RC"
    exit "$PAIRED_GATE_RC"
  fi
fi

PAIRED_GATE_RC=${PAIRED_GATE_RC:-$(
  "$PYTHON_BIN" -c 'import json,sys; print(0 if json.load(open(sys.argv[1]))["all_gates_passed"] else 1)' \
    "$PAIRED_DIR/paired_comparison.json"
)}
{
  echo "paired_report_sha256=$(sha256sum "$PAIRED_DIR/paired_comparison.json" | cut -d ' ' -f 1)"
  echo "paired_gate_exit=$PAIRED_GATE_RC"
  echo "completed=$(date -Iseconds)"
} >>"$OUTPUT_ROOT/run_record.txt"
write_status complete "paired_gate=$PAIRED_GATE_RC"
echo "Paired confirmation complete: $OUTPUT_ROOT"
