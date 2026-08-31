#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "Usage: $0 <base-f-f-dir> <base-vp-f-dir> <dagger-data-dir> <control-bc.pt> <output-root>" >&2
  exit 2
fi

BASE_F_F=$1
BASE_VP_F=$2
DAGGER_DATA=$3
CONTROL_CHECKPOINT=$4
OUTPUT_ROOT=$5
FACTORED_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$FACTORED_REPO_ROOT/catanatron${PYTHONPATH:+:$PYTHONPATH}"
cd "$FACTORED_REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
BC_EPOCHS=${FACTORED_BC_EPOCHS:-10}
BC_SEED=${FACTORED_BC_SEED:-101}
EMBEDDING_DIM=${FACTORED_EMBEDDING_DIM:-224}
PAIRED_GAMES=${FACTORED_PAIRED_GAMES:-200}
BC_DEVICE=${FACTORED_DEVICE:-cuda}

case "$BC_DEVICE" in
  auto|cpu|cuda|mps) ;;
  *)
    echo "FACTORED_DEVICE must be one of: auto, cpu, cuda, mps" >&2
    exit 2
    ;;
esac

for required_dir in "$BASE_F_F" "$BASE_VP_F" "$DAGGER_DATA"; do
  [[ -d "$required_dir" ]] || {
    echo "Required directory not found: $required_dir" >&2
    exit 2
  }
done
[[ -f "$CONTROL_CHECKPOINT" ]] || {
  echo "Required control checkpoint not found: $CONTROL_CHECKPOINT" >&2
  exit 2
}

SOURCE_DIFF_SHA256=$(git diff HEAD --binary | sha256sum | cut -d ' ' -f 1)
SOURCE_STATUS_SHA256=$(git status --porcelain=v1 --untracked-files=all | sha256sum | cut -d ' ' -f 1)
SOURCE_UNTRACKED_SHA256=$(
  git ls-files --others --exclude-standard -- catanatron examples scripts \
    | sort \
    | while IFS= read -r path; do sha256sum "$path"; done \
    | sha256sum \
    | cut -d ' ' -f 1
)
BASE_DATA_SHA256=$(
  find "$BASE_F_F" "$BASE_VP_F" -type f -print \
    | sort \
    | while IFS= read -r path; do sha256sum "$path"; done \
    | sha256sum \
    | cut -d ' ' -f 1
)
POLICY_RUN="$OUTPUT_ROOT/policy-only"
VALUE_RUN="$OUTPUT_ROOT/win-value"
SELECTION_JSON="$OUTPUT_ROOT/selection.json"
PAIRED_DIR="$OUTPUT_ROOT/paired-final-f"
FINAL_REPORT="$OUTPUT_ROOT/selected-final-fast.json"
STATUS_FILE="$OUTPUT_ROOT/status.txt"
RUN_IDENTITY_FILE="$OUTPUT_ROOT/run_identity.sha256"
CURRENT_RUN_IDENTITY=$(
  {
    echo "base_data_sha256=$BASE_DATA_SHA256"
    echo "dagger_manifest_sha256=$(sha256sum "$DAGGER_DATA/manifest.json" | cut -d ' ' -f 1)"
    echo "control_checkpoint_sha256=$(sha256sum "$CONTROL_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "embedding_dim=$EMBEDDING_DIM"
    echo "epochs=$BC_EPOCHS"
    echo "seed=$BC_SEED"
    echo "paired_games=$PAIRED_GAMES"
    echo "device=$BC_DEVICE"
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

write_run_record() {
  {
    echo "base_commit=$(git rev-parse HEAD)"
    echo "source_diff_sha256=$SOURCE_DIFF_SHA256"
    echo "source_status_sha256=$SOURCE_STATUS_SHA256"
    echo "source_untracked_sha256=$SOURCE_UNTRACKED_SHA256"
    echo "run_identity_sha256=$CURRENT_RUN_IDENTITY"
    echo "base_f_f=$BASE_F_F"
    echo "base_vp_f=$BASE_VP_F"
    echo "base_data_sha256=$BASE_DATA_SHA256"
    echo "dagger_manifest_sha256=$(sha256sum "$DAGGER_DATA/manifest.json" | cut -d ' ' -f 1)"
    echo "new_paired_cli_sha256=$(sha256sum examples/colonist_1v1_paired_evaluate.py | cut -d ' ' -f 1)"
    echo "launcher_sha256=$(sha256sum scripts/gpu/run_factored_dagger_comparison.sh | cut -d ' ' -f 1)"
    echo "control_checkpoint_sha256=$(sha256sum "$CONTROL_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "host=$(hostname -f)"
    echo "started=$(date -Iseconds)"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    echo "requested_device=$BC_DEVICE"
    "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")); print("mps=" + str(bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())))'
    echo "embedding_dim=$EMBEDDING_DIM"
    echo "epochs=$BC_EPOCHS"
    echo "seed=$BC_SEED"
    echo "paired_games=$PAIRED_GAMES"
  } >"$OUTPUT_ROOT/run_record.txt"
}

train_variant() {
  local name=$1
  local run_dir=$2
  local win_weight=$3
  local checkpoint="$run_dir/bc.pt"
  local metadata="$run_dir/bc.meta.json"
  if [[ -f "$checkpoint" && -f "$metadata" ]]; then
    echo "Reusing completed $name checkpoint: $checkpoint"
    return
  fi
  if [[ -d "$run_dir" && -n "$(find "$run_dir" -mindepth 1 -print -quit)" ]]; then
    write_status blocked "partial-$name"
    echo "Refusing to overwrite partial run: $run_dir" >&2
    exit 3
  fi

  "$PYTHON_BIN" examples/colonist_1v1_bc.py \
    --data-dir "$BASE_F_F" "$BASE_VP_F" \
    --augmentation-data-dir "$DAGGER_DATA" \
    --augmentation-weight 4 \
    --architecture factored_policy_value \
    --embedding-dim "$EMBEDDING_DIM" \
    --loss hybrid \
    --hybrid-listwise-weight 0.003 \
    --listwise-temperature 0.02 \
    --win-value-weight "$win_weight" \
    --vp-margin-weight 0 \
    --lr 0.001 \
    --epochs "$BC_EPOCHS" \
    --val-fraction 0.1 \
    --test-fraction 0.1 \
    --split-seed "$BC_SEED" \
    --seed "$BC_SEED" \
    --device "$BC_DEVICE" \
    --feature-profile raw \
    --out "$checkpoint" \
    --run-dir "$run_dir" \
    2>&1 | tee "$OUTPUT_ROOT/logs/$name.log"
}

select_variant() {
  "$PYTHON_BIN" - \
    "$POLICY_RUN/bc.meta.json" \
    "$VALUE_RUN/bc.meta.json" \
    "$SELECTION_JSON" <<'PY'
import json
import sys
from pathlib import Path

policy_path, value_path, output_path = map(Path, sys.argv[1:])
rows = []
for name, path in (("policy-only", policy_path), ("win-value", value_path)):
    data = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "name": name,
            "checkpoint": str(path.with_name("bc.pt")),
            "metadata": str(path),
            "selection_metric": data["selection_metric"],
            "selection_value": data["selection_value"],
            "val_metrics": data["val_metrics"],
            "test_metrics": data["test_metrics"],
            "parameter_count": data["parameter_count"],
            "win_value_weight": data["win_value_weight"],
        }
    )
if any(row["selection_metric"] != "mean_regret" for row in rows):
    raise ValueError("Both variants must select by held-out mean_regret")
selected = min(rows, key=lambda row: (row["selection_value"], row["name"]))
payload = {
    "selection_rule": "minimum validation mean_regret; lexical tie-break",
    "selected": selected["name"],
    "selected_checkpoint": selected["checkpoint"],
    "variants": rows,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(selected["name"])
PY
}

write_status preflight
"$PYTHON_BIN" examples/colonist_1v1_distill.py --output "$DAGGER_DATA" --verify
write_run_record

write_status training-policy-only
train_variant policy-only "$POLICY_RUN" 0
write_status training-win-value
train_variant win-value "$VALUE_RUN" 0.25

write_status selecting
SELECTED_VARIANT=$(select_variant)
if [[ "$SELECTED_VARIANT" == "policy-only" ]]; then
  SELECTED_CHECKPOINT="$POLICY_RUN/bc.pt"
else
  SELECTED_CHECKPOINT="$VALUE_RUN/bc.pt"
fi

write_status evaluating-paired-final-f "$SELECTED_VARIANT"
set +e
"$PYTHON_BIN" examples/colonist_1v1_paired_evaluate.py \
  --candidate "T:$SELECTED_CHECKPOINT" \
  --baseline "T:$CONTROL_CHECKPOINT" \
  --opponents F \
  --num-games "$PAIRED_GAMES" \
  --seed-suite final \
  --minimum-delta 0 \
  --output-dir "$PAIRED_DIR" \
  2>&1 | tee "$OUTPUT_ROOT/logs/paired-final-f.log"
PAIRED_GATE_RC=${PIPESTATUS[0]}

"$PYTHON_BIN" examples/colonist_1v1_evaluate.py \
  --agent "T:$SELECTED_CHECKPOINT" \
  --benchmark \
  --protocol fast \
  --num-games 50 \
  --gates \
  --eval-kind final \
  --seed-suite final \
  --gate-mode lower_bound \
  --checkpoint-label "factored-$SELECTED_VARIANT" \
  --report "$FINAL_REPORT" \
  2>&1 | tee "$OUTPUT_ROOT/logs/selected-final-fast.log"
ABSOLUTE_GATE_RC=${PIPESTATUS[0]}
set -e

{
  echo "selected_variant=$SELECTED_VARIANT"
  echo "selected_checkpoint=$SELECTED_CHECKPOINT"
  echo "paired_gate_exit=$PAIRED_GATE_RC"
  echo "absolute_gate_exit=$ABSOLUTE_GATE_RC"
} >>"$OUTPUT_ROOT/run_record.txt"
write_status complete "paired_gate=$PAIRED_GATE_RC absolute_gate=$ABSOLUTE_GATE_RC"
echo "Factored DAgger comparison complete: $OUTPUT_ROOT"
