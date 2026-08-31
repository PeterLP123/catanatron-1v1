#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <student-bc.pt> <base-f-f-dir> <base-vp-f-dir> <output-root> <prior-iteration-dir>..." >&2
  exit 2
fi

STUDENT_CHECKPOINT=$1
BASE_F_F=$2
BASE_VP_F=$3
OUTPUT_ROOT=$4
shift 4
PRIOR_ITERATION_DIRS=("$@")

DAGGER_REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH="$DAGGER_REPO_ROOT/catanatron${PYTHONPATH:+:$PYTHONPATH}"
cd "$DAGGER_REPO_ROOT"

PYTHON_BIN=${PYTHON_BIN:-python}
DAGGER_ITERATION=${DAGGER_ITERATION:-${#PRIOR_ITERATION_DIRS[@]}}
DAGGER_GAMES=${DAGGER_GAMES:-100}
DAGGER_SEED=${DAGGER_SEED:-20260721}
DAGGER_SHARD_GAMES=${DAGGER_SHARD_GAMES:-10}
DAGGER_AUGMENTATION_WEIGHT=${DAGGER_AUGMENTATION_WEIGHT:-4}
BC_EPOCHS=${DAGGER_BC_EPOCHS:-10}
BC_SEED=${DAGGER_BC_SEED:-101}
BC_LR=${DAGGER_BC_LR:-0.001}
BC_INIT_CHECKPOINT=${DAGGER_BC_INIT_CHECKPOINT:-}
BC_DEVICE=${DAGGER_DEVICE:-cuda}
BC_HARD_STATES=${DAGGER_BC_HARD_STATES:-0}
BC_OUTCOME_WEIGHTED=${DAGGER_BC_OUTCOME_WEIGHTED:-0}
BC_OUTCOME_LOSS_BONUS=${DAGGER_BC_OUTCOME_LOSS_BONUS:-1.0}
BC_OUTCOME_VP_DEFICIT_BONUS=${DAGGER_BC_OUTCOME_VP_DEFICIT_BONUS:-0.5}
BC_OUTCOME_VP_DEFICIT_SCALE=${DAGGER_BC_OUTCOME_VP_DEFICIT_SCALE:-10.0}
PAIRED_GAMES=${DAGGER_PAIRED_GAMES:-200}
EVAL_SEED_ROUND=${DAGGER_EVAL_SEED_ROUND:-1}

if [[ "$DAGGER_ITERATION" -ne "${#PRIOR_ITERATION_DIRS[@]}" ]]; then
  echo "DAGGER_ITERATION must equal the number of prior iteration directories" >&2
  exit 2
fi
if [[ "$DAGGER_ITERATION" -lt 1 ]]; then
  echo "Use run_dagger_f_pilot.sh for iteration 0" >&2
  exit 2
fi
case "$BC_DEVICE" in
  auto|cpu|cuda|mps) ;;
  *)
    echo "DAGGER_DEVICE must be one of: auto, cpu, cuda, mps" >&2
    exit 2
    ;;
esac
if [[ "$BC_HARD_STATES" != 0 && "$BC_HARD_STATES" != 1 ]]; then
  echo "DAGGER_BC_HARD_STATES must be 0 or 1" >&2
  exit 2
fi
if [[ "$BC_OUTCOME_WEIGHTED" != 0 && "$BC_OUTCOME_WEIGHTED" != 1 ]]; then
  echo "DAGGER_BC_OUTCOME_WEIGHTED must be 0 or 1" >&2
  exit 2
fi
[[ -f "$STUDENT_CHECKPOINT" ]] || {
  echo "Required student checkpoint not found: $STUDENT_CHECKPOINT" >&2
  exit 2
}
if [[ -n "$BC_INIT_CHECKPOINT" && ! -f "$BC_INIT_CHECKPOINT" ]]; then
  echo "BC initialization checkpoint not found: $BC_INIT_CHECKPOINT" >&2
  exit 2
fi
for required_dir in "$BASE_F_F" "$BASE_VP_F" "${PRIOR_ITERATION_DIRS[@]}"; do
  [[ -d "$required_dir" ]] || {
    echo "Required directory not found: $required_dir" >&2
    exit 2
  }
done

DATA_ROOT=${DAGGER_CURRENT_DATA_ROOT:-$OUTPUT_ROOT/data}
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
printf -v ITERATION_NAME 'iteration-%04d' "$DAGGER_ITERATION"
CURRENT_ITERATION_DIR="$DATA_ROOT/$ITERATION_NAME"
BC_RUN="$OUTPUT_ROOT/bc"
BC_CHECKPOINT="$BC_RUN/bc.pt"
BC_META="$BC_RUN/bc.meta.json"
PAIRED_DIR="$OUTPUT_ROOT/paired-promotion-f"
OFFLINE_REPORT="$OUTPUT_ROOT/matched-holdout.json"
FINAL_REPORT="$OUTPUT_ROOT/final-fast.json"
STATUS_FILE="$OUTPUT_ROOT/status.txt"
RUN_IDENTITY_FILE="$OUTPUT_ROOT/run_identity.sha256"
CURRENT_RUN_IDENTITY=$(
  {
    echo "student_checkpoint_sha256=$(sha256sum "$STUDENT_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "base_f_f=$BASE_F_F"
    echo "base_vp_f=$BASE_VP_F"
    echo "base_data_sha256=$BASE_DATA_SHA256"
    echo "data_root=$DATA_ROOT"
    echo "iteration=$DAGGER_ITERATION"
    echo "games=$DAGGER_GAMES"
    echo "seed=$DAGGER_SEED"
    echo "shard_games=$DAGGER_SHARD_GAMES"
    echo "augmentation_weight=$DAGGER_AUGMENTATION_WEIGHT"
    echo "epochs=$BC_EPOCHS"
    echo "bc_seed=$BC_SEED"
    echo "bc_lr=$BC_LR"
    echo "bc_device=$BC_DEVICE"
    echo "bc_hard_states=$BC_HARD_STATES"
    echo "bc_outcome_weighted=$BC_OUTCOME_WEIGHTED"
    echo "bc_outcome_loss_bonus=$BC_OUTCOME_LOSS_BONUS"
    echo "bc_outcome_vp_deficit_bonus=$BC_OUTCOME_VP_DEFICIT_BONUS"
    echo "bc_outcome_vp_deficit_scale=$BC_OUTCOME_VP_DEFICIT_SCALE"
    echo "paired_games=$PAIRED_GAMES"
    echo "eval_seed_round=$EVAL_SEED_ROUND"
    echo "source_diff_sha256=$SOURCE_DIFF_SHA256"
    echo "source_status_sha256=$SOURCE_STATUS_SHA256"
    echo "source_untracked_sha256=$SOURCE_UNTRACKED_SHA256"
    if [[ -n "$BC_INIT_CHECKPOINT" ]]; then
      echo "bc_init_checkpoint_sha256=$(sha256sum "$BC_INIT_CHECKPOINT" | cut -d ' ' -f 1)"
    fi
    for iteration_dir in "${PRIOR_ITERATION_DIRS[@]}"; do
      echo "prior_manifest_sha256=$(sha256sum "$iteration_dir/manifest.json" | cut -d ' ' -f 1) path=$iteration_dir/manifest.json"
    done
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

verify_prior_iterations() {
  local expected_index=0
  local iteration_dir
  local expected_name
  local data_root
  for iteration_dir in "${PRIOR_ITERATION_DIRS[@]}"; do
    printf -v expected_name 'iteration-%04d' "$expected_index"
    if [[ "$(basename "$iteration_dir")" != "$expected_name" ]]; then
      write_status blocked "prior-order-$expected_index"
      echo "Expected prior iteration $expected_name, got $iteration_dir" >&2
      exit 3
    fi
    [[ -f "$iteration_dir/manifest.json" ]] || {
      write_status blocked "missing-prior-manifest-$expected_index"
      echo "Missing immutable iteration manifest: $iteration_dir/manifest.json" >&2
      exit 3
    }
    data_root=$(dirname "$iteration_dir")
    "$PYTHON_BIN" examples/colonist_1v1_distill.py \
      --output "$data_root" --verify
    expected_index=$((expected_index + 1))
  done
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
    echo "bc_cli_sha256=$(sha256sum examples/colonist_1v1_bc.py | cut -d ' ' -f 1)"
    echo "bc_training_sha256=$(sha256sum catanatron/catanatron/gym/bc_training.py | cut -d ' ' -f 1)"
    echo "colonist_training_sha256=$(sha256sum catanatron/catanatron/gym/colonist_training.py | cut -d ' ' -f 1)"
    echo "bc_compare_cli_sha256=$(sha256sum examples/colonist_1v1_bc_compare.py | cut -d ' ' -f 1)"
    echo "paired_cli_sha256=$(sha256sum examples/colonist_1v1_paired_evaluate.py | cut -d ' ' -f 1)"
    echo "launcher_sha256=$(sha256sum scripts/gpu/run_dagger_f_iteration.sh | cut -d ' ' -f 1)"
    echo "student_checkpoint_sha256=$(sha256sum "$STUDENT_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "current_data_root=$DATA_ROOT"
    if [[ -f "$CURRENT_ITERATION_DIR/manifest.json" ]]; then
      echo "current_manifest_sha256_preflight=$(sha256sum "$CURRENT_ITERATION_DIR/manifest.json" | cut -d ' ' -f 1)"
    fi
    local iteration_dir
    for iteration_dir in "${PRIOR_ITERATION_DIRS[@]}"; do
      echo "prior_manifest_sha256=$(sha256sum "$iteration_dir/manifest.json" | cut -d ' ' -f 1) path=$iteration_dir/manifest.json"
    done
    echo "host=$(hostname -f)"
    echo "started=$(date -Iseconds)"
    echo "python=$($PYTHON_BIN --version 2>&1)"
    echo "requested_device=$BC_DEVICE"
    "$PYTHON_BIN" -c 'import torch; print("torch=" + torch.__version__); print("cuda=" + str(torch.cuda.is_available())); print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")); print("mps=" + str(bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())))'
    echo "iteration=$DAGGER_ITERATION"
    echo "collection_games=$DAGGER_GAMES"
    echo "collection_seed=$DAGGER_SEED"
    echo "augmentation_weight=$DAGGER_AUGMENTATION_WEIGHT"
    echo "epochs=$BC_EPOCHS"
    echo "bc_seed=$BC_SEED"
    echo "bc_lr=$BC_LR"
    echo "bc_hard_states=$BC_HARD_STATES"
    echo "bc_outcome_weighted=$BC_OUTCOME_WEIGHTED"
    echo "bc_outcome_loss_bonus=$BC_OUTCOME_LOSS_BONUS"
    echo "bc_outcome_vp_deficit_bonus=$BC_OUTCOME_VP_DEFICIT_BONUS"
    echo "bc_outcome_vp_deficit_scale=$BC_OUTCOME_VP_DEFICIT_SCALE"
    echo "bc_init_checkpoint=${BC_INIT_CHECKPOINT:-none}"
    if [[ -n "$BC_INIT_CHECKPOINT" ]]; then
      echo "bc_init_checkpoint_sha256=$(sha256sum "$BC_INIT_CHECKPOINT" | cut -d ' ' -f 1)"
    fi
    echo "paired_games=$PAIRED_GAMES"
    echo "eval_seed_round=$EVAL_SEED_ROUND"
  } >"$OUTPUT_ROOT/run_record.txt"
}

write_status preflight
verify_prior_iterations
write_run_record

write_status collecting "$ITERATION_NAME"
if [[ -f "$CURRENT_ITERATION_DIR/manifest.json" ]]; then
  "$PYTHON_BIN" examples/colonist_1v1_distill.py --output "$DATA_ROOT" --verify
  echo "Reusing verified DAgger iteration: $CURRENT_ITERATION_DIR"
elif [[ -d "$DATA_ROOT" && -n "$(find "$DATA_ROOT" -mindepth 1 -print -quit)" ]]; then
  write_status blocked partial-data
  echo "Refusing to overwrite partial immutable DAgger data: $DATA_ROOT" >&2
  exit 3
else
  "$PYTHON_BIN" examples/colonist_1v1_distill.py \
    --student "T:$STUDENT_CHECKPOINT" \
    --teacher F \
    --opponent F \
    --iteration "$DAGGER_ITERATION" \
    --games "$DAGGER_GAMES" \
    --seed "$DAGGER_SEED" \
    --shard-games "$DAGGER_SHARD_GAMES" \
    --feature-profile raw \
    --output "$DATA_ROOT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/distill.log"
  "$PYTHON_BIN" examples/colonist_1v1_distill.py --output "$DATA_ROOT" --verify
fi

write_status training hybrid-bc
if [[ -f "$BC_CHECKPOINT" && -f "$BC_META" ]]; then
  echo "Reusing completed BC checkpoint: $BC_CHECKPOINT"
elif [[ -d "$BC_RUN" && -n "$(find "$BC_RUN" -mindepth 1 -print -quit)" ]]; then
  write_status blocked partial-bc
  echo "Refusing to overwrite partial BC run: $BC_RUN" >&2
  exit 3
else
  BC_AUGMENTATION_ARGS=(
    --augmentation-data-dir "${PRIOR_ITERATION_DIRS[@]}" "$CURRENT_ITERATION_DIR"
  )
  if [[ "$BC_OUTCOME_WEIGHTED" == 1 ]]; then
    BC_AUGMENTATION_ARGS=(
      --augmentation-data-dir "${PRIOR_ITERATION_DIRS[@]}"
      --outcome-weighted-augmentation-data-dir "$CURRENT_ITERATION_DIR"
      --outcome-loss-bonus "$BC_OUTCOME_LOSS_BONUS"
      --outcome-vp-deficit-bonus "$BC_OUTCOME_VP_DEFICIT_BONUS"
      --outcome-vp-deficit-scale "$BC_OUTCOME_VP_DEFICIT_SCALE"
    )
  fi
  BC_COMMAND=(
    "$PYTHON_BIN" examples/colonist_1v1_bc.py
    --data-dir "$BASE_F_F" "$BASE_VP_F"
    "${BC_AUGMENTATION_ARGS[@]}"
    --augmentation-weight "$DAGGER_AUGMENTATION_WEIGHT"
    --architecture mlp
    --hidden 512 512
    --loss hybrid
    --hybrid-listwise-weight 0.003
    --listwise-temperature 0.02
    --win-value-weight 0
    --vp-margin-weight 0
    --lr "$BC_LR"
    --epochs "$BC_EPOCHS"
    --val-fraction 0.1
    --test-fraction 0.1
    --split-seed "$BC_SEED"
    --seed "$BC_SEED"
    --device "$BC_DEVICE"
    --feature-profile raw
    --out "$BC_CHECKPOINT"
    --run-dir "$BC_RUN"
  )
  if [[ "$BC_HARD_STATES" == 1 ]]; then
    BC_COMMAND+=(--hard-states)
  fi
  if [[ -n "$BC_INIT_CHECKPOINT" ]]; then
    BC_COMMAND+=(--init-checkpoint "$BC_INIT_CHECKPOINT")
  fi
  "${BC_COMMAND[@]}" 2>&1 | tee "$OUTPUT_ROOT/logs/bc.log"
fi

write_status evaluating-offline-retention "$ITERATION_NAME"
if [[ -f "$OFFLINE_REPORT" ]]; then
  write_status blocked existing-offline-report
  echo "Refusing to reuse evaluation evidence; choose a fresh output root: $OFFLINE_REPORT" >&2
  exit 3
else
  if ! "$PYTHON_BIN" examples/colonist_1v1_bc_compare.py \
    --candidate "$BC_CHECKPOINT" \
    --baseline "$STUDENT_CHECKPOINT" \
    --data-dir "$BASE_F_F" "$BASE_VP_F" \
    --augmentation-data-dir "${PRIOR_ITERATION_DIRS[@]}" "$CURRENT_ITERATION_DIR" \
    --val-fraction 0.1 \
    --test-fraction 0.1 \
    --split-seed "$BC_SEED" \
    --batch-size 4096 \
    --device "$BC_DEVICE" \
    --output "$OFFLINE_REPORT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/matched-holdout.log"; then
    write_status blocked offline-retention-evaluation
    exit 3
  fi
fi
OFFLINE_GATE_RC=$(
  "$PYTHON_BIN" -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
val_delta = report["deltas"]["val"]["mean_regret"]
test_delta = report["deltas"]["test"]["mean_regret"]
print(0 if val_delta < 0 and test_delta <= 0 else 1)
' "$OFFLINE_REPORT"
)
if [[ "$OFFLINE_GATE_RC" -ne 0 ]]; then
  {
    echo "candidate_checkpoint_sha256=$(sha256sum "$BC_CHECKPOINT" | cut -d ' ' -f 1)"
    echo "candidate_metadata_sha256=$(sha256sum "$BC_META" | cut -d ' ' -f 1)"
    echo "current_manifest_sha256=$(sha256sum "$CURRENT_ITERATION_DIR/manifest.json" | cut -d ' ' -f 1)"
    echo "offline_report_sha256=$(sha256sum "$OFFLINE_REPORT" | cut -d ' ' -f 1)"
    echo "offline_gate_exit=$OFFLINE_GATE_RC"
    echo "paired_gate_exit=skipped"
    echo "absolute_gate_exit=skipped"
  } >>"$OUTPUT_ROOT/run_record.txt"
  write_status complete "offline_gate=$OFFLINE_GATE_RC paired_gate=skipped absolute_gate=skipped"
  echo "DAgger F iteration $DAGGER_ITERATION stopped at offline retention gate: $OUTPUT_ROOT"
  exit 0
fi

write_status evaluating-paired "$ITERATION_NAME"
if [[ -f "$PAIRED_DIR/paired_comparison.json" \
  && -f "$PAIRED_DIR/candidate_report.json" \
  && -f "$PAIRED_DIR/baseline_report.json" ]]; then
  write_status blocked existing-paired-evidence
  echo "Refusing to reuse evaluation evidence; choose a fresh output root: $PAIRED_DIR" >&2
  exit 3
elif [[ -d "$PAIRED_DIR" && -n "$(find "$PAIRED_DIR" -mindepth 1 -print -quit)" ]]; then
  write_status blocked partial-paired-evaluation
  echo "Refusing to overwrite partial paired evaluation: $PAIRED_DIR" >&2
  exit 3
else
  set +e
  "$PYTHON_BIN" examples/colonist_1v1_paired_evaluate.py \
    --candidate "T:$BC_CHECKPOINT" \
    --baseline "T:$STUDENT_CHECKPOINT" \
    --opponents F \
    --num-games "$PAIRED_GAMES" \
    --seed-suite promotion \
    --seed-round "$EVAL_SEED_ROUND" \
    --minimum-delta 0 \
    --output-dir "$PAIRED_DIR" \
    2>&1 | tee "$OUTPUT_ROOT/logs/paired-promotion-f.log"
  PAIRED_GATE_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$PAIRED_GATE_RC" -gt 1 ]]; then
    write_status blocked "paired-evaluation-rc-$PAIRED_GATE_RC"
    exit "$PAIRED_GATE_RC"
  fi
  if [[ ! -f "$PAIRED_DIR/paired_comparison.json" \
    || ! -f "$PAIRED_DIR/candidate_report.json" \
    || ! -f "$PAIRED_DIR/baseline_report.json" ]]; then
    write_status blocked missing-paired-evidence
    exit 3
  fi
fi
PAIRED_GATE_RC=${PAIRED_GATE_RC:-$(
  "$PYTHON_BIN" -c 'import json,sys; print(0 if json.load(open(sys.argv[1]))["all_gates_passed"] else 1)' \
    "$PAIRED_DIR/paired_comparison.json"
)}

write_status evaluating-final-fast "$ITERATION_NAME"
if [[ -f "$FINAL_REPORT" ]]; then
  write_status blocked existing-final-report
  echo "Refusing to reuse evaluation evidence; choose a fresh output root: $FINAL_REPORT" >&2
  exit 3
else
  set +e
  "$PYTHON_BIN" examples/colonist_1v1_evaluate.py \
    --agent "T:$BC_CHECKPOINT" \
    --benchmark \
    --protocol fast \
    --num-games 50 \
    --gates \
    --eval-kind final \
    --seed-suite final \
    --seed-round "$EVAL_SEED_ROUND" \
    --gate-mode lower_bound \
    --checkpoint-label "dagger-f-iteration-$DAGGER_ITERATION" \
    --report "$FINAL_REPORT" \
    2>&1 | tee "$OUTPUT_ROOT/logs/final-fast.log"
  ABSOLUTE_GATE_RC=${PIPESTATUS[0]}
  set -e
  if [[ "$ABSOLUTE_GATE_RC" -gt 1 ]]; then
    write_status blocked "final-evaluation-rc-$ABSOLUTE_GATE_RC"
    exit "$ABSOLUTE_GATE_RC"
  fi
  if [[ ! -f "$FINAL_REPORT" ]]; then
    write_status blocked missing-final-evidence
    exit 3
  fi
fi
ABSOLUTE_GATE_RC=${ABSOLUTE_GATE_RC:-$(
  "$PYTHON_BIN" -c 'import json,sys; print(0 if json.load(open(sys.argv[1]))["all_gates_passed"] else 1)' \
    "$FINAL_REPORT"
)}

{
  echo "candidate_checkpoint_sha256=$(sha256sum "$BC_CHECKPOINT" | cut -d ' ' -f 1)"
  echo "candidate_metadata_sha256=$(sha256sum "$BC_META" | cut -d ' ' -f 1)"
  echo "current_manifest_sha256=$(sha256sum "$CURRENT_ITERATION_DIR/manifest.json" | cut -d ' ' -f 1)"
  echo "offline_report_sha256=$(sha256sum "$OFFLINE_REPORT" | cut -d ' ' -f 1)"
  echo "offline_gate_exit=$OFFLINE_GATE_RC"
  echo "paired_gate_exit=$PAIRED_GATE_RC"
  echo "absolute_gate_exit=$ABSOLUTE_GATE_RC"
} >>"$OUTPUT_ROOT/run_record.txt"
write_status complete "offline_gate=$OFFLINE_GATE_RC paired_gate=$PAIRED_GATE_RC absolute_gate=$ABSOLUTE_GATE_RC"
echo "DAgger F iteration $DAGGER_ITERATION complete: $OUTPUT_ROOT"
