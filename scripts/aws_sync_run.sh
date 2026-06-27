#!/usr/bin/env bash
# Sync a local run directory to/from S3 (requires AWS CLI + bucket env).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: aws_sync_run.sh <local_run_dir> [--pull] [--dry-run]

  upload (default):  local runs/<id>/  ->  s3://$CATANATRON_S3_BUCKET/runs/<id>/
  --pull:            download from S3 into the local run directory
  --dry-run:         pass through to aws s3 sync

Environment:
  CATANATRON_S3_BUCKET   (required) bucket name from storage stack
  AWS_DEFAULT_REGION     region of the bucket

Examples:
  CATANATRON_S3_BUCKET=my-bucket ./scripts/aws_sync_run.sh runs/ec2_proxy_500k
  CATANATRON_S3_BUCKET=my-bucket ./scripts/aws_sync_run.sh runs/ec2_proxy_500k --pull
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

LOCAL_RUN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift

PULL=0
DRY=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull) PULL=1 ;;
    --dry-run) DRY+=(--dryrun) ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

BUCKET="${CATANATRON_S3_BUCKET:-}"
if [[ -z "$BUCKET" ]]; then
  echo "ERROR: set CATANATRON_S3_BUCKET to your artifacts bucket name." >&2
  exit 1
fi

if [[ ! -d "$LOCAL_RUN" && "$PULL" -eq 0 ]]; then
  echo "ERROR: local run directory not found: $LOCAL_RUN" >&2
  exit 1
fi

RUN_ID="$(basename "$LOCAL_RUN")"
S3_URI="s3://${BUCKET}/runs/${RUN_ID}/"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI not found." >&2
  exit 1
fi

# Skip bulky or ephemeral paths when uploading from a live training tree.
EXCLUDES=(
  --exclude ".git/*"
  --exclude "__pycache__/*"
  --exclude "*.pyc"
  --exclude "tb_bc/*"
  --exclude "tb_ppo/*"
  --exclude "tensorboard/*"
)

mkdir -p "$LOCAL_RUN"

if [[ "$PULL" -eq 1 ]]; then
  echo "==> Download $S3_URI -> $LOCAL_RUN/"
  aws s3 sync "$S3_URI" "$LOCAL_RUN/" "${EXCLUDES[@]}" "${DRY[@]}"
else
  echo "==> Upload $LOCAL_RUN/ -> $S3_URI"
  aws s3 sync "$LOCAL_RUN/" "$S3_URI" "${EXCLUDES[@]}" "${DRY[@]}"
fi

echo "Done."
