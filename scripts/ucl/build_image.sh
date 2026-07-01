#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=${1:-"$HOME/Scratch/catanatron-1v1.sif"}

if [[ -e "$IMAGE" && "${FORCE:-0}" != "1" ]]; then
  echo "Image already exists: $IMAGE" >&2
  echo "Run with FORCE=1 to rebuild it." >&2
  exit 2
fi

module load apptainer

mkdir -p "$HOME/Scratch/.apptainer" "$(dirname "$IMAGE")"
export APPTAINER_CACHEDIR="$HOME/Scratch/.apptainer"
export APPTAINER_TMPDIR="${XDG_RUNTIME_DIR:-/tmp}/${USER}_apptainerbuild"
mkdir -p "$APPTAINER_TMPDIR"

TMP_IMAGE="${IMAGE}.building.${$}"
cd "$ROOT"
apptainer build --fakeroot "$TMP_IMAGE" scripts/ucl/catanatron-myriad.def
mv "$TMP_IMAGE" "$IMAGE"

apptainer exec "$IMAGE" python -c \
  'import sys, torch; print("python", sys.version.split()[0]); print("torch", torch.__version__)'
echo "Built $IMAGE"
