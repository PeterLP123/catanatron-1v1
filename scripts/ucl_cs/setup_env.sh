#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
VENV=${VENV:-"$HOME/.venvs/catanatron-1v1"}
PYTHON=${PYTHON:-}
export PIP_NO_CACHE_DIR=${PIP_NO_CACHE_DIR:-1}

if [[ -z "$PYTHON" ]]; then
  for candidate in python3.13 python3.12 python3.11; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON=$(command -v "$candidate")
      break
    fi
  done
fi

if [[ -z "$PYTHON" && -d /opt/Python ]]; then
  PYTHON=$(find /opt/Python -path '*/bin/python3.11' -perm -u+x 2>/dev/null | sort -V | tail -1)
fi

if [[ -z "$PYTHON" ]]; then
  echo "Python 3.11 or newer was not found." >&2
  echo "Inspect /opt/Python on the CS host or ask TSG which Python 3.11 setup is supported." >&2
  exit 2
fi

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
mkdir -p "$(dirname "$VENV")"
"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
  -c "$ROOT/requirements/training-constraints.txt" \
  -e "$ROOT[dev,gym,colonist,tui]"

python - <<'PY'
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY

echo "Environment ready: $VENV"
