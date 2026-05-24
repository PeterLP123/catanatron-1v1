#!/usr/bin/env bash
# Colonist 1v1 training — EC2 bootstrap (Ubuntu 22.04/24.04)
# Run on the instance: bash scripts/ec2_setup.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/catanatron}"
PYTHON="${PYTHON:-python3.12}"

echo "==> System packages"
sudo apt-get update -qq
sudo apt-get install -y \
  git tmux \
  "${PYTHON}" "${PYTHON}-venv" "${PYTHON}-dev" \
  build-essential

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "ERROR: Clone repo first, e.g.:"
  echo "  git clone git@github.com:PeterLP123/catanatron-main.git $REPO_DIR"
  exit 1
fi

cd "$REPO_DIR"
echo "==> Virtualenv ($PYTHON)"
rm -rf .venv
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "==> catanatron + training deps (no pygame — not needed for RL)"
pip install -e .
pip install "gymnasium<=0.29.1" numpy pandas fastparquet pyarrow rich textual tensorboard
pip install torch stable-baselines3 sb3-contrib

echo "==> Verify"
python -c "import catanatron.gym; from sb3_contrib import MaskablePPO; print('imports ok')"
catanatron-play --help >/dev/null && echo "catanatron-play ok"

echo ""
echo "Setup complete. Activate with:"
echo "  source $REPO_DIR/.venv/bin/activate"
echo "  cd $REPO_DIR"
