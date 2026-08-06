#!/usr/bin/env bash
# One-shot Mac dev setup: creates a venv and installs dependencies.
# Requires Python 3.11 or 3.12 (mediapipe does not yet support 3.13+).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in python3.11 python3.12 /opt/homebrew/bin/python3.11 /opt/homebrew/bin/python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find Python 3.11 or 3.12. Install one first, e.g.:"
  echo "  brew install python@3.11"
  exit 1
fi

echo "Using $($PYTHON_BIN --version) at $(command -v "$PYTHON_BIN")"

"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

mkdir -p data/reports

echo ""
echo "Setup complete. Activate the venv with:  source .venv/bin/activate"
echo "Then run:  python main.py"
echo "(macOS will prompt for camera access on first run — click Allow.)"
