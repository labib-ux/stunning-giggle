#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN=""
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: Python 3.11+ is required but was not found in PATH."
  exit 1
fi

"$PYTHON_BIN" -c "import sys; v=sys.version_info; print(f'Using Python {sys.version.split()[0]}'); sys.exit(0 if v >= (3,11) else 1)" \
  || { echo "Error: Python 3.11+ is required."; exit 1; }

if [ ! -d "venv" ]; then
  "$PYTHON_BIN" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
else
  echo ".env already exists — skipping to protect your keys"
fi

mkdir -p ./models

echo "Setup complete. Next step: open .env and add your API keys."
