#!/usr/bin/env bash
set -euo pipefail

echo "=== SAM Bootstrap (macOS / Linux) ==="

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

# --- Check Python ---
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "❌ Python not found. Install Python 3.9+ first."
  exit 1
fi

# --- Create venv ---
if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  "$PYTHON" -m venv .venv
fi

# --- Activate venv ---
source .venv/bin/activate

echo "🐍 Using Python: $(which python)"

# --- Upgrade pip ---
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- Install dependencies ---
REQ="requirements.txt"
if [ -f "$REQ" ]; then
  echo "📚 Installing Python dependencies..."
  pip install -r "$REQ"
else
  echo "⚠️  requirements.txt not found — installing minimal deps"
  pip install requests requests-oauthlib numpy
fi

# --- Build C extensions ---
echo "🧩 Building C extensions..."
python setup.py build_ext --inplace

# --- Run SAM ---
PROFILE="${1:-full}"
echo "🚀 Starting SAM (profile: $PROFILE)"
python run_sam.py --profile "$PROFILE"

