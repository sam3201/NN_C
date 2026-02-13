#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Initializing SAM-D Production Launcher..."

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Create/activate venv
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d "venv" ]]; then
    echo "🧪 Creating venv..."
    python3 -m venv venv
  fi
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "🐍 Using Python: $(python -V)"
echo "📍 Python path: $(python -c 'import sys; print(sys.executable)')"

# Upgrade build tooling (inside venv)
python -m pip install -U pip setuptools wheel

# Install deps (inside venv)
echo "📦 Installing dependencies..."
python -m pip install -r requirements.txt

# Build C extensions (inside venv)
echo "🧠 Building C extensions..."
rm -rf build/
python setup.py build_ext --inplace

mkdir -p logs sam_data/backups

export PYTHONPATH="src/python:."
export SAM_PROFILE="${SAM_PROFILE:-full}"
export SAM_AUTONOMOUS_ENABLED=1
export SAM_STRICT_LOCAL_ONLY=1

PORT="${PORT:-5005}"

echo "========================================================"
echo "🤖 Starting SAM-D"
echo "📊 Dashboard: http://localhost:${PORT}"
echo "========================================================"

while true; do
  echo "🎯 Launching..."
  python src/python/complete_sam_unified.py --port "$PORT" || true
  echo "🔄 Restarting in 2s..."
  sleep 2
done

