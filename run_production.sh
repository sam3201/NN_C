#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Initializing SAM-D Production Launcher..."

VENV_DIR="venv"

die() { echo "❌ $*" >&2; exit 1; }

# Choose Python (prefer python3, fallback to python)
PY_BIN="${PY_BIN:-}"
if [[ -z "$PY_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then PY_BIN="python3"
  elif command -v python >/dev/null 2>&1; then PY_BIN="python"
  else die "No python found on PATH"; fi
fi

# Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  echo "🧪 Creating venv with: $PY_BIN"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

# Activate venv (POSIX vs Windows)
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # macOS/Linux
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  # Windows Git Bash
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  die "Could not find venv activation script"
fi

echo "🐍 Python: $(command -v python)"
python -V

# Guardrail: enforce supported python range from pyproject.toml intent
python - <<'PY'
import sys
maj, minor = sys.version_info[:2]
if not ((maj, minor) >= (3,10) and (maj, minor) < (3,14)):
    raise SystemExit(f"Unsupported Python {maj}.{minor}. Use Python >=3.10,<3.14")
print("✅ Python version OK")
PY

echo "⬆️  Upgrading pip tooling..."
python -m pip install -U pip setuptools wheel

echo "📦 Installing dependencies..."
python -m pip install -r requirements.txt

echo "🧠 Building C extensions..."
rm -rf build/ 2>/dev/null || true
python setup.py build_ext --inplace >/dev/null

mkdir -p logs sam_data/backups

export PYTHONPATH=src/python:.
export SAM_PROFILE=full
export SAM_AUTONOMOUS_ENABLED=1
export SAM_UNBOUNDED_MODE=1
export SAM_RESTART_ENABLED=1
export SAM_STRICT_LOCAL_ONLY=1
export SAM_HOT_RELOAD=1

echo "🎯 Launching..."
while true; do
  python src/python/complete_sam_unified.py --port 5005
  code=$?
  echo "⚠️  Exited ($code). Restarting..."
  sleep 3
done

