#!/usr/bin/env bash
set -euo pipefail

# Repo root = directory containing this script
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source "venv/bin/activate"
fi

# Load env files (optional)
set -a
if [ -f ".env.local" ]; then
  source ".env.local"
fi
PROFILE_NAME="${SAM_PROFILE:-full}"
PROFILE_FILE="$ROOT/profiles/${PROFILE_NAME}.env"
if [ -f "$PROFILE_FILE" ]; then
  source "$PROFILE_FILE"
fi
set +a

exec "$ROOT/tools/run_sam_two_phase.sh"
