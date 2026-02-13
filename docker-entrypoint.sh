#!/usr/bin/env bash
set -euo pipefail

echo "🐳 SAM-D container starting..."
echo "🐍 $(python -V)"
echo "📦 $(python -m pip -V)"

# Build extensions on startup if you mount source (optional safety)
if [[ "${SAM_DOCKER_REBUILD_EXT:-0}" == "1" ]]; then
  echo "🧠 Rebuilding C extensions (SAM_DOCKER_REBUILD_EXT=1)..."
  python setup.py build_ext --inplace >/dev/null
fi

exec "$@"

