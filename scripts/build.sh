#!/bin/bash
set -euo pipefail

# SAM-D Build Script
# Enforces clean builds and handles C extensions

echo "🛠️ Starting SAM-D build process..."

# Use virtualenv python if available, otherwise fallback to system python
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

echo "🐍 Using python: $($PYTHON --version)"

# 1. Clean previous builds
echo "🧹 Cleaning old build artifacts..."
rm -rf build/
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# 2. Build C extensions
echo "🧩 Building C extensions..."
$PYTHON setup.py build_ext --inplace

# 3. Verify build
echo "🔍 Verifying C extensions..."
$PYTHON -c "import sam_sav_dual_system, sam_meta_controller_c, consciousness_algorithmic, orchestrator_and_agents; print('✅ All C extensions imported successfully')"

echo "✨ Build complete!"
