#!/bin/bash
set -e

# SAM-D Production Launcher (with Auto-Restart & Hot-Reload Support)
# Automates environment setup, compilation, and launch.

echo "🚀 Initializing SAM-D (ΨΔ•Ω-Core) Production Launcher..."

# 1. Environment Check
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "⚠️  No virtual environment detected."
    if [ -d "venv" ]; then
        echo "   Activating existing venv..."
        source venv/bin/activate
    else
        echo "   Creating new virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
    fi
fi

# 2. Dependency Check
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# 3. C-Core Compilation (Critical)
echo "🧠 Compiling C-Core (God Equation Regulator & Agents)..."
# Clean previous builds to ensure freshness
rm -rf build/
# Build in-place
python3 setup.py build_ext --inplace > /dev/null

# 4. Directory Setup
mkdir -p logs sam_data/backups

# 5. Continuous Launch Loop
echo "========================================================"
echo "🤖 SAM-D is starting in AUTONOMOUS MODE."
echo "🔥 Hot-Reload & Git-Push enabled (SAM_HOT_RELOAD=1)."
echo "📊 Dashboard: http://localhost:5005"
echo "📜 Logs:      tail -f logs/sam_runtime.log"
echo "========================================================"

# Set critical environment variables
export PYTHONPATH=src/python:.
export SAM_PROFILE=full
export SAM_AUTONOMOUS_ENABLED=1
export SAM_UNBOUNDED_MODE=1
export SAM_RESTART_ENABLED=1
export SAM_STRICT_LOCAL_ONLY=1 
export SAM_HOT_RELOAD=1

# Restart loop
while true; do
    echo "🎯 Launching system..."
    # We don't use nohup here inside the loop because the loop itself should be backgrounded if needed.
    # But for ease of use, we'll let the user run this script in foreground or background.
    python3 src/python/complete_sam_unified.py --port 5005
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "🔄 System requested restart (hot-reload). Restarting in 2s..."
    else
        echo "⚠️  System exited with code $EXIT_CODE. Restarting in 5s..."
        sleep 3
    fi
    sleep 2
done
