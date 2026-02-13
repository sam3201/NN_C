#!/bin/bash
# SAM-D Launcher
# Usage: ./run_sam.sh [profile]
# Profiles: full (default), experimental

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROFILE="${1:-full}"
export SAM_PROFILE="$PROFILE"

exec bash scripts/run_sam.sh
