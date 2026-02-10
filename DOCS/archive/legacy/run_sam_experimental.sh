#!/usr/bin/env bash
set -euo pipefail
export SAM_PROFILE=experimental
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/run_sam_simple.sh"
