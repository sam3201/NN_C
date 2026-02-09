#!/bin/bash
set -e
cd /Users/samueldasari/Personal/NN_C
source .venv/bin/activate
set -a
if [ -f .env.local ]; then
  source .env.local
fi
set +a
exec /Users/samueldasari/Personal/NN_C/tools/run_sam_two_phase.sh
