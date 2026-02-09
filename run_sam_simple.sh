#!/bin/bash
set -e
cd /Users/samueldasari/Personal/NN_C
source .venv/bin/activate
set -a
if [ -f .env.local ]; then
  source .env.local
fi
set +a
PROFILE_NAME="${SAM_PROFILE:-full}"
PROFILE_FILE="/Users/samueldasari/Personal/NN_C/profiles/${PROFILE_NAME}.env"
if [ -f "$PROFILE_FILE" ]; then
  set -a
  source "$PROFILE_FILE"
  set +a
fi
exec /Users/samueldasari/Personal/NN_C/tools/run_sam_two_phase.sh
