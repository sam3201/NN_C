#!/bin/bash
set -e
cd /Users/samueldasari/Personal/NN_C

if [ -d ".venv" ]; then
  source .venv/bin/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

PROFILE_NAME="${SAM_PROFILE:-full}"
PROFILE_FILE="profiles/${PROFILE_NAME}.env"

if [ -f ".env.local" ]; then
  set -a
  source .env.local
  set +a
fi
if [ -f "$PROFILE_FILE" ]; then
  set -a
  source "$PROFILE_FILE"
  set +a
fi

python3 -m py_compile complete_sam_unified.py
python3 -c "import sam_sav_dual_system, sam_meta_controller_c; print('C extensions import OK')"
python3 -c "from complete_sam_unified import UnifiedSAMSystem; print('System import OK')"

if [ -f training/tasks/default_tasks.jsonl ]; then
  python3 -m training.regression_suite \
    --tasks training/tasks/default_tasks.jsonl \
    --provider "${SAM_REGRESSION_PROVIDER:-ollama:mistral:latest}" \
    --min-pass "${SAM_REGRESSION_MIN_PASS:-0.7}" \
    --max-examples 5 \
    --output-json reports/regression_latest.json \
    --timeout "${SAM_REGRESSION_TIMEOUT_S:-120}"
fi

pytest -q || true
