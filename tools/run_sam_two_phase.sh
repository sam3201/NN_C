#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cd "$ROOT"
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
elif [ -f "venv/bin/activate" ]; then
  source "venv/bin/activate"
fi
set -a
if [ -f ".env.local" ]; then
  source ".env.local"
fi
set +a

export SAM_TWO_PHASE_BOOT=1
export SAM_META_ONLY_BOOT=0
export SAM_REQUIRE_META_AGENT=1
export SAM_AUTONOMOUS_ENABLED="${SAM_AUTONOMOUS_ENABLED:-1}"
export SAM_REQUIRE_SELF_MOD=1

export SAM_PROVIDER_AUTO_SWITCH=1
DEFAULT_HF_MODEL_DIR="${SAM_HF_MODEL_DIR:-$ROOT/training/output_lora_qwen2.5_1.5b_fp16_v2}"
export SAM_POLICY_PROVIDER_PRIMARY="${SAM_POLICY_PROVIDER_PRIMARY:-hf:Qwen/Qwen2.5-1.5B@${DEFAULT_HF_MODEL_DIR}}"
export SAM_POLICY_PROVIDER_FALLBACK="ollama:qwen2.5-coder:7b"
export SAM_TEACHER_POOL_PRIMARY="${SAM_TEACHER_POOL_PRIMARY:-hf:Qwen/Qwen2.5-1.5B@${DEFAULT_HF_MODEL_DIR}}"
export SAM_TEACHER_POOL_FALLBACK="ollama:mistral:latest"
export SAM_CHAT_PROVIDER="${SAM_CHAT_PROVIDER:-ollama:qwen2.5-coder:7b}"

export SAM_HF_DEVICE_MAP=cpu
export SAM_HF_DTYPE=float16
export SAM_HF_FORCE_GREEDY=1

PYTHONUNBUFFERED=1 python "$ROOT/complete_sam_unified.py"
