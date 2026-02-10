#!/bin/bash
set -e
cd /Users/samueldasari/Personal/NN_C
source .venv/bin/activate
set -a
source .env.local
set +a

export SAM_TWO_PHASE_BOOT=1
export SAM_META_ONLY_BOOT=0
export SAM_REQUIRE_META_AGENT=1
export SAM_AUTONOMOUS_ENABLED="${SAM_AUTONOMOUS_ENABLED:-1}"
export SAM_REQUIRE_SELF_MOD=1

export SAM_PROVIDER_AUTO_SWITCH=1
export SAM_POLICY_PROVIDER_PRIMARY="hf:Qwen/Qwen2.5-1.5B@/Users/samueldasari/Personal/NN_C/training/output_lora_qwen2.5_1.5b_fp16_v2"
export SAM_POLICY_PROVIDER_FALLBACK="ollama:qwen2.5-coder:7b"
export SAM_TEACHER_POOL_PRIMARY="hf:Qwen/Qwen2.5-1.5B@/Users/samueldasari/Personal/NN_C/training/output_lora_qwen2.5_1.5b_fp16_v2"
export SAM_TEACHER_POOL_FALLBACK="ollama:mistral:latest"
export SAM_CHAT_PROVIDER="${SAM_CHAT_PROVIDER:-ollama:qwen2.5-coder:7b}"

export SAM_HF_DEVICE_MAP=cpu
export SAM_HF_DTYPE=float16
export SAM_HF_FORCE_GREEDY=1

PYTHONUNBUFFERED=1 python /Users/samueldasari/Personal/NN_C/complete_sam_unified.py
