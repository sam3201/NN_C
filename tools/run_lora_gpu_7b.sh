#!/usr/bin/env bash
set -euo pipefail

# Run 7B LoRA on a CUDA machine. Requires: CUDA + bitsandbytes + transformers + peft.
# Usage:
#   HF_TOKEN=[REDACTED] ./tools/run_lora_gpu_7b.sh

BASE_MODEL="Qwen/Qwen2.5-7B"
DATASET="training/distilled/merged_distill_v2.jsonl"
OUTDIR="training/output_lora_qwen2.5_7b_4bit_gpu"
LOGFILE="training/logs/lora_qwen2.5_7b_4bit_gpu.log"

mkdir -p training/logs

export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
if [[ -z "${HF_TOKEN}" ]]; then
  echo "HF_TOKEN is required for fast downloads." >&2
  exit 1
fi

python -m training.training_loop \
  --model "${BASE_MODEL}" \
  --dataset "${DATASET}" \
  --output "${OUTDIR}" \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 4 \
  --lr 1e-4 \
  --max-seq-len 1024 \
  --logging-steps 5 \
  --torch-dtype float16 \
  --device-map auto \
  --low-cpu-mem-usage \
  --gradient-checkpointing \
  --load-in-4bit \
  --lora \
  --trust-remote-code \
  --log-file "${LOGFILE}"

echo "GPU LoRA run complete: ${OUTDIR}"
