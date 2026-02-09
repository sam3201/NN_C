#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
DATASET_EXPANDED="$ROOT/training/distilled/expanded_distill.jsonl"
LOG_DIR="$ROOT/logs"
OUT_DIR="$ROOT/training"

mkdir -p "$LOG_DIR"

run_job() {
  local tag="$1"
  local model="$2"
  shift 2
  local out="$OUT_DIR/output_lora_${tag}"
  local log="$LOG_DIR/training_${tag}.log"
  local stdout="$LOG_DIR/training_${tag}.stdout"

  echo "=== START ${tag} $(date) ===" | tee -a "$log"
  echo "model=${model}" | tee -a "$log"
  echo "output=${out}" | tee -a "$log"

  SAM_TRAIN_LOG_FILE="$log" "$PY" -m training.training_loop \
    --model "$model" \
    --dataset "$DATASET_EXPANDED" \
    --output "$out" \
    --max-seq-len 256 \
    --batch-size 1 \
    --grad-accum 16 \
    --epochs 1 \
    --lr 2e-5 \
    --lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --trust-remote-code \
    "$@" \
    >> "$stdout" 2>&1

  local status=$?
  echo "=== END ${tag} status=${status} $(date) ===" | tee -a "$log"
  return $status
}

# 1) Qwen2.5-7B (fp16, device_map auto, grad checkpointing)
run_job "qwen2.5_7b_fp16" "Qwen/Qwen2.5-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --low-cpu-mem-usage \
  --gradient-checkpointing

# 2) Qwen2.5-7B 4-bit
run_job "qwen2.5_7b_4bit" "Qwen/Qwen2.5-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --low-cpu-mem-usage \
  --gradient-checkpointing \
  --load-in-4bit

# 3) Qwen2.5-7B 8-bit
run_job "qwen2.5_7b_8bit" "Qwen/Qwen2.5-7B" \
  --torch-dtype float16 \
  --device-map auto \
  --low-cpu-mem-usage \
  --gradient-checkpointing \
  --load-in-8bit

# 4) Qwen2.5-3B
run_job "qwen2.5_3b_fp16" "Qwen/Qwen2.5-3B" \
  --torch-dtype float16 \
  --device-map auto \
  --low-cpu-mem-usage \
  --gradient-checkpointing

# 5) Qwen2.5-1.5B
run_job "qwen2.5_1.5b_fp16" "Qwen/Qwen2.5-1.5B" \
  --torch-dtype float16 \
  --device-map auto \
  --low-cpu-mem-usage \
  --gradient-checkpointing

echo "=== SWEEP COMPLETE $(date) ===" | tee -a "$LOG_DIR/training_sweep.log"
