from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import List


def _require(module: str) -> None:
    try:
        __import__(module)
    except Exception as exc:
        raise SystemExit(
            f"Missing dependency '{module}'. Install training requirements before running. Error: {exc}"
        )


_require("torch")
_require("datasets")
_require("transformers")
_require("accelerate")

import torch  # type: ignore

from datasets import load_dataset  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model  # type: ignore
    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


def build_dataset(path: str, max_seq_len: int):
    ds = load_dataset("json", data_files=path, split="train")

    def concat(example):
        prompt = example.get("prompt", "")
        response = example.get("response", "")
        text = f"{prompt}\n{response}"
        return {"text": text}

    ds = ds.map(concat, remove_columns=ds.column_names)
    return ds


def _setup_logger(level: str, log_file: str | None) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("training")


def _count_params(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}

def _parse_max_memory(spec: str | None) -> dict | None:
    if not spec:
        return None
    spec = spec.strip()
    if not spec:
        return None
    if spec.startswith("{"):
        try:
            return json.loads(spec)
        except Exception as exc:
            raise SystemExit(f"Invalid --max-memory JSON: {exc}") from exc
    result = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit("Invalid --max-memory format. Use cpu=10GiB,mps=4GiB or JSON.")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Training loop scaffold (LoRA or full fine-tune)")
    parser.add_argument("--model", required=True, help="Base model name or path")
    parser.add_argument("--dataset", required=True, help="Distillation dataset JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="*", default=None)
    parser.add_argument("--log-level", default=os.getenv("SAM_TRAIN_LOG_LEVEL", "INFO"))
    parser.add_argument("--log-file", default=os.getenv("SAM_TRAIN_LOG_FILE"))
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--trust-remote-code", action="store_true",
                        default=os.getenv("SAM_TRAIN_TRUST_REMOTE_CODE", "0") == "1")
    parser.add_argument("--torch-dtype", default=os.getenv("SAM_TRAIN_TORCH_DTYPE", "float16"))
    parser.add_argument("--device-map", default=os.getenv("SAM_TRAIN_DEVICE_MAP", "auto"))
    parser.add_argument("--low-cpu-mem-usage", action="store_true",
                        default=os.getenv("SAM_TRAIN_LOW_CPU_MEM", "1") == "1")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        default=os.getenv("SAM_TRAIN_GRAD_CHECKPOINT", "1") == "1")
    parser.add_argument("--load-in-4bit", action="store_true",
                        default=os.getenv("SAM_TRAIN_LOAD_4BIT", "0") == "1")
    parser.add_argument("--load-in-8bit", action="store_true",
                        default=os.getenv("SAM_TRAIN_LOAD_8BIT", "0") == "1")
    parser.add_argument("--llm-int8-fp32-cpu-offload", action="store_true",
                        default=os.getenv("SAM_TRAIN_LLM_INT8_FP32_CPU_OFFLOAD", "0") == "1")
    parser.add_argument("--custom-device-map", type=str, default=None,
                        help="Custom device mapping JSON string or file path")
    parser.add_argument("--max-memory", type=str, default=os.getenv("SAM_TRAIN_MAX_MEMORY"),
                        help="Max memory per device (e.g. cpu=10GiB,mps=4GiB) or JSON string")
    parser.add_argument("--offload-dir", type=str, default=os.getenv("SAM_TRAIN_OFFLOAD_DIR"),
                        help="Folder for disk offload when max memory is exceeded")
    parser.add_argument("--offload-state-dict", action="store_true",
                        default=os.getenv("SAM_TRAIN_OFFLOAD_STATE_DICT", "0") == "1")
    args = parser.parse_args()

    logger = _setup_logger(args.log_level, args.log_file)

    if args.lora and not PEFT_AVAILABLE:
        raise SystemExit("peft is not installed. Install training requirements.")
    if not args.lora and not args.full:
        raise SystemExit("Specify --lora or --full")

    logger.info("Loading model=%s dataset=%s output=%s", args.model, args.dataset, args.output)
    logger.info("Mode: %s", "lora" if args.lora else "full")
    logger.info("max_seq_len=%d batch_size=%d grad_accum=%d epochs=%d lr=%s",
                args.max_seq_len, args.batch_size, args.grad_accum, args.epochs, args.lr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args.dataset, args.max_seq_len)
    logger.info("Dataset loaded: %d rows", len(dataset))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_len)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    dtype_name = args.torch_dtype.lower()
    if not hasattr(torch, dtype_name):
        raise SystemExit(f"Unknown torch dtype: {args.torch_dtype}")
    torch_dtype = getattr(torch, dtype_name)
    
    # Handle custom device mapping
    device_map = None
    if args.custom_device_map:
        try:
            # Try to parse as JSON string first
            if args.custom_device_map.startswith('{'):
                device_map = json.loads(args.custom_device_map)
            else:
                # Try to load as file path
                with open(args.custom_device_map, 'r') as f:
                    device_map = json.load(f)
            logger.info("Using custom device map: %s", device_map)
        except Exception as e:
            logger.warning("Failed to load custom device map: %s. Using default.", e)
            device_map = None if args.device_map.lower() in ("none", "null", "off") else args.device_map
    else:
        device_map = None if args.device_map.lower() in ("none", "null", "off") else args.device_map
    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("Choose only one: --load-in-4bit or --load-in-8bit")

    quant_config = None
    if args.load_in_4bit or args.load_in_8bit:
        if args.load_in_8bit and not args.llm_int8_fp32_cpu_offload:
            # Auto-enable CPU offload for int8 to allow CPU/disk dispatch.
            args.llm_int8_fp32_cpu_offload = True
            logger.info("Auto-enabled llm_int8_fp32_cpu_offload for 8-bit load.")
        try:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=bool(args.load_in_4bit),
                load_in_8bit=bool(args.load_in_8bit),
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=args.llm_int8_fp32_cpu_offload,
            )
            logger.info("LLM int8 FP32 CPU offload enabled: %s", args.llm_int8_fp32_cpu_offload)
        except Exception as exc:
            raise SystemExit(f"Quantized load requested but bitsandbytes is unavailable: {exc}")

    if args.load_in_8bit and device_map == "auto" and not args.custom_device_map:
        # Without CUDA, default to CPU placement to satisfy offload requirements.
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_cuda:
            device_map = "cpu" if not has_mps else "cpu"
            logger.info("Using device_map=%s for 8-bit load (no CUDA).", device_map)

    max_memory = _parse_max_memory(args.max_memory)
    if max_memory:
        logger.info("Using max_memory=%s", max_memory)
        if args.offload_dir:
            os.makedirs(args.offload_dir, exist_ok=True)
            logger.info("Using offload_dir=%s", args.offload_dir)

    logger.info(
        "torch_dtype=%s device_map=%s low_cpu_mem=%s grad_ckpt=%s load_in_4bit=%s load_in_8bit=%s llm_int8_fp32_cpu_offload=%s max_memory=%s offload_dir=%s",
        args.torch_dtype,
        device_map or "none",
        args.low_cpu_mem_usage,
        args.gradient_checkpointing,
        args.load_in_4bit,
        args.load_in_8bit,
        args.llm_int8_fp32_cpu_offload,
        max_memory or "none",
        args.offload_dir or "none",
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        quantization_config=quant_config,
        max_memory=max_memory,
        offload_folder=args.offload_dir,
        offload_state_dict=args.offload_state_dict,
    )

    if args.lora:
        target_modules = args.target_modules or ["q_proj", "v_proj"]
        logger.info("LoRA target_modules=%s r=%d alpha=%d dropout=%.3f",
                    ",".join(target_modules), args.lora_r, args.lora_alpha, args.lora_dropout)
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        except Exception as exc:
            logger.warning("Gradient checkpointing enable failed: %s", exc)

    param_counts = _count_params(model)
    logger.info("Params: total=%d trainable=%d", param_counts["total"], param_counts["trainable"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=bool(dtype_name == "float16"),
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    logger.info("Training complete in %.2fs", elapsed)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "output": args.output,
        "lora": args.lora,
        "epochs": args.epochs,
        "elapsed_s": elapsed,
    }
    with open(os.path.join(args.output, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
