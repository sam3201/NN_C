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

from datasets import load_dataset  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
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

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

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
        fp16=False,
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
