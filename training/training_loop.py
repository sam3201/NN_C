from __future__ import annotations

import argparse
import json
import os
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
    args = parser.parse_args()

    if args.lora and not PEFT_AVAILABLE:
        raise SystemExit("peft is not installed. Install training requirements.")
    if not args.lora and not args.full:
        raise SystemExit("Specify --lora or --full")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_dataset(args.dataset, args.max_seq_len)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_seq_len)

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.lora:
        target_modules = args.target_modules or ["q_proj", "v_proj"]
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        logging_steps=10,
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

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    meta = {
        "model": args.model,
        "dataset": args.dataset,
        "output": args.output,
        "lora": args.lora,
        "epochs": args.epochs,
    }
    with open(os.path.join(args.output, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
