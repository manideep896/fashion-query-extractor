#!/usr/bin/env python3
"""
2_train.py — QLoRA fine-tuning of Qwen3-4B-Instruct with Unsloth.

Key training decisions:
  - Response-only training: loss computed ONLY on the JSON extraction output,
    not on the system prompt or user query. This focuses the gradient signal
    on what actually matters and converges faster.
  - QLoRA (r=32): trains ~32M adapter params instead of all 4.44B.
  - BF16: matches Qwen3's native dtype, stable on A100.
  - Cosine LR with warmup: smooth decay, prevents early overfitting.
  - Best checkpoint restored automatically via load_best_model_at_end.

Reads:   data/train.jsonl, data/eval.jsonl
Writes:  checkpoints/   (best checkpoint + final adapter)

Usage:
    python 2_train.py
"""

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unsloth import FastLanguageModel

from config import (
    CHECKPOINT_DIR, DATA_DIR,
    EVAL_STEPS, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, LOGGING_STEPS, LORA_ALPHA, LORA_DROPOUT,
    LORA_R, LORA_TARGET_MODULES, LR_SCHEDULER, MAX_SEQ_LENGTH,
    MODEL_NAME, NUM_EPOCHS, PER_DEVICE_BATCH_SIZE, SAVE_STEPS,
    SEED, WARMUP_RATIO, WEIGHT_DECAY,
)


def print_gpu_info():
    if not torch.cuda.is_available():
        print("  WARNING: No CUDA GPU detected. Training will be very slow.")
        return
    gpu   = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    print(f"  GPU   : {gpu.name}")
    print(f"  VRAM  : {total / 1e9:.1f} GB total  /  {free / 1e9:.1f} GB free")
    print(f"  Torch : {torch.__version__}   CUDA: {torch.version.cuda}")


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)

    print("=" * 60)
    print("  Fashion Extractor — QLoRA Training")
    print("=" * 60)
    print("\nGPU info:")
    print_gpu_info()

    # ── Verify data ────────────────────────────────────────────────────────────
    data_dir = Path(DATA_DIR)
    for split in ("train", "eval"):
        p = data_dir / f"{split}.jsonl"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}\n"
                "Run:  python 1_format_data.py --input data/raw.jsonl"
            )

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "eval":  str(data_dir / "eval.jsonl"),
        },
    )
    n_train = len(dataset["train"])
    n_eval  = len(dataset["eval"])
    print(f"  Train : {n_train:,}")
    print(f"  Eval  : {n_eval:,}")

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,              # auto → BF16 on Ampere+
    )

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    print("\nApplying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable : {trainable:,}  ({100 * trainable / total:.2f}%)")
    print(f"  Frozen    : {total - trainable:,}")

    # ── Response-only data collator ───────────────────────────────────────────
    # This is the single most important training quality decision:
    # Loss is computed ONLY on the assistant's JSON output, not on
    # the system prompt or user query+schema. Without this, the model
    # wastes gradient signal trying to "predict" the fixed system prompt.
    # With it, every gradient step directly improves extraction quality.
    #
    # In Qwen3's ChatML, the assistant turn always starts with:
    #   <|im_start|>assistant\n
    # Everything after that newline is the JSON we want the model to learn.
    response_template = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )
    print(f"\nResponse-only training enabled.")
    print(f"  Loss computed only on assistant output (JSON extraction).")
    print(f"  System prompt + user query/schema are masked from loss.")

    # ── Training plan ─────────────────────────────────────────────────────────
    eff_batch       = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = n_train // eff_batch
    total_steps     = steps_per_epoch * NUM_EPOCHS

    print(f"\nTraining plan:")
    print(f"  Effective batch size : {eff_batch}")
    print(f"  Steps / epoch        : {steps_per_epoch:,}")
    print(f"  Total steps          : {total_steps:,}")
    print(f"  Warmup steps         : {int(total_steps * WARMUP_RATIO)}")
    print(f"  Epochs               : {NUM_EPOCHS}")
    print(f"  Eval every           : {EVAL_STEPS} steps")

    # ── TrainingArguments ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,           # gradient clipping for stability
        optim="adamw_8bit",
        bf16=True,
        fp16=False,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        data_seed=SEED,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        report_to="none",            # set to "wandb" if you want loss curves
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        data_collator=data_collator,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    print("=" * 60)

    stats = trainer.train()

    m = stats.metrics
    print("\n" + "=" * 60)
    print("Training complete.")
    print(f"  Runtime           : {m.get('train_runtime', 0):.0f}s  "
          f"({m.get('train_runtime', 0) / 3600:.2f}h)")
    print(f"  Samples / second  : {m.get('train_samples_per_second', 0):.2f}")
    print(f"  Final train loss  : {m.get('train_loss', 0):.4f}")

    # ── Save final adapter ────────────────────────────────────────────────────
    adapter_path = Path(CHECKPOINT_DIR) / "final_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nFinal adapter saved → {adapter_path}")

    # ── Save training stats ───────────────────────────────────────────────────
    stats_path = Path(CHECKPOINT_DIR) / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"Training stats    → {stats_path}")

    print("\nNext step: python 3_evaluate.py")


if __name__ == "__main__":
    main()
