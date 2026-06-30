#!/usr/bin/env python3
"""
4_export_push.py — Merge LoRA adapters, export GGUF, push to HuggingFace.

Outputs:
  merged_model/   — merged FP16 model (ready to load with transformers)
  gguf/           — Q4_K_M and Q8_0 quantized GGUF files
  adapter_only/   — lightweight LoRA-only weights (for fine-tuners)

Usage:
    python 4_export_push.py --hf_token hf_xxx
    python 4_export_push.py --hf_token hf_xxx --skip_gguf
    python 4_export_push.py --local_only              # don't push to HuggingFace
    HF_TOKEN=hf_xxx python 4_export_push.py           # token from env var
"""

import argparse
import os
from pathlib import Path

from unsloth import FastLanguageModel

from config import (
    CHECKPOINT_DIR, GGUF_DIR, HF_GGUF_ID, HF_MODEL_ID,
    MAX_SEQ_LENGTH, MERGED_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--model_path",
        default=str(Path(CHECKPOINT_DIR) / "final_adapter"),
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--skip_gguf", action="store_true",
        help="Skip GGUF export and upload",
    )
    parser.add_argument(
        "--local_only", action="store_true",
        help="Save files locally only; skip all HuggingFace uploads",
    )
    args = parser.parse_args()

    if not args.local_only and not args.hf_token:
        raise ValueError(
            "HuggingFace token is required for upload.\n"
            "Pass --hf_token hf_xxx  OR  export HF_TOKEN=hf_xxx  OR  use --local_only"
        )

    adapter_path = Path(args.model_path)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}\n"
            "Run 2_train.py first."
        )

    print("=" * 60)
    print("  Fashion Extractor — Export & Push")
    print("=" * 60)
    print(f"\n  Adapter  : {adapter_path}")
    print(f"  Merged   : {MERGED_DIR}")
    print(f"  GGUF     : {GGUF_DIR}")
    print(f"  HF model : {HF_MODEL_ID}")
    print(f"  HF GGUF  : {HF_GGUF_ID}")

    # ── Load adapter ──────────────────────────────────────────────────────────
    print(f"\nLoading adapter from: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Merged FP16
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Merging LoRA → FP16...")
    merged_dir = Path(MERGED_DIR)
    merged_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_merged(
        str(merged_dir),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"  Saved → {merged_dir}")

    if not args.local_only:
        print(f"  Uploading to HuggingFace: {HF_MODEL_ID}...")
        model.push_to_hub_merged(
            HF_MODEL_ID,
            tokenizer,
            save_method="merged_16bit",
            token=args.hf_token,
        )
        print(f"  Live at: https://huggingface.co/{HF_MODEL_ID}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: GGUF
    # ─────────────────────────────────────────────────────────────────────────
    if not args.skip_gguf:
        print("\n[2/3] Exporting GGUF (Q4_K_M + Q8_0)...")
        gguf_dir = Path(GGUF_DIR)
        gguf_dir.mkdir(parents=True, exist_ok=True)

        for quant in ("q4_k_m", "q8_0"):
            print(f"  Quantizing {quant}...")
            model.save_pretrained_gguf(str(gguf_dir), tokenizer, quantization_method=quant)

        print(f"  Saved → {gguf_dir}")

        if not args.local_only:
            print(f"  Uploading GGUF to HuggingFace: {HF_GGUF_ID}...")
            model.push_to_hub_gguf(
                HF_GGUF_ID,
                tokenizer,
                quantization_method=["q4_k_m", "q8_0"],
                token=args.hf_token,
            )
            print(f"  Live at: https://huggingface.co/{HF_GGUF_ID}")
    else:
        print("\n[2/3] GGUF skipped (--skip_gguf)")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Adapter-only weights (lightweight, for people who want to fine-tune further)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Saving LoRA adapter-only weights...")
    adapter_out = Path(MERGED_DIR).parent / "adapter_only"
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    print(f"  Saved → {adapter_out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Export complete.")
    if not args.local_only:
        print(f"\n  FP16 model : https://huggingface.co/{HF_MODEL_ID}")
        if not args.skip_gguf:
            print(f"  GGUF       : https://huggingface.co/{HF_GGUF_ID}")
    print(f"\n  Local files:")
    print(f"    {MERGED_DIR}/       — merged FP16 (use with transformers)")
    if not args.skip_gguf:
        print(f"    {GGUF_DIR}/           — GGUF Q4_K_M + Q8_0 (use with Ollama / llama.cpp)")
    print(f"    adapter_only/         — LoRA adapters (for further fine-tuning)")
    print("=" * 60)

    print("\nNext step: python inference.py")


if __name__ == "__main__":
    main()
