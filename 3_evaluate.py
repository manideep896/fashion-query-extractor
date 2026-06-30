#!/usr/bin/env python3
"""
3_evaluate.py — Evaluate the trained model on the held-out test set.

Metrics reported:
  • Valid JSON rate        – model output parses as JSON
  • Schema compliance      – valid JSON + all required fields + correct types
  • Exact match rate       – full output == ground truth
  • Null accuracy          – correctly outputs null for unstated fields
  • Field-level F1 (macro) – precision/recall per field, averaged

Reads:   data/test.jsonl
Reads:   checkpoints/final_adapter   (or --model_path)
Writes:  eval_results.json

Usage:
    python 3_evaluate.py
    python 3_evaluate.py --model_path ./checkpoints/final_adapter
    python 3_evaluate.py --max_samples 500   # quick smoke-test
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from config import (
    CHECKPOINT_DIR, DATA_DIR, EVAL_BATCH_SIZE,
    EVAL_RESULTS_PATH, MAX_NEW_TOKENS, MAX_SEQ_LENGTH,
    MODEL_NAME, SEED, SYSTEM_PROMPT,
)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_inference_prompt(query: str, schema: dict, tokenizer) -> str:
    """Build the inference prompt (no assistant answer) using the chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Query: {query}\n\nSchema:\n{json.dumps(schema, indent=2)}"},
    ]
    # add_generation_prompt=True appends <|im_start|>assistant\n
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # Qwen3: skip thinking at eval for speed
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


# ── Inference ─────────────────────────────────────────────────────────────────

def run_batch(model, tokenizer, prompts: list[str]) -> list[str]:
    """Run inference on a batch and return decoded new tokens only."""
    tokenizer.padding_side = "left"   # required for batch generation
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        n_input = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
        new_ids = out[n_input:]
        decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        results.append(decoded)
    return results


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> block if present."""
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[-1].strip()
    return text


# ── Metrics ───────────────────────────────────────────────────────────────────

def check_schema_compliance(output: dict, schema: dict) -> bool:
    if not isinstance(output, dict):
        return False

    required   = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in output:
            return False

    for field, value in output.items():
        if field not in properties:
            # additionalProperties defaults to allowed; skip
            continue
        if value is None:
            continue  # null is always valid for nullable fields

        spec = properties[field]
        expected = spec.get("type")
        if not expected:
            continue

        if isinstance(expected, list):
            non_null = [t for t in expected if t != "null"]
            expected = non_null[0] if non_null else None

        type_map = {"string": str, "array": list, "object": dict,
                    "number": (int, float), "integer": int, "boolean": bool}
        python_type = type_map.get(expected)
        if python_type and not isinstance(value, python_type):
            return False

    return True


def field_f1(pred: dict, gt: dict, schema: dict) -> dict[str, float]:
    """Return per-field F1 score (1.0 = perfect, 0.0 = wrong/missing)."""
    properties = schema.get("properties", {})
    scores = {}

    for field in properties:
        gt_val   = gt.get(field)
        pred_val = pred.get(field)

        if gt_val is None and pred_val is None:
            scores[field] = 1.0
            continue
        if (gt_val is None) != (pred_val is None):
            scores[field] = 0.0
            continue

        spec = properties[field]
        ftype = spec.get("type", "string")
        if isinstance(ftype, list):
            non_null = [t for t in ftype if t != "null"]
            ftype = non_null[0] if non_null else "string"

        if ftype == "array":
            gt_set   = set(str(v).lower().strip() for v in (gt_val   or []))
            pred_set = set(str(v).lower().strip() for v in (pred_val or []))
            if not gt_set and not pred_set:
                scores[field] = 1.0
            elif not gt_set or not pred_set:
                scores[field] = 0.0
            else:
                inter = len(gt_set & pred_set)
                p = inter / len(pred_set)
                r = inter / len(gt_set)
                scores[field] = 2 * p * r / (p + r) if (p + r) else 0.0

        elif ftype == "object":
            scores[field] = (
                1.0 if json.dumps(gt_val, sort_keys=True) ==
                       json.dumps(pred_val, sort_keys=True)
                else 0.0
            )

        else:
            scores[field] = (
                1.0 if str(gt_val).lower().strip() ==
                       str(pred_val).lower().strip()
                else 0.0
            )

    return scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default=str(Path(CHECKPOINT_DIR) / "final_adapter"),
        help="Path to trained LoRA adapter (default: checkpoints/final_adapter)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Evaluate on first N samples only (for quick sanity checks)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Fashion Extractor — Evaluation")
    print("=" * 60)

    # ── Load test data ────────────────────────────────────────────────────────
    test_path = Path(DATA_DIR) / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_path}\n"
            "Run 1_format_data.py first."
        )

    test_samples = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))

    if args.max_samples:
        test_samples = test_samples[: args.max_samples]

    print(f"\nTest samples: {len(test_samples)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)

    # ── Accumulate metrics ────────────────────────────────────────────────────
    total       = 0
    n_valid_json = 0
    n_compliant = 0
    n_exact     = 0
    null_correct = 0
    null_total  = 0
    ff1_totals  : dict[str, float] = {}
    ff1_counts  : dict[str, int]   = {}

    # ── Batch inference ───────────────────────────────────────────────────────
    print(f"\nRunning inference (batch={EVAL_BATCH_SIZE})...")

    batch_prompts  = []
    batch_metadata = []

    def process_batch():
        nonlocal total, n_valid_json, n_compliant, n_exact
        nonlocal null_correct, null_total

        if not batch_prompts:
            return

        try:
            raw_preds = run_batch(model, tokenizer, batch_prompts)
        except Exception as e:
            print(f"  [batch error] {e}")
            return

        for raw_pred, meta in zip(raw_preds, batch_metadata):
            total += 1
            pred_text = strip_thinking(raw_pred)
            gt        = meta["extraction"]
            schema    = meta["schema"]

            # Valid JSON?
            try:
                pred = json.loads(pred_text)
                n_valid_json += 1
            except json.JSONDecodeError:
                continue

            # Schema compliance
            if check_schema_compliance(pred, schema):
                n_compliant += 1

            # Exact match
            if json.dumps(pred, sort_keys=True) == json.dumps(gt, sort_keys=True):
                n_exact += 1

            # Field F1 + null accuracy
            scores = field_f1(pred, gt, schema)
            for field, f1 in scores.items():
                ff1_totals[field] = ff1_totals.get(field, 0.0) + f1
                ff1_counts[field] = ff1_counts.get(field, 0) + 1

                # Null accuracy: field is null in ground truth
                if gt.get(field) is None:
                    null_total += 1
                    if pred.get(field) is None:
                        null_correct += 1

    with tqdm(total=len(test_samples), unit="samples") as bar:
        for sample in test_samples:
            query      = sample.get("query", "")
            schema     = sample.get("schema", {})
            extraction = sample.get("extraction", {})

            if not query or not schema or not extraction:
                bar.update(1)
                continue

            prompt = build_inference_prompt(query, schema, tokenizer)
            batch_prompts.append(prompt)
            batch_metadata.append({"schema": schema, "extraction": extraction})

            if len(batch_prompts) >= EVAL_BATCH_SIZE:
                process_batch()
                batch_prompts.clear()
                batch_metadata.clear()

            bar.update(1)

        # Flush remaining
        process_batch()

    # ── Compute final metrics ─────────────────────────────────────────────────
    per_field_f1 = {
        f: ff1_totals[f] / ff1_counts[f]
        for f in ff1_totals
        if ff1_counts[f] > 0
    }
    macro_f1 = sum(per_field_f1.values()) / len(per_field_f1) if per_field_f1 else 0.0

    safe = lambda n, d: n / d if d else 0.0

    results = {
        "total_samples":          total,
        "valid_json_rate":        safe(n_valid_json, total),
        "schema_compliance_rate": safe(n_compliant,  total),
        "exact_match_rate":       safe(n_exact,      total),
        "null_accuracy":          safe(null_correct, null_total),
        "field_f1_macro":         macro_f1,
        "field_f1_per_field":     per_field_f1,
    }

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Total samples          : {total:,}")
    print(f"  Valid JSON rate        : {results['valid_json_rate']:.1%}")
    print(f"  Schema compliance      : {results['schema_compliance_rate']:.1%}")
    print(f"  Exact match            : {results['exact_match_rate']:.1%}")
    print(f"  Null accuracy          : {results['null_accuracy']:.1%}")
    print(f"  Field F1 (macro avg)   : {results['field_f1_macro']:.1%}")

    if per_field_f1:
        print("\n  Per-field F1:")
        for field, score in sorted(per_field_f1.items(), key=lambda x: -x[1]):
            bar_chars = int(score * 20)
            bar_str   = "█" * bar_chars + "░" * (20 - bar_chars)
            print(f"    {field:<22} {bar_str}  {score:.1%}")

    # ── Save results ──────────────────────────────────────────────────────────
    with open(EVAL_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved → {EVAL_RESULTS_PATH}")

    print("\nNext step: python 4_export_push.py --hf_token YOUR_TOKEN")


if __name__ == "__main__":
    main()
