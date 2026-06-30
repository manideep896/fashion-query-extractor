#!/usr/bin/env python3
"""
inference.py — Runtime inference with Outlines constrained JSON decoding.

Outlines guarantees the output is always valid JSON conforming to your schema —
no post-processing needed, no json.loads() failures.

Usage:
    python inference.py                            # built-in example
    python inference.py --model_path ./merged_model
    python inference.py --query "blue linen shirt size M under $60"
    python inference.py --no_outlines              # raw model, for comparison
"""

import argparse
import json

from config import MERGED_DIR, SYSTEM_PROMPT

# ── Example inputs ────────────────────────────────────────────────────────────

EXAMPLE_QUERY = (
    "I want a Hawaiian print shirt for size 32, "
    "something in blue or white, casual vibe, preferably cotton, under $50"
)

EXAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "category":    {"type": "string"},
        "pattern":     {"type": ["string", "null"]},
        "size":        {"type": ["string", "null"]},
        "color":       {"type": "array", "items": {"type": "string"}},
        "material":    {"type": ["array", "null"], "items": {"type": "string"}},
        "style":       {"type": ["string", "null"]},
        "gender":      {"type": ["string", "null"]},
        "price_range": {
            "type": ["object", "null"],
            "properties": {
                "min":      {"type": ["number", "null"]},
                "max":      {"type": ["number", "null"]},
                "currency": {"type": "string"},
            },
        },
    },
    "required": [
        "category", "pattern", "size", "color",
        "material", "style", "gender", "price_range",
    ],
}


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query: str, schema: dict) -> str:
    """Build the ChatML inference prompt manually (no tokenizer needed for Outlines)."""
    user_content = f"Query: {query}\n\nSchema:\n{json.dumps(schema, indent=2)}"
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Inference with Outlines (constrained, guaranteed valid JSON) ───────────────

def extract_with_outlines(model_path: str, query: str, schema: dict) -> dict:
    import outlines

    print("  Loading model with Outlines...")
    model = outlines.models.transformers(
        model_path,
        model_kwargs={
            "torch_dtype": "auto",
            "device_map": "auto",
        },
    )

    generator = outlines.generate.json(model, schema)
    prompt    = build_prompt(query, schema)

    print("  Generating...")
    result = generator(prompt, max_tokens=512)
    return result


# ── Inference without Outlines (raw model output, for comparison) ─────────────

def extract_raw(model_path: str, query: str, schema: dict) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    prompt = build_prompt(query, schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # Strip thinking block if model generated one
    if "<think>" in decoded and "</think>" in decoded:
        decoded = decoded.split("</think>", 1)[-1].strip()

    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return {"__error__": "model did not output valid JSON", "__raw__": decoded}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", default=MERGED_DIR,
        help=f"Path to merged FP16 model (default: {MERGED_DIR})",
    )
    parser.add_argument(
        "--query", default=EXAMPLE_QUERY,
        help="Fashion search query to extract from",
    )
    parser.add_argument(
        "--no_outlines", action="store_true",
        help="Use raw model inference instead of Outlines (output not guaranteed valid JSON)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Fashion Extractor — Inference")
    print("=" * 60)
    print(f"\n  Query  : {args.query}")
    print(f"  Fields : {list(EXAMPLE_SCHEMA['properties'].keys())}")
    print(f"  Mode   : {'raw (no Outlines)' if args.no_outlines else 'Outlines constrained'}")
    print()

    if args.no_outlines:
        result = extract_raw(args.model_path, args.query, EXAMPLE_SCHEMA)
    else:
        result = extract_with_outlines(args.model_path, args.query, EXAMPLE_SCHEMA)

    print("\n  Extracted JSON:")
    print(json.dumps(result, indent=2))

    if "__error__" in result:
        print(f"\n  WARNING: {result['__error__']}")
    else:
        print(f"\n  Fields extracted : {sum(1 for v in result.values() if v is not None)}")
        print(f"  Fields null      : {sum(1 for v in result.values() if v is None)}")


if __name__ == "__main__":
    main()
