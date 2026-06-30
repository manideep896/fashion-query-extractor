#!/usr/bin/env python3
"""
1_format_data.py — Format raw data into Qwen3 ChatML JSONL for training.

Expected input format (JSONL — one sample per line):
    {
        "query":      "I want a Hawaiian print shirt size 32",
        "schema":     {"type": "object", "properties": {...}, "required": [...]},
        "extraction": {"category": "shirt", "pattern": "Hawaiian", ...}
    }

  - schema and extraction can be dicts OR valid JSON strings (both accepted)
  - Optional "thinking" field: if present, training sample includes a
    <think>...</think> reasoning block before the JSON (improves complex queries)

Also accepts .csv (columns: query, schema, extraction [, thinking])
and .json (top-level list of dicts).

Quality checks applied:
  1. Required fields present
  2. schema and extraction are valid JSON objects
  3. extraction contains only fields defined in the schema
  4. Token length ≤ MAX_SEQ_LENGTH (longer samples dropped)
  5. Duplicate queries detected and reported

Output (in data/):
    train.jsonl, eval.jsonl, test.jsonl

Each line:
    {
        "text":       "<full Qwen3 ChatML string for training>",
        "query":      "...",
        "schema":     {...},
        "extraction": {...}
    }

The extra fields (query/schema/extraction) are kept so 3_evaluate.py can
use them directly without re-parsing the formatted text.

Usage:
    python 1_format_data.py --input data/raw.jsonl
    python 1_format_data.py --input data/raw.csv --output_dir data
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from config import (
    DATA_DIR, MAX_SEQ_LENGTH, MODEL_NAME, SEED,
    SYSTEM_PROMPT, TRAIN_RATIO, EVAL_RATIO,
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [warn] skipping line {i}: {e}")
    return samples


def load_csv(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_json_array(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a top-level list.")
    return data


def load_raw(input_path: str) -> list[dict]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")
    ext = path.suffix.lower()
    if ext == ".jsonl":   return load_jsonl(path)
    if ext == ".csv":     return load_csv(path)
    if ext == ".json":    return load_json_array(path)
    raise ValueError(f"Unsupported format '{ext}'. Use .jsonl, .csv, or .json.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def as_dict(value, field_name: str) -> dict:
    """Accept a dict or a JSON string; always return a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)   # raises JSONDecodeError on bad JSON
        if not isinstance(parsed, dict):
            raise ValueError(f"'{field_name}' must decode to a JSON object")
        return parsed
    raise ValueError(f"'{field_name}' must be a dict or JSON string, got {type(value).__name__}")


# ── Validation ────────────────────────────────────────────────────────────────

def validate(sample: dict) -> tuple[bool, str]:
    """Return (is_valid, reason). reason is empty string when valid."""
    # Required fields
    for field in ("query", "schema", "extraction"):
        if field not in sample:
            return False, f"missing '{field}'"

    # Query must be non-empty string
    if not isinstance(sample["query"], str) or not sample["query"].strip():
        return False, "query must be a non-empty string"

    # Schema must parse to dict
    try:
        schema = as_dict(sample["schema"], "schema")
    except (ValueError, json.JSONDecodeError) as e:
        return False, f"schema error: {e}"

    # Extraction must parse to dict
    try:
        extraction = as_dict(sample["extraction"], "extraction")
    except (ValueError, json.JSONDecodeError) as e:
        return False, f"extraction error: {e}"

    # Extraction must only contain fields defined in the schema.
    # A ground truth with extra fields is bad training data — it teaches
    # the model to hallucinate fields that the schema doesn't ask for.
    schema_props = schema.get("properties", {})
    if schema_props:
        extra = [f for f in extraction if f not in schema_props]
        if extra:
            return False, f"extraction has fields not in schema: {extra}"

    return True, ""


# ── Formatting ────────────────────────────────────────────────────────────────

def format_sample(sample: dict, tokenizer) -> dict:
    """Convert one validated raw sample into a training-ready dict."""
    query      = sample["query"].strip()
    schema     = as_dict(sample["schema"],     "schema")
    extraction = as_dict(sample["extraction"], "extraction")
    thinking   = sample.get("thinking", "").strip()

    user_content = (
        f"Query: {query}\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}"
    )

    if thinking:
        # Thinking mode: model reasons before producing JSON.
        # Used for ~10-15% of training samples (complex/ambiguous queries).
        # At inference, thinking can be disabled for speed.
        assistant_content = f"<think>\n{thinking}\n</think>\n{json.dumps(extraction)}"
    else:
        assistant_content = json.dumps(extraction)

    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text":       text,
        "query":      query,
        "schema":     schema,
        "extraction": extraction,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True, help="Raw data file (.jsonl / .csv / .json)")
    parser.add_argument("--output_dir", default=DATA_DIR)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    raw = load_raw(args.input)
    print(f"  Loaded {len(raw):,} raw samples")

    # ── Validate + format ─────────────────────────────────────────────────────
    formatted  : list[dict] = []
    n_invalid   = 0
    n_too_long  = 0
    skip_reasons: list[str] = []

    # Track duplicate queries
    query_counter: Counter = Counter()

    print("\nValidating and formatting...")
    for idx, sample in enumerate(raw):
        ok, reason = validate(sample)
        if not ok:
            skip_reasons.append(f"[#{idx}] {reason}")
            n_invalid += 1
            continue

        query_counter[sample["query"].strip().lower()] += 1

        try:
            record = format_sample(sample, tokenizer)
        except Exception as e:
            skip_reasons.append(f"[#{idx}] format error: {e}")
            n_invalid += 1
            continue

        token_len = len(tokenizer.encode(record["text"]))
        if token_len > MAX_SEQ_LENGTH:
            n_too_long += 1
            continue

        formatted.append(record)

    # Print skip reasons (first 10)
    if skip_reasons:
        print(f"\n  First {min(10, len(skip_reasons))} skipped samples:")
        for r in skip_reasons[:10]:
            print(f"    {r}")
        if len(skip_reasons) > 10:
            print(f"    ... and {len(skip_reasons) - 10} more")

    # Duplicate report
    duplicates = {q: c for q, c in query_counter.items() if c > 1}
    if duplicates:
        print(f"\n  Duplicate queries detected: {len(duplicates):,}")
        print(f"  (These are kept — varied schemas for the same query is valid training data.)")

    # Thinking mode stats
    n_thinking = sum(1 for r in formatted if "<think>" in r["text"])
    print(f"\n  Formatted (valid)    : {len(formatted):,}")
    print(f"  Skipped (invalid)    : {n_invalid:,}")
    print(f"  Skipped (too long)   : {n_too_long:,}")
    print(f"  Thinking mode samples: {n_thinking:,} ({100*n_thinking/max(len(formatted),1):.1f}%)")

    if not formatted:
        print("\nERROR: No valid samples. Check your input data format.")
        sys.exit(1)

    # ── Shuffle + split ───────────────────────────────────────────────────────
    random.seed(SEED)
    random.shuffle(formatted)

    n       = len(formatted)
    n_train = int(n * TRAIN_RATIO)
    n_eval  = int(n * EVAL_RATIO)

    splits = {
        "train": formatted[:n_train],
        "eval":  formatted[n_train : n_train + n_eval],
        "test":  formatted[n_train + n_eval :],
    }

    # ── Write ─────────────────────────────────────────────────────────────────
    print("")
    for name, records in splits.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(records, path)
        print(f"  {name:<6}  {len(records):>7,} samples  →  {path}")

    print(f"\nTotal usable: {n:,}")
    print("Next step: python 2_train.py")


if __name__ == "__main__":
    main()
