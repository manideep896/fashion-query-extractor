# Fashion Query Extractor — Complete Fine-Tuning Guide

> **Who this is for:** Someone who knows what they want to build but has never fine-tuned an LLM before. Every concept is explained from scratch.

---

## What We Are Building

A small language model (~4B parameters) that takes two inputs:

1. A natural language fashion search query
   > *"I want a Hawaiian print shirt in size 32, blue or white, casual"*

2. A JSON schema defining what to extract
   ```json
   {"properties": {"category": {}, "pattern": {}, "size": {}, "color": {}, "style": {}}, "required": [...]}
   ```

And outputs a structured JSON object:
```json
{
  "category": "shirt",
  "pattern": "Hawaiian print",
  "size": "32",
  "color": ["blue", "white"],
  "style": "casual"
}
```

The model works with **any schema you pass at runtime** — you are not locked into one fixed set of fields. This is the key design decision.

---

## What Fine-Tuning Is (Plain English)

The base model — Qwen3-4B — is already trained on trillions of words from the internet. It can read, reason, and write. But it doesn't automatically know how to extract structured fashion attributes from queries.

**Fine-tuning** is giving the model a set of examples — "here's a query, here's a schema, here's the correct JSON output" — and adjusting it just enough to become very good at that specific task.

Think of it like this:
- The base model is a highly educated generalist
- Fine-tuning is a specialization residency program
- Your dataset is the curriculum

---

## What QLoRA Is (Why We Don't Train Everything)

Training all 4.44 billion parameters from scratch would need:
- Multiple A100s for weeks
- Terabytes of data
- Massive cost

**QLoRA** avoids this by:
1. **Freezing** 99% of the model's parameters (they don't change)
2. **Adding tiny adapter layers** (~32M trainable parameters, less than 1% of the model)
3. **Quantizing** the frozen base model to 4-bit (uses ~4x less memory)
4. Training **only the adapters**

The result: you get 95%+ of the quality of full fine-tuning at 1/80th the compute cost.

---

## Pipeline Overview

```
Your Dataset (JSONL)
        │
        ▼
┌─────────────────────┐
│  1_format_data.py   │  Validates, formats to ChatML, splits 80/10/10
└─────────────────────┘
        │
        ▼
  data/train.jsonl
  data/eval.jsonl
  data/test.jsonl
        │
        ▼
┌─────────────────────┐
│    2_train.py       │  QLoRA fine-tuning with Unsloth (2–5 hours on A100)
└─────────────────────┘
        │
        ▼
  checkpoints/final_adapter/
        │
        ▼
┌─────────────────────┐
│   3_evaluate.py     │  Measures schema compliance, field F1, null accuracy
└─────────────────────┘
        │
        ▼
  eval_results.json
        │
        ▼
┌─────────────────────┐
│  4_export_push.py   │  Merges adapters → FP16 + GGUF → HuggingFace
└─────────────────────┘
        │
        ▼
  HuggingFace Hub (public model)
        │
        ▼
┌─────────────────────┐
│   inference.py      │  Use the model + Outlines for guaranteed valid JSON
└─────────────────────┘
```

---

## Dataset Format

Your JSONL file (`data/raw.jsonl`) — one JSON object per line:

```jsonl
{"query": "blue denim jacket size M slim fit", "schema": {"type": "object", "properties": {"category": {"type": "string"}, "color": {"type": "array", "items": {"type": "string"}}, "size": {"type": ["string", "null"]}, "fit": {"type": ["string", "null"]}}, "required": ["category", "color", "size", "fit"]}, "extraction": {"category": "jacket", "color": ["blue"], "size": "M", "fit": "slim"}}
{"query": "floral summer dress under $80", "schema": {...}, "extraction": {...}}
```

**Required fields per sample:**
| Field | Type | Description |
|---|---|---|
| `query` | string | The raw search query |
| `schema` | object | JSON Schema defining what to extract |
| `extraction` | object | Ground truth extracted values |
| `thinking` | string | *(optional)* Reasoning before the JSON (use for 10-15% of complex samples) |

**Rules for good training data:**
- `extraction` must only contain fields that exist in `schema.properties`
- `extraction` values should match the type specified in `schema`
- Use `null` for fields not mentioned in the query
- The format script enforces all of these — bad samples are automatically dropped

---

## How the Training Format Works (ChatML)

The model is trained on conversations in this exact format:

```
<|im_start|>system
You are a structured extraction model. Given a search query and a JSON schema,
extract the requested fields. Output only valid JSON conforming to the schema.
Use null for fields not mentioned in the query. Do not add fields that are not
in the schema.<|im_end|>
<|im_start|>user
Query: blue denim jacket size M slim fit

Schema:
{
  "type": "object",
  "properties": {
    "category": {"type": "string"},
    "color":    {"type": "array", "items": {"type": "string"}},
    "size":     {"type": ["string", "null"]},
    "fit":      {"type": ["string", "null"]}
  },
  "required": ["category", "color", "size", "fit"]
}<|im_end|>
<|im_start|>assistant
{"category": "jacket", "color": ["blue"], "size": "M", "fit": "slim"}<|im_end|>
```

**Critical training decision — Response-Only Loss:**
The loss (the signal that changes the model's weights) is computed **only on the assistant's JSON output**. The system prompt and user query are masked from the loss. This means every training step is focused purely on making the JSON extraction better, not on predicting the fixed system prompt.

---

## Step-by-Step: Running on RunPod

### Step 0: Start Your Pod

- Choose an A100 80GB template (or 40GB — both work)
- Start a pod with PyTorch pre-installed
- Open the terminal

### Step 1: Clone This Repository

```bash
cd /workspace
git clone https://github.com/manideep896/fashion-query-extractor.git
cd fashion-query-extractor
```

### Step 2: Install Dependencies

```bash
bash setup.sh
```

This installs Unsloth (the fast fine-tuning library), HuggingFace Transformers, TRL, PEFT, and all other dependencies. Takes ~5 minutes.

### Step 3: Add Your Dataset

```bash
mkdir -p data
# Upload your JSONL file as data/raw.jsonl
# or copy it from wherever you have it
```

### Step 4: Format the Data

```bash
python 1_format_data.py --input data/raw.jsonl
```

**What happens:**
- Loads all your samples
- Validates each one (bad samples are dropped with a reason)
- Checks that extractions only have fields defined in the schema
- Wraps each sample in the Qwen3 ChatML format
- Shuffles and splits into 80% train / 10% eval / 10% test
- Saves to `data/train.jsonl`, `data/eval.jsonl`, `data/test.jsonl`

**Expected output:**
```
Loaded 45,000 raw samples
Formatted (valid)    : 44,231
Skipped (invalid)    : 512
Skipped (too long)   : 257
Thinking mode samples: 4,821 (10.9%)

train    35,384 samples  →  data/train.jsonl
eval      4,423 samples  →  data/eval.jsonl
test      4,424 samples  →  data/test.jsonl
```

### Step 5: Train

```bash
python 2_train.py
```

**What happens:**
- Downloads Qwen3-4B-Instruct (~8GB, cached after first run)
- Loads it in 4-bit quantization (~4GB VRAM)
- Attaches LoRA adapters (~32M trainable parameters)
- Trains for 3 epochs with eval every 200 steps
- Saves a checkpoint whenever eval loss improves
- At the end, restores the best checkpoint and saves to `checkpoints/final_adapter/`

**Expected training output (per logging step):**
```
{'loss': 0.8234, 'learning_rate': 0.00019, 'epoch': 0.10}
{'loss': 0.5123, 'learning_rate': 0.00018, 'epoch': 0.25}
{'eval_loss': 0.4891, 'eval_runtime': 45.2, 'epoch': 0.50}
...
```

Loss should decrease from ~1.0 → ~0.2–0.4 over 3 epochs. If it plateaus early, training stops automatically.

**Expected time on A100 80GB:** 3–5 hours for 35K samples × 3 epochs.

### Step 6: Evaluate

```bash
python 3_evaluate.py
```

**What happens:**
- Loads the trained model from `checkpoints/final_adapter/`
- Runs inference on every sample in `data/test.jsonl`
- For each output, checks:
  - **Valid JSON rate**: did the model output parseable JSON?
  - **Schema compliance**: does the output have required fields with correct types?
  - **Exact match**: does the output exactly match ground truth?
  - **Null accuracy**: for fields not in the query, did the model correctly output null?
  - **Field F1**: per-field precision/recall (averaged across all fields)
- Saves results to `eval_results.json`

**Expected output:**
```
RESULTS
Total samples          : 4,424
Valid JSON rate        : 98.7%
Schema compliance      : 96.2%
Exact match            : 52.1%
Null accuracy          : 93.8%
Field F1 (macro avg)   : 89.3%

Per-field F1:
  category              ████████████████████  97.2%
  color                 ███████████████████░  94.6%
  pattern               ██████████████████░░  91.1%
  size                  █████████████████░░░  87.4%
  fit                   ████████████████░░░░  83.2%
  price_range           ██████████████░░░░░░  72.4%
```

### Step 7: Export and Push to HuggingFace

```bash
python 4_export_push.py --hf_token hf_YOUR_TOKEN_HERE
```

Or set the token as an environment variable:
```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
python 4_export_push.py
```

**What happens:**
1. Merges the LoRA adapters into the base model weights (creates a standalone model)
2. Saves merged FP16 model to `merged_model/` (~9GB)
3. Pushes FP16 model to `ionio-io/fashion-query-extractor-4B` on HuggingFace
4. Creates GGUF quantized versions:
   - Q4_K_M (~2.5GB) — for Ollama / llama.cpp
   - Q8_0 (~4.7GB) — higher quality, still smaller than FP16
5. Pushes GGUF files to `ionio-io/fashion-query-extractor-4B-GGUF`

### Step 8: Use the Model

```bash
python inference.py --query "something floral and summery, under 50 bucks, for the beach"
```

**Output:**
```json
{
  "category": null,
  "pattern": "floral",
  "season": ["summer"],
  "occasion": ["beach"],
  "price_range": {"min": null, "max": 50, "currency": "USD"},
  "style": null,
  "color": null
}
```

---

## How Outlines Works (Why JSON Is Guaranteed)

After fine-tuning, the model is very good at producing valid JSON — but not perfect. For production, we use **Outlines** (constrained decoding library) to mathematically guarantee valid JSON output.

How it works: instead of letting the model freely generate any token, Outlines builds a finite state machine from your JSON schema and only allows token generation that keeps the output valid according to the schema. If a token would break the JSON structure, it's masked with probability 0.

Result:
- Without Outlines: ~97% valid JSON (after fine-tuning)
- With Outlines: 100% valid JSON, always conforming to schema

This is what you use in production. The fine-tuned model supplies the extraction intelligence; Outlines supplies the structural guarantee.

---

## Key Configuration (config.py)

All tuneable values are in one file. Change here, not in individual scripts.

| Setting | Value | Why |
|---|---|---|
| `MODEL_NAME` | `unsloth/Qwen3-4B-Instruct` | Best ~4B model for structured output (2025) |
| `LORA_R` | 32 | Enough capacity for schema generalization, not so much it overfits |
| `LORA_ALPHA` | 64 | Standard 2× rank scaling |
| `NUM_EPOCHS` | 3 | Right for ~40K samples; stops early if eval loss plateaus |
| `LEARNING_RATE` | 2e-4 | Optimal for LoRA SFT at this scale |
| `PER_DEVICE_BATCH_SIZE` | 8 | Tuned for A100 80GB |
| `EVAL_STEPS` | 200 | Evaluate every 200 steps; saves best checkpoint |

---

## Training Metrics Explained

| Metric | What it means | Target |
|---|---|---|
| `train_loss` | How wrong the model is on training data | Should decrease steadily |
| `eval_loss` | How wrong on unseen eval data | Should track train_loss; if it rises while train_loss falls → overfitting |
| Schema compliance | % of outputs that parse as JSON AND have all required fields with correct types | > 96% |
| Field F1 | For each schema field, how accurate is the extracted value? (precision × recall) | > 88% avg |
| Null accuracy | % of fields that are correctly null when not in query | > 93% |
| Exact match | % of outputs that exactly match ground truth JSON | ~50–60% (ceiling metric) |

---

## Troubleshooting

**`CUDA out of memory`**
- Reduce `PER_DEVICE_BATCH_SIZE` to 4 in `config.py`
- Or reduce `MAX_SEQ_LENGTH` to 768

**`unsloth not found`**
- Run `bash setup.sh` again

**`model not found: unsloth/Qwen3-4B-Instruct`**
- Make sure you have internet access on RunPod
- Try `huggingface-cli login` with your HF token

**Training loss not decreasing after epoch 1**
- Check your data quality — are schemas and extractions consistent?
- Try reducing `LEARNING_RATE` to `1e-4`

**Schema compliance < 80%**
- Your eval data may have schemas the model hasn't seen
- Consider adding more schema variants to training data

**`DataCollatorForCompletionOnlyLM` warning: some samples have all tokens masked**
- A few samples may have the response template not found in the tokenized sequence
- Usually < 1% of samples; safe to ignore
- Caused by very long schemas that push the template past MAX_SEQ_LENGTH after truncation

---

## Files Reference

```
fashion-query-extractor/
├── setup.sh              ← Run once on RunPod to install everything
├── requirements.txt      ← Package list
├── config.py             ← All settings in one place (edit this)
├── 1_format_data.py      ← Format your data for training
├── 2_train.py            ← Fine-tuning (the main script)
├── 3_evaluate.py         ← Measure quality on test set
├── 4_export_push.py      ← Export model + push to HuggingFace
├── inference.py          ← Use the trained model in production
├── FLOW.md               ← This document
└── data/                 ← Put your raw.jsonl here (not in git)
    ├── raw.jsonl         ← Your input data
    ├── train.jsonl       ← Generated by 1_format_data.py
    ├── eval.jsonl        ← Generated by 1_format_data.py
    └── test.jsonl        ← Generated by 1_format_data.py
```
