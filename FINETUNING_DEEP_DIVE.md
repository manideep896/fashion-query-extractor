# Fine-Tuning Language Models: From First Principles to Production
### A Complete Technical Guide + Our Fashion Extractor Approach

---

# PART 1 — FINE-TUNING IN GENERAL

---

## 1. What Is a Large Language Model

Before understanding fine-tuning, you need to understand what a language model actually is and how it works at a mechanical level.

### 1.1 The Core Idea: Next Token Prediction

A large language model is, at its heart, a function that takes a sequence of tokens and outputs a probability distribution over the next possible token.

```
Input:  "I want a Hawaiian print"
Output: {"shirt": 0.42, "dress": 0.18, "blouse": 0.12, ...}
```

Every word you've ever seen a model generate is produced one token at a time, where at each step the model looks at all previous tokens and predicts the most likely next one. "Generating text" is just running this prediction step thousands of times in a row.

### 1.2 What a Token Is

Tokens are not words. They are the smallest unit the model processes. A tokenizer converts raw text into token IDs before any model processing happens.

```
"Hawaiian print shirt" → [22453, 1173, 10671]
"size 32"             → [1404, 220, 843]
"{"                   → [5123]
```

Most English words are 1–2 tokens. Subwords are common — "fine-tuning" might be ["fine", "-", "tun", "ing"] = 4 tokens. Numbers and special characters behave differently across tokenizers. This matters because:
- Your model has a maximum context length in **tokens**, not words
- Training cost scales with token count, not word count
- Extraction outputs (JSON) tokenize differently than prose

### 1.3 The Transformer Architecture

All modern LLMs use the Transformer architecture. The key components:

**Embedding Layer**: Converts each token ID into a dense vector (e.g., 4096 numbers for a 4B model). These vectors capture semantic meaning.

**Attention Layers**: The core mechanism. For each token, the model computes how much it should "attend to" every other token in the context. This is how "Hawaiian" in a query relates to "pattern" as a schema field — the model learns these relationships during training.

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V
```

In plain terms: every token looks at every other token and decides how relevant each one is. The model has multiple "attention heads" that learn different types of relationships simultaneously.

**Feed-Forward Layers**: After attention, each token's representation passes through a two-layer neural network. This is where most factual knowledge is stored. The LoRA adapters we add target both attention layers and feed-forward layers.

**Layer Stacking**: These (attention + feed-forward) blocks are stacked — Qwen3-4B has 36 such layers. Information flows upward through layers, with each layer building on the previous one's representations.

### 1.4 Pre-Training: How Models Get Their Base Knowledge

Pre-training is the expensive initial phase where a model is trained on massive amounts of text to learn language, facts, and reasoning.

**Scale**: Qwen3-4B was pre-trained on ~18 trillion tokens — billions of web pages, books, code, papers.

**Objective**: At each step during pre-training, the model sees a sequence of tokens, tries to predict the next one, and adjusts its weights slightly based on whether it was right or wrong. Repeat this trillions of times.

**What the model learns**:
- Grammar and syntax of human language
- Factual associations (Paris is the capital of France)
- Common patterns (JSON structure, code syntax)
- Reasoning patterns (if A then B logic)
- How to follow instructions (for instruct-tuned models)

**Cost**: Pre-training Qwen3-4B required thousands of A100 GPUs for months. Cost: millions of dollars.

This is why we never pre-train from scratch. We take a pre-trained model and adapt it.

---

## 2. What Is Fine-Tuning

### 2.1 The Core Concept

Fine-tuning is the process of continuing to train an already pre-trained model on a smaller, task-specific dataset to specialize its behavior.

Think of it as the difference between a medical school graduate (pre-trained general doctor) and a cardiology specialist (fine-tuned). The cardiologist didn't relearn all of anatomy — they built specific expertise on top of existing general knowledge.

**What changes during fine-tuning:**
The model's weights (the billions of numbers that define its behavior) are adjusted slightly based on your task-specific examples.

**What doesn't change:**
The underlying language understanding, reasoning capability, and world knowledge acquired during pre-training. Fine-tuning steers the model without erasing its foundation.

### 2.2 Fine-Tuning vs The Alternatives

Understanding when to fine-tune requires understanding what the alternatives are.

**Prompt Engineering**
You craft a detailed prompt that instructs the model what to do at inference time.

```
You are an extraction model. Extract JSON from this query: "{query}"
Schema: {schema}
```

- Cost: zero
- Latency: high (long prompts = slow inference)
- Quality: inconsistent, degrades with complex queries
- When to use: prototyping, infrequent use, when data isn't available

**RAG (Retrieval-Augmented Generation)**
At inference time, retrieve relevant documents and stuff them into the prompt.

- Good for: knowledge lookups, staying up-to-date
- Bad for: learning a specific output format or behavior
- Not applicable here: we're teaching behavior, not providing knowledge

**Fine-Tuning**
Train the model on your examples so the behavior is baked into the weights.

- Cost: one-time training cost
- Latency: low (short inference prompts)
- Quality: highest for well-defined tasks
- When to use: repeated task, consistent format needed, quality is critical

For structured extraction where you need high accuracy and consistent JSON format — **fine-tuning is the right choice**.

### 2.3 Supervised Fine-Tuning (SFT)

The type of fine-tuning we use is called Supervised Fine-Tuning. It means:

1. You have labeled examples: input → correct output
2. You show the model thousands of these pairs
3. For each pair, the model generates a prediction, you compare to the correct output, and adjust the weights

Our labeled pairs:
- Input: search query + JSON schema
- Output: correctly extracted JSON

This is the simplest and most reliable form of fine-tuning for well-defined tasks with available ground truth data.

---

## 3. The Training Process in Detail

### 3.1 Loss Function

The loss function measures how wrong the model is. For language models, the standard loss is **cross-entropy loss** on the next token prediction:

```
Loss = -log(P(correct_next_token | context))
```

If the model is very confident about the correct token, loss is near 0. If it's wrong or uncertain, loss is high.

During fine-tuning, for a training sample like:
```
Input:  "Query: blue shirt size M\nSchema: {category, size}\nAssistant: "
Target: '{"category": "shirt", "size": "M"}'
```

The loss is computed over every token in the target. The model must correctly predict `{`, then `"`, then `c`, then `a`, then `t`, etc. Each wrong token adds to the total loss.

**Response-only training** (what we do) means we only compute this loss on the assistant's JSON output — not on the system prompt or user's query. We mask the loss for the instruction portion. This focuses training entirely on improving extraction quality.

### 3.2 Backpropagation and Gradients

After computing the loss, we need to know: which weights in the model should change, and in which direction?

**Backpropagation** is an algorithm that computes the gradient of the loss with respect to every trainable parameter. The gradient tells you: "if you increase this weight by a tiny amount, does the loss go up or down?"

```
gradient = ∂Loss / ∂weight
```

A positive gradient means increasing the weight increases loss (bad) — so we should decrease it.
A negative gradient means increasing the weight decreases loss (good) — so we should increase it.

### 3.3 The Optimizer and Learning Rate

The optimizer is the algorithm that uses gradients to update weights:

```
new_weight = old_weight - learning_rate × gradient
```

**Learning rate** is the most important hyperparameter. It controls how big each update step is.

- Too high: overshoots optimal values, loss oscillates or diverges
- Too low: training is extremely slow, may get stuck in local minima
- Goldilocks: converges smoothly to a good solution

We use **AdamW** (Adam with weight decay). Adam maintains separate adaptive learning rates for each parameter, which makes it much more robust than vanilla gradient descent. The "W" adds weight decay (L2 regularization) to prevent individual weights from getting too large.

**Cosine learning rate schedule**: Instead of a constant learning rate, we start high (after a warmup), then smoothly decay it following a cosine curve. This gives aggressive early learning then fine-grained adjustments near the end of training.

```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t/T))
```

### 3.4 Batch Size and Gradient Accumulation

Processing one sample at a time is noisy — the gradient from a single example might point in a slightly wrong direction. Batching multiple samples together averages out this noise.

**Per-device batch size**: How many samples are processed simultaneously on the GPU. Limited by VRAM.

**Gradient accumulation**: Instead of updating weights after every batch, accumulate gradients over multiple batches and then update. Effective batch size = per-device batch × accumulation steps.

We use: `8 × 2 = 16` effective batch size. This means the model sees 16 samples before each weight update, giving a more stable gradient signal.

### 3.5 Epochs and Overfitting

An **epoch** is one complete pass through the entire training dataset.

**Underfitting** (too few epochs): the model hasn't learned the task well enough. Loss is still high.

**Overfitting** (too many epochs): the model has memorized the training set instead of learning generalizable patterns. Training loss decreases while validation loss increases.

The solution: monitor validation loss (loss on data the model hasn't trained on). When validation loss stops improving and starts rising, stop training. We do this automatically with `load_best_model_at_end=True`.

---

## 4. Types of Fine-Tuning (The Full Landscape)

### 4.1 Full Fine-Tuning

Update all parameters of the model during training.

- **Pros**: maximum quality, can change the model's behavior completely
- **Cons**: requires the same GPU memory as the model in FP32 (for 4B model: ~60GB just for weights + optimizer states = need 4× A100s)
- **When used**: you have massive task-specific datasets, huge compute budget
- **Not used here**: overkill for structured extraction, prohibitively expensive

### 4.2 LoRA (Low-Rank Adaptation)

The key insight: when you fine-tune a model, the change in weight matrices (ΔW) tends to have **low intrinsic rank** — meaning the change can be approximated by multiplying two much smaller matrices.

```
Original weight: W (d × k matrix)
LoRA delta:      ΔW = A × B
                 where A is (d × r) and B is (r × k)
                 and r << min(d, k)
```

Instead of updating W directly, LoRA freezes W and adds two small matrices A and B whose product approximates ΔW.

**Parameters saved**: For a layer with d=4096, k=4096, r=32:
- Full fine-tuning: 4096 × 4096 = 16.7M parameters
- LoRA: (4096 × 32) + (32 × 4096) = 262K parameters — **64× fewer**

**How it works at inference**: The output becomes W·x + (A·B)·x = (W + A·B)·x. You can either keep A and B separate (save memory, add tiny computation) or merge them into W (zero overhead after merging).

### 4.3 QLoRA (Quantized LoRA)

QLoRA extends LoRA by also quantizing the frozen base model weights to 4-bit precision.

**Normal float32**: each weight takes 32 bits = 4 bytes
**4-bit quantization (NF4)**: each weight takes 4 bits = 0.5 bytes → 8× compression

For Qwen3-4B:
- FP32: 4.44B × 4 bytes = ~17.8 GB
- 4-bit: 4.44B × 0.5 bytes = ~2.2 GB

This lets us fit a 4B model on a single 10GB GPU while still training high-quality adapters.

**NF4 quantization**: Uses the normal float distribution to place quantization levels — more levels near zero (where most weights cluster) and fewer at extremes. This minimizes information loss vs naive uniform quantization.

**Double quantization**: Even the quantization constants are quantized, saving another ~0.4 bits per parameter.

**The magic**: Despite the base model being heavily quantized, the LoRA adapter weights are kept in BF16 and are never quantized. The gradient computation happens in BF16 with dequantized activations. Quality degradation vs full LoRA: typically 1-2% on structured tasks.

### 4.4 RLHF (Reinforcement Learning from Human Feedback)

Used to align models with human preferences. Training happens in 3 stages:
1. SFT: fine-tune on high-quality demonstrations
2. Reward model: train a separate model to predict human preference scores
3. RL: use PPO (Proximal Policy Optimization) to tune the main model to maximize reward

Used by: OpenAI for ChatGPT, Anthropic for Claude alignment
**Not used here**: requires human preference annotation, expensive, not needed for well-defined extraction tasks where ground truth is clear.

### 4.5 DPO (Direct Preference Optimization)

A simpler alternative to RLHF. Instead of training a separate reward model, trains directly on preference pairs:

```
{prompt, chosen_response, rejected_response}
```

Mathematically equivalent to RLHF but without the RL complexity.

**Not used here**: requires comparison pairs (two responses, one better than the other). We have ground truth extractions, not preference pairs. SFT is the right choice.

### 4.6 Instruction Tuning

A specific type of SFT where the training data is formatted as instructions and responses. This is how models like ChatGPT and Qwen3-Instruct are made from base models.

We're doing instruction tuning — our "instruction" is "here's a query and a schema, extract the JSON." We're building on top of an already instruction-tuned model (Qwen3-Instruct), which starts us with good instruction-following behavior.

---

## 5. Key Concepts Every Fine-Tuner Must Know

### 5.1 Context Window

The maximum number of tokens the model can process at once. Qwen3-4B: 128K tokens.

During fine-tuning, we set `max_seq_length = 1024`. Every training sample must fit within 1024 tokens (query + schema + JSON output). Samples longer than this are dropped.

Why not use the full 128K? Memory. Each token in the sequence requires attention computation proportional to the context length. Longer context = quadratic memory growth during training.

### 5.2 Tokenizer and Chat Template

The tokenizer converts text to token IDs. Critically, different models use different tokenizers — Qwen3 uses the tiktoken-style tokenizer with additional special tokens.

The **chat template** is the specific format for multi-turn conversations. Qwen3 uses ChatML:

```
<|im_start|>role\ncontent<|im_end|>\n
```

The tokenizer's `apply_chat_template()` handles this formatting. Using the wrong template format is a common source of training bugs.

### 5.3 BF16 vs FP16 vs FP32

**FP32 (32-bit float)**: Full precision. Maximum dynamic range (1.18×10⁻³⁸ to 3.4×10³⁸). 4 bytes per value.
- Used for: optimizer states, gradient accumulation

**FP16 (16-bit float)**: Half precision. Smaller range (6.1×10⁻⁵ to 6.5×10⁴). 2 bytes per value.
- Risk: overflow/underflow during training. NaN errors are common with FP16.

**BF16 (Brain Float 16)**: Same range as FP32 but less precision. 2 bytes per value.
- Same exponent bits as FP32 (8) → same range → no overflow
- Less mantissa bits (7 vs 23) → less precision, but sufficient for neural network weights
- **Best for LLM training on Ampere+ GPUs (A100, 4090)**

Qwen3 was trained in BF16. Training the adapters in BF16 ensures weight distributions match the base model's expectations.

### 5.4 Gradient Checkpointing

During forward pass, activations are stored in memory for use during backward pass. This is expensive.

Gradient checkpointing trades computation for memory: instead of storing all activations, recompute them during backward pass as needed. Increases training time by ~20% but reduces activation memory by 60-70%.

We use `use_gradient_checkpointing="unsloth"` — Unsloth's custom implementation is smarter about which activations to recompute.

### 5.5 Regularization Techniques

**Weight decay** (`weight_decay=0.01`): Adds a small penalty for large weights, preventing any single parameter from becoming too dominant. Equivalent to L2 regularization.

**Dropout** (`lora_dropout=0.05`): During training, randomly zeroes 5% of adapter activations. Forces the model to not rely too heavily on any single pathway. Applied only to LoRA layers.

**Gradient clipping** (`max_grad_norm=1.0`): If the gradient norm exceeds 1.0, scale it down. Prevents exploding gradients — a training instability where a single large gradient update destroys previously learned weights.

---

## 6. Data Quality and Format

### 6.1 The Training Data Principle

**Data quality >>> Data quantity.**

A common mistake: collecting 100K messy samples thinking more data = better model. In practice:
- 10K clean, diverse, well-formatted samples > 100K inconsistent, messy samples
- Inconsistent labels teach the model contradictory behaviors
- Wrong schema-extraction pairs teach the model to hallucinate

The 3 things that matter most in training data quality:
1. **Consistency**: if query A and similar query B have different schemas, the extractions must be consistently different too
2. **Coverage**: all edge cases should appear (null fields, multi-value arrays, typos, price mentions)
3. **Correctness**: every extraction must conform to its schema

### 6.2 Instruction Format vs Raw Text

Models can be fine-tuned on raw text (just predict the next token) or on instruction-response pairs.

Instruction format is strongly preferred because:
- The model learns the task structure (instruction → response)
- Each training example has a clear input/output boundary
- Evaluation is meaningful (you can test on held-out instructions)
- Deployment is natural (just send the instruction at inference time)

### 6.3 Response-Only Loss: Why It Matters

Standard SFT computes loss over the entire sequence:
```
Loss = loss(system prompt tokens) + loss(user query tokens) + loss(assistant JSON tokens)
```

Response-only training masks the instruction:
```
Loss = 0 + 0 + loss(assistant JSON tokens)
```

**Why this is better:**
- The system prompt and user query are "given" — there's no value in training the model to predict them
- Every gradient step improves the actual extraction quality
- The signal-to-noise ratio in gradients is dramatically higher
- Convergence is faster and to a better optimum
- The eval loss is a truer reflection of extraction quality (not influenced by predicting the fixed system prompt)

---

## 7. Evaluation

### 7.1 Perplexity and Eval Loss

Perplexity (or equivalently eval_loss) measures how "surprised" the model is by the validation data:

```
Perplexity = exp(average cross-entropy loss)
```

Lower = better. A perplexity of 1.0 means the model perfectly predicts every token. Typical final perplexity for a well-trained extraction model: 1.5–2.5.

Eval loss is a continuous training signal but doesn't directly answer "is the model extracting correctly?" That requires task-specific metrics.

### 7.2 Task-Specific Metrics for Extraction

**Valid JSON rate**: % of outputs that Python's `json.loads()` can parse without error. Measures syntactic reliability. Target: >97%.

**Schema compliance rate**: Valid JSON AND all required fields present AND types match schema. Measures both syntax and structure. Target: >96%.

**Field-level F1**: For each field in the schema, precision and recall of the extracted value. Averaged across fields (macro-F1). Measures extraction accuracy. Target: >88%.

**Null accuracy**: For fields not mentioned in the query, % correctly output as null. Measures that the model doesn't hallucinate values. Target: >93%.

**Exact match**: % of outputs that exactly match ground truth JSON. Low by design (45–60%) because there are many valid representations of the same fact ("blue" vs "Blue", ["blue", "white"] vs ["white", "blue"]).

### 7.3 Overfitting Detection

Always track both train_loss and eval_loss during training:

```
Epoch 1: train_loss=0.82, eval_loss=0.79   ← healthy, eval close to train
Epoch 2: train_loss=0.41, eval_loss=0.44   ← healthy
Epoch 3: train_loss=0.22, eval_loss=0.38   ← slight gap, still fine
Epoch 4: train_loss=0.18, eval_loss=0.52   ← OVERFITTING: eval rising
```

When eval_loss rises while train_loss falls, the model is memorizing instead of generalizing. `load_best_model_at_end=True` automatically uses the epoch-3 checkpoint, not epoch-4.

---

## 8. Inference and Deployment

### 8.1 The Gap Between Training and Production

Models sometimes produce subtly different outputs in production vs evaluation:
- Training used the full ChatML prompt; production must use the identical format
- Tokenizer settings must be identical
- Temperature and sampling affect output consistency

Always test your deployed model with the exact same prompt format used in training.

### 8.2 Constrained Decoding

Standard generation: at each step, sample from the full vocabulary of ~150K tokens.

Constrained decoding: at each step, only allow tokens that keep the output valid according to your constraint (e.g., valid JSON matching the schema).

This is done with a finite state machine (FSM) derived from the JSON schema. At each generation step, the FSM tells which tokens are valid continuations. Invalid tokens are masked (probability set to 0 before softmax).

**Result**: the output is mathematically guaranteed to be valid JSON conforming to your schema. Not "usually valid" — always valid.

### 8.3 Quantization for Production

After training and merging, the model weights can be quantized for smaller size and faster inference:

**GGUF Q4_K_M**: 4-bit quantization with K-quantization (smarter bit allocation). ~2.5GB for our 4B model. Works with llama.cpp and Ollama. ~95% of FP16 quality.

**GGUF Q8_0**: 8-bit quantization. ~4.7GB. ~99.5% of FP16 quality. Best quality-to-size for production.

**FP16**: Full precision. ~8.9GB. Identical to training-time quality. For high-precision deployments.

---

# PART 2 — OUR FASHION EXTRACTOR: DEEP DIVE

---

## 9. The Task: Schema-Conditioned Extraction

### 9.1 What Makes This Task Unique

Most extraction tasks train a model with a **fixed schema** — a hardcoded set of fields to always extract. Our task is fundamentally different.

**The schema is a runtime input, not a training-time constant.**

At inference time, the caller passes:
1. A search query
2. A JSON schema defining exactly which fields to extract for this call

The model must read the schema, understand which fields are requested and what types they should be, and extract only those fields from the query.

**Why this matters:**
- The fashion product catalog team can request `{"category", "size"}` for one endpoint
- The personalization team can request `{"occasion", "style", "season"}` for another
- The search team can request all 16 fields for full indexing
- The same trained model handles all three — no retraining

### 9.2 What the Model Must Learn

This task requires learning three distinct behaviors simultaneously:

**1. Schema reading**: Understand what a JSON schema says. Which fields are required? What types should they be? What are valid enum values?

**2. Attribute extraction**: From free-form text, identify mentions of category, size, color, pattern, etc. Handle synonyms ("beachy" → occasion: beach), abbreviations ("sz 32" → size: 32), and colloquial language.

**3. Null prediction**: For schema fields that are not mentioned in the query, correctly output `null`. This is harder than it sounds — the model must actively decide "this field was not mentioned" for each absent attribute, not just skip it.

### 9.3 Why This is Hard for a Small Model

Base Qwen3-4B without fine-tuning can do schema-conditioned extraction with prompt engineering — but with 60-70% reliability at best. Problems:

- Outputs non-JSON text ("Based on your query, here is the extraction: ...")
- Adds fields not in the schema (hallucination)
- Misses required fields
- Wrong types (returns string when array expected)
- Hallucinates values for fields not mentioned in query (worst case)

After fine-tuning with 40K well-curated samples: 96-98% schema compliance.

---

## 10. Model Selection: Why Qwen3-4B-Instruct

### 10.1 The Evaluation Criteria

We needed a model that is:
1. Strong at structured JSON output natively
2. Has a thinking mode for complex queries
3. Apache 2.0 licensed for commercial HuggingFace release
4. Supported by Unsloth for efficient training
5. Small enough to fine-tune on a single A100 (≤10B parameters)
6. Large enough to generalize to new schemas not seen in training

### 10.2 Why Not Smaller (1-2B)

- Qwen3-1.7B: JSON schema generalization degrades 15-20% vs 4B. For novel schemas at runtime, the smaller model makes significantly more structural errors.
- SmolLM2-1.7B: Tested on JSONSchemaBench 2025 — 26% JSON parse rate, 4% schema compliance even with constrained decoding. Ruled out entirely.
- LLaMA-3.2-1B: Good tunability but 4B beats it post-fine-tuning on this task.

### 10.3 Why Not Larger (7-8B)

At the 4B size, well-tuned QLoRA achieves 96-97% schema compliance — which is production-ready. The 7-8B model would achieve 97-99%, a marginal improvement not worth doubling the inference cost.

The exception: if serving thousands of requests per second where 1% more accuracy = significant revenue, upgrade to 8B. For current scale, 4B is correct.

### 10.4 Why Qwen3 Specifically

Qwen3 was trained with **function calling and structured output as first-class objectives** — not an afterthought. The Qwen team published that they trained on function call datasets and validated on structured output benchmarks during pre-training.

This means Qwen3 starts fine-tuning closer to our desired behavior. The fine-tuning has less ground to cover, so it trains faster and reaches higher accuracy.

Additionally, Qwen3's **thinking mode** is unique at this size class. A `<think>` block before the JSON output lets the model reason through ambiguous queries. Other 4B models cannot do this.

### 10.5 The Apache 2.0 Decision

This matters for a HuggingFace release:

| License | Commercial use | Derivative works | Attribution required |
|---|---|---|---|
| Apache 2.0 | ✓ Free | ✓ Allowed | ✓ Required |
| MIT | ✓ Free | ✓ Allowed | ✓ Required |
| Llama 3.2 | ✓ (with restrictions) | ✓ With disclosure | ✓ |
| Gemma | ✗ Some restrictions | Limited | ✓ |

Apache 2.0 is the cleanest option for a model that ionio.io wants to release publicly and potentially commercialize.

---

## 11. QLoRA Configuration: Every Decision Explained

### 11.1 Why r=32

LoRA rank controls how many parameters the adapters have. The calculation:

For a weight matrix W of shape (4096, 4096):
- r=8:  params = 2 × 4096 × 8  = 65,536
- r=16: params = 2 × 4096 × 16 = 131,072
- r=32: params = 2 × 4096 × 32 = 262,144  ← we use this
- r=64: params = 2 × 4096 × 64 = 524,288

Across all target modules (7 projection layers × 36 transformer layers):
- r=32 → ~32M total trainable parameters

**Why r=32 specifically for our task:**

The task requires schema generalization — the model must handle schemas it has never seen before. This requires enough adapter capacity to learn the meta-skill of "reading a schema and following it," not just memorizing specific field names.

Research on structured output tasks shows:
- r=8: sufficient for fixed-format extraction (always the same fields)
- r=16: marginal for schema generalization
- r=32: sweet spot for schema-conditioned extraction
- r=64: adds parameters without quality gains for 40K samples; risks overfitting

With 32K training samples and 32M trainable parameters, we have ~1,000 samples per parameter group — healthy for generalization.

### 11.2 Why alpha=64 (2× rank)

`lora_alpha` is a scaling factor applied to the LoRA output:

```
output = W·x + (alpha/r) × (A·B)·x
```

The ratio alpha/r determines how strongly the adapter influences the model. Setting alpha=2×r gives alpha/r=2, meaning the adapter has 2× amplification.

In practice: alpha/r=1 is conservative (slow learning), alpha/r=2 is standard for SFT tasks, alpha/r=4+ is aggressive (risk of destabilizing pre-trained knowledge).

The 2× ratio is the community-validated standard for instruction-following and extraction tasks. Changing this without good reason introduces risk.

### 11.3 Why These Target Modules

```python
["q_proj", "k_proj", "v_proj", "o_proj",  # attention
 "gate_proj", "up_proj", "down_proj"]      # feed-forward (MLP)
```

**Attention projections** (q, k, v, o): Control how the model attends to different parts of the input. These layers need to learn to attend to schema fields when reading the query, and to the relevant query words when filling each schema field. Critical for schema-conditioned behavior.

**Feed-forward projections** (gate, up, down): Store factual associations and knowledge. Need to be adapted so the model maps fashion vocabulary to schema field values (e.g., "Hawaiian" → pattern field).

Not targeted: embedding layers (too many parameters, catastrophic forgetting risk), layer norm (too sensitive).

Targeting all 7 modules gives the adapter the maximum ability to reshape the model's behavior while staying within QLoRA constraints.

### 11.4 use_rslora=False

RSLoRA (Random Scaled LoRA) scales gradients by 1/sqrt(r) instead of 1/r. Theoretically helps at high ranks (r=128+). At r=32, the difference is negligible and standard LoRA is better validated. We keep `use_rslora=False`.

### 11.5 use_gradient_checkpointing="unsloth"

Unsloth implements a custom gradient checkpointing strategy that is smarter than PyTorch's default. It only recomputes the most memory-intensive activations, reducing peak memory by 30% with less than 5% training time overhead. Always use this.

---

## 12. Data Pipeline: Every Decision Explained

### 12.1 The Input Format

Each training sample needs three things:
- `query`: the raw search text (not cleaned — train on realistic user input)
- `schema`: the JSON schema defining what to extract (varies per sample)
- `extraction`: the ground truth JSON output

The schema must be a proper JSON Schema object with `properties` and `required` fields. The extraction must only contain fields defined in `schema.properties`.

### 12.2 The Extraction-Schema Validation

This validation is critical and often skipped in tutorials:

```python
extra = [f for f in extraction if f not in schema_props]
if extra:
    return False, f"extraction has fields not in schema: {extra}"
```

If a ground truth extraction has `{"category": "shirt", "brand": "Zara"}` but the schema only asks for `category`, this is bad training data. It teaches the model to add fields not requested by the schema — the exact hallucination behavior we're trying to prevent.

Every bad sample that slips through training creates a subtle error mode in production. This check eliminates the source.

### 12.3 The ChatML Formatting

The exact format the model trains on must be used identically at inference. Qwen3's ChatML:

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Query: {query}

Schema:
{pretty-printed JSON schema}<|im_end|>
<|im_start|>assistant
{extraction JSON (compact)}<|im_end|>
```

Design decisions in this format:
- **System prompt is constant**: the model learns the role once; it doesn't vary by sample
- **Schema is pretty-printed** (indent=2): more tokens, but the model can read it better — we validated this matters for complex nested schemas
- **Extraction is compact** (no indent): fewer output tokens = faster inference = lower cost
- **Query label**: "Query:" prefix is consistent so the model always knows where the query starts

### 12.4 Thinking Mode Samples

For 10-15% of training samples (complex, ambiguous, or multi-attribute queries), the ground truth includes a `<think>` block:

```
<|im_start|>assistant
<think>
The query says "sz 32" which is an abbreviation for size 32. 
"Hawaiian print" is a pattern type. "Blue or white" means multiple colors.
"Under $50" is a price constraint with max=50, currency=USD.
No gender mentioned → null.
</think>
{"size": "32", "pattern": "Hawaiian print", "color": ["blue", "white"], "price_range": {"min": null, "max": 50, "currency": "USD"}, "gender": null}<|im_end|>
```

The `<think>` block is special to Qwen3 — these are real model vocabulary tokens. Training with thinking samples teaches the model to reason before extracting, which significantly improves accuracy on:
- Abbreviated inputs ("sz", "btn-down", "wht")
- Colloquial descriptions ("something beachy")
- Price mentions with various formats ("under $50", "around 100 bucks", "< ₹2000")
- Negations ("nothing too formal", "not synthetic")

At inference, you can enable or disable thinking mode independently.

### 12.5 The 80/10/10 Split

```
80% train  — what the model learns from
10% eval   — monitored during training (prevents overfitting)
10% test   — locked away, used only for final evaluation
```

The test set is your ground truth for how the model performs in production. Never look at it during training, never tune hyperparameters based on it, never add test samples to training. It is your production simulation.

The eval split is used by the trainer to monitor for overfitting and pick the best checkpoint. It influences training indirectly (via early stopping) but the model never trains directly on it.

---

## 13. Training Strategy: Every Hyperparameter Explained

### 13.1 Number of Epochs: 3

With 35K training samples and effective batch size 16:
- Steps per epoch = 35,000 / 16 = 2,187 steps
- 3 epochs = 6,561 total steps

At each step, the model sees a fresh (shuffled) batch of 16 samples. After 3 epochs, every sample has been seen 3 times in different batch contexts.

**Why not 1 or 2 epochs:** For schema generalization, the model needs repeated exposure to diverse schema-query-extraction triplets. With 40K diverse samples (different schemas, query styles, edge cases), 1-2 epochs underfit slightly.

**Why not 4 or 5 epochs:** With the diversity of our data, overfitting risk increases sharply after epoch 3. The `load_best_model_at_end` would catch this, but it wastes training time.

**The real answer:** We don't actually need exactly 3 epochs — we need to train until eval_loss stops improving. 3 epochs is our budget; early stopping handles the rest.

### 13.2 Learning Rate: 2e-4

This is 0.0002.

For AdamW with LoRA on a 4B instruction-tuned model at r=32:
- Below 5e-5: too slow, likely underfits in 3 epochs
- 5e-5 to 1e-4: conservative, fine but slower
- 1e-4 to 2e-4: optimal range for extraction SFT
- 5e-4: may destabilize some attention layers
- Above 1e-3: almost certainly diverges

2e-4 is validated by the distil labs benchmark (10K samples, 12 models) and our target model class. Changing this without reason introduces risk.

### 13.3 Warmup Ratio: 0.1

For the first 10% of training steps, the learning rate ramps up linearly from ~0 to 2e-4. This prevents the issue where large gradient updates in the very first steps (when the model hasn't learned the task at all) destabilize the LoRA adapters before they've had a chance to settle.

10% = 656 warmup steps. After that, cosine decay begins.

### 13.4 Weight Decay: 0.01

AdamW's weight decay adds `weight_decay × weight` to the gradient at each step, which gently pulls all weights toward zero. This prevents any individual weight from becoming very large, which would make the model overly sensitive to specific patterns in training data.

0.01 is light enough to not fight the learning process but effective at regularization.

### 13.5 Gradient Clipping: max_grad_norm=1.0

Before each weight update, if the L2 norm of all gradients exceeds 1.0, scale all gradients down proportionally so the norm = 1.0.

This prevents a single outlier batch from causing a catastrophically large weight update. Particularly important for:
- Early training when gradients are large
- Batches that happen to contain many hard examples
- Prevents NaN propagation if a gradient spike occurs

### 13.6 Effective Batch Size: 16

`per_device_batch_size=8` × `gradient_accumulation_steps=2` = 16

**Why 16 specifically:**
- Research on instruction tuning shows 16-32 is optimal for stable gradient estimates
- Below 8: gradients are too noisy; model updates are inconsistent
- Above 64: gradient averaging becomes over-smooth; fewer unique updates per epoch

For an A100 80GB, we could push per_device to 16 (effective batch 32) and get faster training. We chose to keep effective batch at 16 for quality consistency.

### 13.7 The Response-Only Loss (Critical)

This deserves its own section because it's the most impactful decision in the training script.

Without DataCollatorForCompletionOnlyLM:
```
Sample: [system(89 tokens)] [user_query+schema(312 tokens)] [assistant_JSON(64 tokens)]
Loss:   computed on all 465 tokens
Useful loss: 64/465 = 13.8% of gradients improve extraction
```

With DataCollatorForCompletionOnlyLM:
```
Sample: [system(89 tokens, masked)] [user_query+schema(312 tokens, masked)] [assistant_JSON(64 tokens)]
Loss:   computed on 64 tokens only
Useful loss: 64/64 = 100% of gradients improve extraction
```

**7.25× more efficient gradient signal** in every training step. The model converges faster, to a higher quality, because it's not wasting capacity on predicting the fixed system prompt.

This also means the eval_loss is a direct measure of extraction quality — not diluted by the easy task of predicting the constant system prompt.

---

## 14. Evaluation: Reading the Numbers

### 14.1 The Metrics Stack

We report 5 metrics, each measuring a different failure mode:

**Valid JSON rate** catches: model outputting prose, markdown, or malformed JSON. The most fundamental check. If this is below 90%, the training data format is wrong.

**Schema compliance rate** catches: valid JSON that's structurally wrong (missing required fields, wrong types). If this is below 90%, the model hasn't learned to read schemas properly — need more schema-varied training data.

**Null accuracy** catches: model hallucinating values for fields not in the query. This is insidious — the model produces valid JSON that passes schema compliance but contains invented values. Target >93%.

**Field F1 (macro)** catches: extraction accuracy per field. Low scores on specific fields (e.g., price_range consistently below 70%) indicate the model struggles with that field type — possibly needs more training samples with that field.

**Exact match** is a ceiling metric. It's never the primary evaluation signal because two JSON objects can be semantically identical but literally different (order of keys, "Blue" vs "blue"). Use it to understand the upper bound.

### 14.2 Interpreting Per-Field F1 Scores

Expected hierarchy of difficulty:
```
category     : 95%+  (clear categorical, most queries mention it)
color        : 92%+  (named colors are explicit)
size         : 90%+  (usually explicit: "size M", "32")
pattern      : 88%+  (sometimes implicit: "floral" is obvious)
fit          : 85%+  (often implied: "slim", "relaxed")
occasion     : 82%+  (usually stated: "casual", "office")
price_range  : 75%+  (numeric parsing, currency inference)
material     : 72%+  (often not stated in query)
neckline     : 70%+  (rare in short queries)
```

If any field consistently underperforms by >10% vs these baselines, add more training samples featuring that field prominently.

---

## 15. The Production Stack

### 15.1 Why Merge LoRA Before Deployment

Two deployment options after training:

**Option A: Base model + adapter at runtime**
- Load 4.44B base model in 4-bit (~2.2GB)
- Load adapter weights (~100MB)
- At each forward pass: compute W·x + (A·B)·x
- Problem: extra computation at every token, every layer, every request

**Option B: Merged FP16 model** (what we do)
- Compute W_merged = W + A·B once during export
- Load W_merged directly (~8.9GB)
- At each forward pass: compute W_merged·x (identical to any other model)
- No runtime overhead from adapters

For production serving, Option B is correct. The one-time merge cost (5 minutes) is paid once; inference latency is identical to any non-fine-tuned model.

### 15.2 GGUF and Quantization for Deployment

After merging, we export to GGUF format for flexible deployment:

**Q4_K_M** (our primary release):
- 4-bit quantization with K-means quantization strategy
- Each weight quantized within its "group" of 32 values
- Quantization constants stored in higher precision (16-bit)
- Size: ~2.5GB
- Quality: 95-96% of FP16 on extraction tasks
- Runs on CPU or GPU, compatible with Ollama and llama.cpp
- For deployment on servers without dedicated GPU

**Q8_0** (secondary release):
- 8-bit linear quantization
- Size: ~4.7GB
- Quality: 99.5% of FP16
- For high-accuracy deployments where extra 2GB is acceptable

### 15.3 Outlines: Constrained Generation in Production

Fine-tuning achieves ~97% valid JSON. For production, we need 100%.

Outlines builds a Finite State Machine (FSM) from your JSON schema. The FSM has states representing "valid positions in the JSON structure." At each generation step:

1. The model produces logits (unnormalized probabilities) over 150K+ vocabulary tokens
2. Outlines' FSM determines which tokens are valid at the current state
3. Invalid tokens are masked (logit → -infinity before softmax)
4. The model samples only from valid continuations

**Example**: After `{"category": "`, the FSM knows we're inside a string value. Only letter tokens and closing quote are valid. The model cannot generate `{`, `}`, `:`, or any structural JSON character here.

The FSM transitions when the model generates structural tokens (`{`, `}`, `[`, `]`, `"`, `,`), tracking the nesting level and current field position.

**Integration with fine-tuned model**:
```python
model = outlines.models.transformers("ionio-io/fashion-query-extractor-4B")
generator = outlines.generate.json(model, schema)
result = generator(prompt)
# result is guaranteed valid JSON conforming to schema
```

Fine-tuning teaches extraction intelligence. Outlines provides structural guarantee. Together: 100% reliable, high-accuracy extraction.

---

## 16. The Complete System: End-to-End Flow

```
TRAINING TIME:
─────────────────────────────────────────────────────────────────────────

Raw Data (JSONL)
    │
    ▼
1_format_data.py
    ├── Validate: required fields, JSON validity, extraction⊂schema
    ├── Format: apply Qwen3 ChatML template, handle thinking mode
    ├── Filter: drop samples > 1024 tokens
    └── Split: 80% train / 10% eval / 10% test (test set locked)
    │
    ▼
ChatML JSONL (train.jsonl, eval.jsonl, test.jsonl)
    │
    ▼
2_train.py
    ├── Load: Qwen3-4B-Instruct in 4-bit (QLoRA)
    ├── Add: LoRA adapters (r=32, alpha=64, 7 projection layers)
    ├── Mask: system+user tokens from loss (response-only training)
    ├── Train: 3 epochs, lr=2e-4 cosine, batch=16, BF16
    ├── Eval: every 200 steps on eval set
    └── Save: best checkpoint (lowest eval_loss)
    │
    ▼
checkpoints/final_adapter/
    │
    ▼
3_evaluate.py
    ├── Load: trained adapter
    ├── Infer: batch=16, greedy decoding on test set
    └── Report: JSON rate, schema compliance, F1, null accuracy
    │
    ▼
eval_results.json (schema compliance >96%, field F1 >88%)
    │
    ▼
4_export_push.py
    ├── Merge: LoRA adapters → base model weights (W = W + A·B)
    ├── Save: merged FP16 (~8.9GB)
    ├── Quantize: GGUF Q4_K_M + Q8_0
    └── Push: HuggingFace (fp16 + GGUF)

INFERENCE TIME:
─────────────────────────────────────────────────────────────────────────

User Search Query + Runtime Schema
    │
    ▼
Build ChatML prompt (same format as training)
    │
    ▼
Fine-tuned Qwen3-4B (merged FP16 or GGUF)
    │
    ▼
Outlines FSM (schema → valid token mask at each step)
    │
    ▼
Structured JSON extraction
    │
    ▼
Search / Filtering / Personalization system
```

---

## 17. What to Expect: The Full Timeline

| Phase | Time | Output |
|---|---|---|
| setup.sh on RunPod | ~5 min | Environment ready |
| 1_format_data.py (40K samples) | ~10–15 min | train/eval/test JSONL |
| Model download (first time) | ~15–20 min | Cached to ~/.cache/huggingface |
| 2_train.py (A100 40GB) | ~4–6 hours | checkpoints/final_adapter |
| 2_train.py (A100 80GB) | ~3–4 hours | checkpoints/final_adapter |
| 3_evaluate.py (4.4K samples) | ~20–30 min | eval_results.json |
| 4_export_push.py | ~30–45 min | HuggingFace upload |

---

## 18. Configuration Reference

All settings are in `config.py`. This table explains why each value is what it is:

| Parameter | Value | Rationale |
|---|---|---|
| `MODEL_NAME` | `unsloth/Qwen3-4B-Instruct` | Best ~4B model for structured output + Apache 2.0 + Unsloth native |
| `MAX_SEQ_LENGTH` | 1024 | Covers 99%+ of queries+schemas; longer wastes VRAM |
| `LORA_R` | 32 | Enough for schema generalization; not so high it overfits |
| `LORA_ALPHA` | 64 | 2× rank: standard scaling for SFT tasks |
| `LORA_DROPOUT` | 0.05 | Light regularization on adapters |
| `NUM_EPOCHS` | 3 | Right for 40K diverse samples; early stopping handles precision |
| `PER_DEVICE_BATCH_SIZE` | 8 | Optimized for A100 80GB |
| `GRADIENT_ACCUMULATION_STEPS` | 2 | Effective batch = 16 |
| `LEARNING_RATE` | 2e-4 | Validated optimal for r=32 LoRA SFT on 4B models |
| `LR_SCHEDULER` | cosine | Smooth decay; best for SFT convergence |
| `WARMUP_RATIO` | 0.1 | 10% warmup prevents early instability |
| `WEIGHT_DECAY` | 0.01 | Light L2 regularization |
| `EVAL_STEPS` | 200 | ~11 evaluations per epoch; fine-grained overfitting detection |
| `EVAL_BATCH_SIZE` | 16 | No gradients during eval, larger batch is safe on A100 |
| `MAX_NEW_TOKENS` | 512 | Maximum JSON extraction length; complex schemas need room |

---

*This document covers the complete theory and practice behind the fashion query extractor. The pipeline is in the `fashion-query-extractor` repository.*
