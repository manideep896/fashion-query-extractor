# ─────────────────────────────────────────────────────────────────────────────
# config.py  —  Single source of truth for all training settings
# ─────────────────────────────────────────────────────────────────────────────

# ── Model ─────────────────────────────────────────────────────────────────────
# Try "unsloth/Qwen3-4B-Instruct-2507" first if it exists on HuggingFace.
# Falls back to "unsloth/Qwen3-4B-Instruct" which is always available.
MODEL_NAME      = "unsloth/Qwen3-4B-Instruct"
MAX_SEQ_LENGTH  = 1024

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R          = 32
LORA_ALPHA      = 64       # always 2 × r
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS                  = 3
PER_DEVICE_BATCH_SIZE       = 8     # A100 80 GB → safe at 8; bump to 16 if you want faster
GRADIENT_ACCUMULATION_STEPS = 2     # effective batch = 8 × 2 = 16
LEARNING_RATE               = 2e-4
LR_SCHEDULER                = "cosine"
WARMUP_RATIO                = 0.1
WEIGHT_DECAY                = 0.01
EVAL_STEPS                  = 200
SAVE_STEPS                  = 200
LOGGING_STEPS               = 10
SEED                        = 42

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
EVAL_RATIO  = 0.10
TEST_RATIO  = 0.10

# ── Evaluation inference ───────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 16      # no gradients → larger batch is fine on A100 80 GB
MAX_NEW_TOKENS  = 512

# ── Paths  (relative; scripts resolve from their own __file__ dir) ─────────────
DATA_DIR          = "./data"
CHECKPOINT_DIR    = "./checkpoints"
MERGED_DIR        = "./merged_model"
GGUF_DIR          = "./gguf"
EVAL_RESULTS_PATH = "./eval_results.json"

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_MODEL_ID = "ionio-io/fashion-query-extractor-4B"
HF_GGUF_ID  = "ionio-io/fashion-query-extractor-4B-GGUF"

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a structured extraction model. "
    "Given a search query and a JSON schema, extract the requested fields. "
    "Output only valid JSON conforming to the schema. "
    "Use null for fields not mentioned in the query. "
    "Do not add fields that are not in the schema."
)
