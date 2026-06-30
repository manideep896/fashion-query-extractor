#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — One-shot RunPod environment setup
# Run once at the start of your pod session:
#   bash setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "========================================================"
echo "  Fashion Extractor — RunPod Environment Setup"
echo "========================================================"

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "[1/5] GPU status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── pip upgrade ───────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip --quiet

# ── Unsloth ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing Unsloth (latest from git)..."
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" --quiet

# ── Training stack ────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Installing training dependencies..."
pip install \
    "transformers>=4.43.0" \
    "trl>=0.9.0" \
    "peft>=0.12.0" \
    "accelerate>=0.33.0" \
    "bitsandbytes>=0.43.0" \
    "datasets>=2.20.0" \
    --quiet

# ── Eval + export stack ───────────────────────────────────────────────────────
echo ""
echo "[5/5] Installing evaluation and export dependencies..."
pip install \
    "scikit-learn>=1.5.0" \
    "numpy>=1.26.0" \
    "pandas>=2.2.0" \
    "tqdm>=4.66.0" \
    "outlines>=0.0.46" \
    "huggingface_hub>=0.24.0" \
    --quiet

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Verification"
echo "========================================================"
python - <<'PYEOF'
import sys

packages = {
    "torch":            "torch",
    "unsloth":          "unsloth",
    "transformers":     "transformers",
    "trl":              "trl",
    "peft":             "peft",
    "datasets":         "datasets",
    "outlines":         "outlines",
    "huggingface_hub":  "huggingface_hub",
    "bitsandbytes":     "bitsandbytes",
}

all_ok = True
for display, pkg in packages.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  ✓  {display:<20} {ver}")
    except ImportError:
        print(f"  ✗  {display:<20} NOT INSTALLED")
        all_ok = False

import torch
print(f"\n  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  VRAM:            {vram:.1f} GB")

if not all_ok:
    print("\n  Some packages failed. Re-run setup.sh.")
    sys.exit(1)
else:
    print("\n  All packages OK. Ready to train.")
PYEOF

echo ""
echo "========================================================"
echo "  Setup complete. Next steps:"
echo "    1. Put your data file in:  ./data/raw.jsonl"
echo "    2. python 1_format_data.py --input ./data/raw.jsonl"
echo "    3. python 2_train.py"
echo "    4. python 3_evaluate.py"
echo "    5. python 4_export_push.py --hf_token YOUR_TOKEN"
echo "========================================================"
