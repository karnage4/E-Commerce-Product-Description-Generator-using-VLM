"""
Centralised configuration for all Phase 4 model experiments.

Execution strategy:
  - Fine-tuning:  runs on Google Colab (free T4 GPU, 15 GB VRAM)
  - API baseline: runs locally  (Gemini API — free tier)
  - Evaluation:   runs locally  (CPU-only, no model loading)
"""

from pathlib import Path

# ── Paths (relative to project root) ──────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]   # .../Project/
DATA_DIR       = PROJECT_ROOT / "data" / "data" / "processed"
METADATA_FILE  = DATA_DIR / "metadata" / "listings_final.jsonl"
IMAGES_DIR     = DATA_DIR / "images"
SPLITS_DIR     = DATA_DIR / "splits"

MODELS_DIR     = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
RESULTS_DIR    = MODELS_DIR / "results"

# ── Dataset split files ────────────────────────────────────────────────────────
TRAIN_SPLIT = SPLITS_DIR / "train.txt"
VAL_SPLIT   = SPLITS_DIR / "val.txt"
TEST_SPLIT  = SPLITS_DIR / "test.txt"

# ── Training hyperparameters (used in Colab notebooks) ────────────────────────
BLIP_CONFIG = {
    "model_name":        "Salesforce/blip-image-captioning-base",
    "learning_rate":     1e-5,
    "batch_size":        8,
    "num_epochs":        5,
    "weight_decay":      0.01,
    "max_input_length":  128,
    "max_target_length": 200,
    "warmup_steps":      100,
    "fp16":              True,
    "save_every_epoch":  True,
    "checkpoint_dir":    "checkpoints/blip",
    # Set to an int to train on a subset — useful for quick hyperparameter runs.
    # e.g. subset_samples=200 → ~15 min/run on Colab T4 instead of ~90 min.
    "subset_samples":    None,
}

CLIP_GPT2_CONFIG = {
    "clip_model":        "openai/clip-vit-base-patch32",
    "gpt2_model":        "gpt2",
    "prefix_length":     10,
    "learning_rate":     2e-5,
    "batch_size":        8,
    "num_epochs":        5,
    "weight_decay":      0.01,
    "max_target_length": 200,
    "fp16":              True,
    "checkpoint_dir":    "checkpoints/clip_gpt2",
    "subset_samples":    None,
}

# ── Hyperparameter search spaces (used by models/hparam_sweep.py) ─────────────
INFERENCE_SWEEP = {
    "num_beams":            [2, 4, 6],
    "no_repeat_ngram_size": [2, 3, 4],
    "max_new_tokens":       [100, 150, 200],
}

TRAINING_SWEEP = {
    # Run each combo on subset_samples=200 to find best LR before full training
    "blip": {
        "learning_rate": [5e-6, 1e-5, 3e-5],
        "warmup_steps":  [50, 100, 200],
        "subset_samples": 200,
        "num_epochs":    3,
    },
    "clip_gpt2": {
        "learning_rate": [1e-5, 2e-5, 5e-5],
        "prefix_length": [5, 10, 20],
        "subset_samples": 200,
        "num_epochs":    3,
    },
}

# ── Gemini API config (free tier — runs locally) ───────────────────────────────
GEMINI_CONFIG = {
    "model":            "gemini-1.5-flash",   # free tier, multimodal
    "max_output_tokens": 300,
    "temperature":      0.7,
    "top_p":            0.9,
    "results_file":     "results/gemini_baseline.jsonl",
}

# ── Metadata prompt templates ──────────────────────────────────────────────────
def build_stage1_prompt(record: dict) -> str:
    """
    Minimal prompt for Stage 1 VLM — just category so the model is forced
    to extract color, material, design, etc. from the image itself.
    """
    category = record.get("category", "")
    subcategory = record.get("subcategory", "")
    parts = []
    if category:    parts.append(f"Category: {category}")
    if subcategory: parts.append(f"Subcategory: {subcategory}")
    return ". ".join(parts)


def build_metadata_prompt(record: dict) -> str:
    """
    Full metadata prompt for Stage 2 — all available structured fields.
    Used as model input alongside the product image.
    """
    parts = []
    if record.get("item_name"):   parts.append(f"Product: {record['item_name']}")
    if record.get("brand"):       parts.append(f"Brand: {record['brand']}")
    if record.get("category"):    parts.append(f"Category: {record['category']}")
    if record.get("subcategory"): parts.append(f"Subcategory: {record['subcategory']}")
    if record.get("price_pkr"):   parts.append(f"Price: PKR {record['price_pkr']:.0f}")
    if record.get("rating"):      parts.append(f"Rating: {record['rating']:.1f}/5")

    attrs = record.get("attributes") or {}
    for k, v in list(attrs.items())[:6]:   # limit to 6 attributes
        if k and v:
            parts.append(f"{k}: {v}")

    return ". ".join(parts)


# ── Inference prompt for API models ───────────────────────────────────────────
INFERENCE_PROMPT_TEMPLATE = """\
You are a professional e-commerce copywriter for Daraz Pakistan.
Given the product image and the following metadata, write an informative,
engaging, and persuasive product description in 3-5 sentences.
Do NOT mention price. Use natural English.

Metadata:
{metadata}

Description:"""
