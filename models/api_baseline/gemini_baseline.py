"""
Gemini API Baseline — runs locally on 8GB RAM (no GPU needed).

Uses Google Gemini 1.5 Flash (free tier: 15 req/min, 1M tokens/day)
to generate product descriptions from image + metadata.

This is used as:
  1. A strong zero-shot baseline (before any fine-tuning)
  2. A comparison against fine-tuned BLIP/CLIP-GPT2

Setup:
  1. Get a free API key: https://aistudio.google.com/app/apikey
  2. Set env variable:  set GEMINI_API_KEY=your_key_here
  3. Run: python -m models.api_baseline.gemini_baseline

Install:
  pip install google-generativeai pillow tqdm
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import PIL.Image
import google.generativeai as genai
from tqdm import tqdm

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT,
    GEMINI_CONFIG, INFERENCE_PROMPT_TEMPLATE,
    RESULTS_DIR, build_metadata_prompt
)
from models.shared.metrics import compute_all_metrics, print_metrics_table, save_metrics


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup_gemini() -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "\n[!] GEMINI_API_KEY not set.\n"
            "    1. Go to https://aistudio.google.com/app/apikey\n"
            "    2. Create a free API key\n"
            "    3. Run:  set GEMINI_API_KEY=your_key_here\n"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=GEMINI_CONFIG["model"],
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=GEMINI_CONFIG["max_output_tokens"],
            temperature=GEMINI_CONFIG["temperature"],
            top_p=GEMINI_CONFIG["top_p"],
        ),
    )
    print(f"  Gemini model: {GEMINI_CONFIG['model']} (free tier)")
    return model


# ── Load test records ─────────────────────────────────────────────────────────

def load_test_records(max_samples: int = 200) -> list[dict]:
    """Load test split records that have images and descriptions."""
    test_ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").strip().splitlines())

    records = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("item_id") not in test_ids:
                continue
            if not rec.get("images") or not rec.get("description", "").strip():
                continue
            records.append(rec)
            if len(records) >= max_samples:
                break

    print(f"  Loaded {len(records)} test records")
    return records


# ── Load image for a record ───────────────────────────────────────────────────

def load_pil_image(rec: dict) -> PIL.Image.Image | None:
    for rel_path in rec.get("images", []):
        abs_path = IMAGES_DIR.parent / rel_path
        if abs_path.exists():
            try:
                return PIL.Image.open(abs_path).convert("RGB")
            except Exception:
                continue
    return None


# ── Single inference call ─────────────────────────────────────────────────────

def generate_description(
    model: genai.GenerativeModel,
    rec: dict,
) -> str:
    """Generate a product description for one record via Gemini API."""
    image = load_pil_image(rec)
    metadata_str = build_metadata_prompt(rec)
    prompt = INFERENCE_PROMPT_TEMPLATE.format(metadata=metadata_str)

    if image is not None:
        # Multimodal: image + text
        response = model.generate_content([image, prompt])
    else:
        # Text-only fallback if image unavailable
        response = model.generate_content(prompt)

    return response.text.strip()


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_gemini_baseline(max_samples: int = 200, delay_sec: float = 4.1) -> None:
    """
    Run Gemini API over the test set and compute metrics.

    Args:
        max_samples: How many test products to evaluate (free tier: 15 req/min)
        delay_sec:   Seconds to wait between requests (15 req/min = 4s gap)
    """
    model   = setup_gemini()
    records = load_test_records(max_samples=max_samples)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / GEMINI_CONFIG["results_file"].split("/")[-1]

    # Load already-processed IDs for resumption
    done_ids: set[str] = set()
    hypotheses: list[str] = []
    references: list[str] = []

    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done_ids.add(entry["item_id"])
                    hypotheses.append(entry["generated"])
                    references.append(entry["reference"])
                except Exception:
                    pass
        print(f"  Resuming — {len(done_ids)} already done")

    with open(results_file, "a", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="Gemini inference", unit="product"):
            if rec["item_id"] in done_ids:
                continue

            try:
                generated = generate_description(model, rec)
            except Exception as e:
                print(f"\n  [!] API error for {rec['item_id']}: {e}")
                time.sleep(10)
                continue

            reference = rec["description"].strip()
            entry = {
                "item_id":   rec["item_id"],
                "category":  rec.get("category", ""),
                "generated": generated,
                "reference": reference,
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()

            hypotheses.append(generated)
            references.append(reference)

            time.sleep(delay_sec)   # respect free-tier rate limit

    # ── Compute metrics ───────────────────────────────────────────────────────
    if hypotheses:
        print(f"\n  Evaluating {len(hypotheses)} samples...")
        metrics = compute_all_metrics(hypotheses, references)
        results = {"Gemini-1.5-Flash (zero-shot)": metrics}
        print_metrics_table(results)
        save_metrics(metrics, str(RESULTS_DIR / "gemini_metrics.json"))
    else:
        print("  [!] No results to evaluate.")


# ── Metadata-only baseline (no model needed) ───────────────────────────────────

def run_metadata_baseline(max_samples: int = 200) -> None:
    """
    Baseline 1: Use raw metadata string as the 'generated' description.
    No API calls needed — runs fully locally.
    """
    records = load_test_records(max_samples=max_samples)
    hypotheses = [build_metadata_prompt(r) for r in records]
    references  = [r["description"].strip() for r in records]

    print(f"\n  [Baseline 1: Metadata-Only] {len(records)} samples")
    metrics = compute_all_metrics(hypotheses, references)
    results = {"Metadata-Only (Baseline 1)": metrics}
    print_metrics_table(results)
    save_metrics(metrics, str(RESULTS_DIR / "metadata_baseline_metrics.json"))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini API baseline inference")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run Metadata-Only baseline (no API calls)")
    parser.add_argument("--max-samples", type=int, default=150,
                        help="Max test samples to evaluate (default: 150)")
    args = parser.parse_args()

    if args.baseline_only:
        run_metadata_baseline(max_samples=args.max_samples)
    else:
        run_metadata_baseline(max_samples=args.max_samples)
        print("\n" + "=" * 60)
        run_gemini_baseline(max_samples=args.max_samples)
