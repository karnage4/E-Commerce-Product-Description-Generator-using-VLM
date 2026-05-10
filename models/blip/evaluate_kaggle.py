"""
BLIP Evaluation — Kaggle, with Phase 1 generation fixes.

Improvements over evaluate_colab.py
───────────────────────────────────
- Generation: no_repeat_ngram_size=4, repetition_penalty=1.3, length_penalty=1.1
  These cut the repetition loops we saw in earlier outputs.
- Price hallucination patch: any "PKR <number>" / "Rs. <number>" in the output
  is replaced with the correct metadata price (post-processing band-aid).
- Reads the SAME unified metadata file used in training (listings.jsonl) and
  evaluates against the ORIGINAL Daraz description in `description_original`
  if present, else `description`. We never evaluate against the augmented
  field — that would be evaluating the model against itself.

How to run
──────────
  Cell 1:  !pip install -q transformers==4.40.0 nltk rouge-score
           import nltk; nltk.download("punkt"); nltk.download("wordnet")
  Cell 2:  paste this entire file. Edit DATASET_SLUG if needed.
"""

import json
import re
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


# ── Paths (Kaggle layout) ─────────────────────────────────────────────────────
DATASET_SLUG    = "daraz-vlm-augmented"

DATA_ROOT       = Path(f"/kaggle/input/{DATASET_SLUG}")
METADATA_FILE   = DATA_ROOT / "metadata" / "listings.jsonl"
TEST_SPLIT_FILE = DATA_ROOT / "splits" / "test.txt"
RESULTS_DIR     = Path("/kaggle/working/results")

# Use the checkpoint produced by train_kaggle.py
CHECKPOINT_DIR  = Path("/kaggle/working/checkpoints/blip/best_model")

MAX_SAMPLES = 150


# ── Generation hyperparameters (Phase 1 fix) ──────────────────────────────────
GEN_KWARGS = dict(
    max_new_tokens       = 150,
    num_beams            = 4,
    no_repeat_ngram_size = 4,    # was 3
    repetition_penalty   = 1.3,  # new — penalises repeating any token
    length_penalty       = 1.1,  # new — gentle nudge toward complete outputs
    early_stopping       = True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_metadata_prompt(rec: dict) -> str:
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    for k, v in list((rec.get("attributes") or {}).items())[:3]:
        if k and v:
            parts.append(f"{k}: {v}")
    return ". ".join(parts)


def load_test_records(max_samples: int) -> list[dict]:
    test_ids = set(TEST_SPLIT_FILE.read_text(encoding="utf-8").strip().splitlines())
    out = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except Exception:
                continue
            if rec.get("item_id") not in test_ids:
                continue
            if not rec.get("images"):
                continue
            # Reference description = the ORIGINAL human-written one, never augmented
            ref = (rec.get("description_original") or rec.get("description") or "").strip()
            if not ref:
                continue
            rec["__reference"] = ref
            out.append(rec)
            if len(out) >= max_samples:
                break
    print(f"  Loaded {len(out)} test records")
    return out


def load_image(rec: dict) -> Image.Image:
    for rel in rec.get("images", []):
        rel = rel.replace("\\", "/")
        p = DATA_ROOT / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


# ── Phase 1 post-processing ───────────────────────────────────────────────────

PRICE_PATTERN = re.compile(r"(?:PKR|Rs\.?|₨)\s*[\d,]+", flags=re.IGNORECASE)

def fix_price_hallucination(text: str, rec: dict) -> str:
    """Replace any hallucinated price token with the metadata price."""
    if not rec.get("price_pkr"):
        return text
    correct = f"PKR {rec['price_pkr']:.0f}"
    return PRICE_PATTERN.sub(correct, text)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(hyps: list[str], refs: list[str]) -> dict:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rouge_lib

    smooth = SmoothingFunction().method4
    refs_tok = [[r.split()] for r in refs]
    hyps_tok = [h.split() for h in hyps]

    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25, .25, .25, .25), smoothing_function=smooth)

    scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(refs, hyps)) / len(refs)
    meteor  = sum(meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)) / len(refs)

    return {
        "BLEU-1":  round(bleu1  * 100, 2),
        "BLEU-4":  round(bleu4  * 100, 2),
        "ROUGE-L": round(rouge_l * 100, 2),
        "METEOR":  round(meteor  * 100, 2),
    }


def print_table(model_name: str, metrics: dict):
    header = f"{'Model':<35}" + "".join(f"{k:>10}" for k in metrics)
    row    = f"{model_name:<35}" + "".join(f"{v:>10.2f}" for v in metrics.values())
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    print(row)
    print("=" * len(header))


# ── Main ──────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
print(f"Checkpoint: {CHECKPOINT_DIR}")
print(f"Exists:     {CHECKPOINT_DIR.exists()}")

print(f"\nLoading BLIP from {CHECKPOINT_DIR} ...")
processor = BlipProcessor.from_pretrained(str(CHECKPOINT_DIR))
model = BlipForConditionalGeneration.from_pretrained(
    str(CHECKPOINT_DIR),
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
).to(device)
model.eval()

records = load_test_records(MAX_SAMPLES)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_file = RESULTS_DIR / "blip_results.jsonl"

hyps: list[str] = []
refs: list[str] = []

t0 = time.time()
with open(results_file, "w", encoding="utf-8") as out_f:
    for rec in tqdm(records, desc="BLIP inference", unit="item"):
        image  = load_image(rec)
        prompt = build_metadata_prompt(rec)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    output_ids = model.generate(**inputs, **GEN_KWARGS)
            else:
                output_ids = model.generate(**inputs, **GEN_KWARGS)

        raw       = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        cleaned   = fix_price_hallucination(raw, rec)
        reference = rec["__reference"]

        out_f.write(json.dumps({
            "item_id":   rec["item_id"],
            "category":  rec.get("category", ""),
            "raw":       raw,
            "generated": cleaned,
            "reference": reference,
        }, ensure_ascii=False) + "\n")
        out_f.flush()

        hyps.append(cleaned)
        refs.append(reference)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min  ({elapsed/max(1,len(hyps)):.1f}s/sample)")

metrics = compute_metrics(hyps, refs)
print_table("BLIP Fine-tuned (augmented + Phase 1 gen)", metrics)

with open(RESULTS_DIR / "blip_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nSaved results  -> {results_file}")
print(f"Saved metrics  -> {RESULTS_DIR / 'blip_metrics.json'}")
print("\nClick 'Save Version' on Kaggle to persist /kaggle/working/")
