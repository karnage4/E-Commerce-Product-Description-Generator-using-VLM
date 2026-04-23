"""
BLIP Evaluation — Self-contained Colab/Kaggle cell.

Paste the ENTIRE content of this file into ONE Colab cell and run it.
No external file imports needed — everything is defined inline.

Prerequisites (must be done first):
  - Cell 0A or 0B from eval_setup.py (data + checkpoints accessible)
  - Cell 1 from eval_setup.py  (nltk + rouge-score installed)
"""

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# ── PATHS — edit if using Kaggle (/tmp/...) ───────────────────────────────────
DATA_ROOT       = Path("/content/daraz_data")
METADATA_FILE   = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_DIR      = DATA_ROOT / "images"
TEST_SPLIT_FILE = DATA_ROOT / "splits" / "test.txt"
RESULTS_DIR     = Path("/content/results")

# Checkpoint: check same-session path first, then Drive-restored path
_BLIP_CANDIDATES = [
    Path("/content/checkpoints/blip/best_model"),   # same session
    Path("/content/blip_best_model"),               # if you unzipped directly
]
CHECKPOINT_DIR = next((p for p in _BLIP_CANDIDATES if p.exists()), _BLIP_CANDIDATES[0])

MAX_SAMPLES = 150   # how many test products to evaluate

# ── Config ────────────────────────────────────────────────────────────────────
print(f"Checkpoint: {CHECKPOINT_DIR}")
print(f"Exists:     {CHECKPOINT_DIR.exists()}")
print(f"Files:      {list(CHECKPOINT_DIR.iterdir()) if CHECKPOINT_DIR.exists() else 'NOT FOUND'}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_metadata_prompt(rec: dict) -> str:
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    for k, v in list((rec.get("attributes") or {}).items())[:3]:
        if k and v: parts.append(f"{k}: {v}")
    return ". ".join(parts)


def load_test_records(max_samples: int) -> list[dict]:
    test_ids = set(TEST_SPLIT_FILE.read_text().strip().splitlines())
    records  = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            try: rec = json.loads(line.strip())
            except: continue
            if rec.get("item_id") not in test_ids: continue
            if not rec.get("images") or not rec.get("description", "").strip(): continue
            records.append(rec)
            if len(records) >= max_samples: break
    print(f"Loaded {len(records)} test records")
    return records


def load_image(rec: dict) -> Image.Image:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try: return Image.open(p).convert("RGB")
            except: pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rouge_lib

    smooth    = SmoothingFunction().method4
    refs_tok  = [[r.split()] for r in references]
    hyps_tok  = [h.split() for h in hypotheses]

    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)) / len(references)

    meteor = sum(meteor_score([r.split()], h.split()) for r, h in zip(references, hypotheses)) / len(references)

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


# ── Main evaluation ───────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

print(f"\nLoading BLIP from {CHECKPOINT_DIR} ...")
processor = BlipProcessor.from_pretrained(str(CHECKPOINT_DIR))
model     = BlipForConditionalGeneration.from_pretrained(
    str(CHECKPOINT_DIR),
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
).to(device)
model.eval()
print("Model loaded!")

records    = load_test_records(MAX_SAMPLES)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_file = RESULTS_DIR / "blip_results.jsonl"

hypotheses: list[str] = []
references: list[str] = []

t0 = time.time()
with open(results_file, "w", encoding="utf-8") as out_f:
    for rec in tqdm(records, desc="BLIP inference", unit="item"):
        image  = load_image(rec)
        prompt = build_metadata_prompt(rec)

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    output_ids = model.generate(**inputs, max_new_tokens=150,
                                                num_beams=4, no_repeat_ngram_size=3)
            else:
                output_ids = model.generate(**inputs, max_new_tokens=150,
                                            num_beams=4, no_repeat_ngram_size=3)

        generated = processor.decode(output_ids[0], skip_special_tokens=True).strip()
        reference = rec["description"].strip()

        entry = {"item_id": rec["item_id"], "category": rec.get("category",""),
                 "generated": generated, "reference": reference}
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()

        hypotheses.append(generated)
        references.append(reference)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min  ({elapsed/len(hypotheses):.1f}s/sample)")

# ── Compute and display metrics ───────────────────────────────────────────────
metrics = compute_metrics(hypotheses, references)
print_table("BLIP Fine-tuned", metrics)

# Save metrics JSON
metrics_file = RESULTS_DIR / "blip_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved results  -> {results_file}")
print(f"Saved metrics  -> {metrics_file}")

# ── Download results to your laptop ──────────────────────────────────────────
from google.colab import files
files.download(str(metrics_file))
files.download(str(results_file))

# Also back up to Drive
try:
    import shutil
    shutil.copy(str(metrics_file),  "/content/drive/MyDrive/daraz_cv_project/blip_metrics.json")
    shutil.copy(str(results_file),  "/content/drive/MyDrive/daraz_cv_project/blip_results.jsonl")
    print("Backed up to Drive!")
except Exception as e:
    print(f"Drive backup skipped: {e}")
