"""
Full Results Table — Metadata Baseline + Both Fine-tuned Models.

Run this cell in the SAME Colab session (data still at /content/daraz_data/).
This computes the Metadata-Only baseline (no model needed, runs in seconds)
and prints the complete comparison table for your report.
"""

import json
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rouge_lib

DATA_ROOT       = Path("/content/daraz_data")
METADATA_FILE   = DATA_ROOT / "metadata" / "listings_final.jsonl"
TEST_SPLIT_FILE = DATA_ROOT / "splits" / "test.txt"
RESULTS_DIR     = Path("/content/results")
MAX_SAMPLES     = 150

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_metadata_prompt(rec):
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    for k, v in list((rec.get("attributes") or {}).items())[:3]:
        if k and v: parts.append(f"{k}: {v}")
    return ". ".join(parts)


def load_test_records(max_samples):
    test_ids = set(TEST_SPLIT_FILE.read_text().strip().splitlines())
    records = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            try: rec = json.loads(line.strip())
            except: continue
            if rec.get("item_id") not in test_ids: continue
            if not rec.get("description", "").strip(): continue
            records.append(rec)
            if len(records) >= max_samples: break
    return records


def compute_metrics(hypotheses, references):
    smooth   = SmoothingFunction().method4
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]
    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth)
    scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r,h)["rougeL"].fmeasure for r,h in zip(references,hypotheses)) / len(references)
    meteor  = sum(meteor_score([r.split()], h.split()) for r,h in zip(references,hypotheses)) / len(references)
    return {
        "BLEU-1":  round(bleu1  * 100, 2),
        "BLEU-4":  round(bleu4  * 100, 2),
        "ROUGE-L": round(rouge_l * 100, 2),
        "METEOR":  round(meteor  * 100, 2),
    }


# ── Metadata-Only Baseline (no model) ────────────────────────────────────────
print("Computing Metadata-Only baseline...")
records    = load_test_records(MAX_SAMPLES)
hypotheses = [build_metadata_prompt(r) for r in records]
references  = [r["description"].strip() for r in records]
meta_metrics = compute_metrics(hypotheses, references)
print(f"  Done. {len(records)} samples evaluated.")

with open(RESULTS_DIR / "metadata_baseline_metrics.json", "w") as f:
    json.dump(meta_metrics, f, indent=2)


# ── Load fine-tuned model metrics ─────────────────────────────────────────────
all_results = {
    "Metadata-Only (Baseline)":  meta_metrics,
}

for name, filename in [
    ("CLIP-GPT2 Fine-tuned", "clip_gpt2_metrics.json"),
    ("BLIP Fine-tuned",      "blip_metrics.json"),
]:
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            all_results[name] = json.load(f)
    else:
        print(f"  [!] Not found: {filename} — run evaluate_colab.py first")


# ── Print full table ──────────────────────────────────────────────────────────
metrics_order = ["BLEU-1", "BLEU-4", "ROUGE-L", "METEOR"]
col_w  = 10
name_w = 35
header = f"{'Model':<{name_w}}" + "".join(f"{m:>{col_w}}" for m in metrics_order)
sep    = "=" * len(header)

print(f"\n{sep}")
print("FULL RESULTS TABLE")
print(sep)
print(header)
print("-" * len(header))
for model_name, scores in all_results.items():
    row = f"{model_name:<{name_w}}" + "".join(f"{scores.get(m, 0.0):>{col_w}.2f}" for m in metrics_order)
    print(row)
print(sep)

# Save combined
combined_path = RESULTS_DIR / "all_metrics.json"
with open(combined_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved -> {combined_path}")

from google.colab import files
files.download(str(combined_path))
files.download(str(RESULTS_DIR / "metadata_baseline_metrics.json"))
