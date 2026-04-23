"""
Combined Results Table — run this cell AFTER both evaluate_colab cells.

Loads all saved metrics JSONs from /content/results/ and prints the
comparison table exactly like the Stanford paper Table 1.

Also downloads all_metrics.json to your laptop.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("/content/results")

# ── Load all metric files ─────────────────────────────────────────────────────
RESULT_FILES = {
    "CLIP-GPT2 Fine-tuned":  "clip_gpt2_metrics.json",
    "BLIP Fine-tuned":       "blip_metrics.json",
}

results = {}
for display_name, filename in RESULT_FILES.items():
    path = RESULTS_DIR / filename
    if path.exists():
        with open(path) as f:
            results[display_name] = json.load(f)
        print(f"  Loaded: {filename}")
    else:
        print(f"  [~] Not found (skipped): {filename}")

# ── Print table ───────────────────────────────────────────────────────────────
metrics_order = ["BLEU-1", "BLEU-4", "ROUGE-L", "METEOR"]

header = f"{'Model':<35}" + "".join(f"{m:>10}" for m in metrics_order)
sep    = "=" * len(header)

print(f"\n{sep}")
print("RESULTS TABLE (mirroring Stanford paper format)")
print(sep)
print(header)
print("-" * len(header))
for model_name, scores in results.items():
    row = f"{model_name:<35}" + "".join(f"{scores.get(m, 0.0):>10.2f}" for m in metrics_order)
    print(row)
print(sep)

# ── Save combined ─────────────────────────────────────────────────────────────
combined_path = RESULTS_DIR / "all_metrics.json"
with open(combined_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved combined -> {combined_path}")

# ── Download all results ──────────────────────────────────────────────────────
from google.colab import files
files.download(str(combined_path))
for filename in RESULT_FILES.values():
    p = RESULTS_DIR / filename
    if p.exists():
        files.download(str(p))

print("\nAll files downloaded!")

# Back up to Drive
try:
    import shutil
    drive_dir = "/content/drive/MyDrive/daraz_cv_project"
    shutil.copy(str(combined_path), f"{drive_dir}/all_metrics.json")
    print("Backed up to Drive!")
except Exception as e:
    print(f"Drive backup skipped: {e}")
