import json
from pathlib import Path

results_dir = Path("models/results")
results_dir.mkdir(parents=True, exist_ok=True)

metadata = {"BLEU-1": 1.04, "BLEU-4": 0.30, "ROUGE-L": 13.71, "METEOR": 8.80}
blip     = {"BLEU-1": 6.33, "BLEU-4": 0.65, "ROUGE-L": 14.17, "METEOR": 12.13}
clip     = {"BLEU-1": 8.77, "BLEU-4": 1.54, "ROUGE-L": 11.03, "METEOR": 10.28}

all_metrics = {
    "Metadata-Only (Baseline)": metadata,
    "CLIP-GPT2 Fine-tuned":     clip,
    "BLIP Fine-tuned":          blip,
}

with open(results_dir / "metadata_baseline_metrics.json", "w") as f:
    json.dump(metadata, f, indent=2)
with open(results_dir / "all_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=2)

print("All metrics saved to models/results/")

metrics_order = ["BLEU-1", "BLEU-4", "ROUGE-L", "METEOR"]
name_w, col_w = 35, 10
header = "{:<{}}".format("Model", name_w) + "".join("{:>{}}".format(m, col_w) for m in metrics_order)
sep = "=" * len(header)
print("\n" + sep)
print("FINAL RESULTS TABLE")
print(sep)
print(header)
print("-" * len(header))
for model, scores in all_metrics.items():
    row = "{:<{}}".format(model, name_w) + "".join("{:>{}.2f}".format(scores[m], col_w) for m in metrics_order)
    print(row)
print(sep)
