"""
Results aggregator — runs locally on CPU.

Loads all saved metric JSON files from models/results/
and prints a combined comparison table (matches Stanford paper Table 1 format).

Usage:
    python -m models.compare_results
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.shared.metrics import print_metrics_table, save_metrics
from models.shared.config import RESULTS_DIR


# Map result file names → display names (order matters for table rows)
RESULT_FILES = {
    "Metadata-Only (Baseline 1)":        "metadata_baseline_metrics.json",
    "Gemini-1.5-Flash (zero-shot API)":  "gemini_metrics.json",
    "CLIP-GPT2 Fine-tuned":              "clip_gpt2_metrics.json",
    "BLIP Fine-tuned":                   "blip_metrics.json",
}


def load_all_results() -> dict[str, dict]:
    results = {}
    for display_name, filename in RESULT_FILES.items():
        path = RESULTS_DIR / filename
        if path.exists():
            with open(path) as f:
                results[display_name] = json.load(f)
        else:
            print(f"  [~] Not found (skipped): {filename}")
    return results


def main():
    print(f"\n  Loading results from {RESULTS_DIR}...")
    results = load_all_results()

    if not results:
        print("  [!] No result files found. Run baselines and model evaluation first.")
        return

    print(f"\n  Found results for: {list(results.keys())}")
    print_metrics_table(results)

    # Save combined results
    combined_path = RESULTS_DIR / "all_metrics.json"
    save_metrics(results, str(combined_path))


if __name__ == "__main__":
    main()
