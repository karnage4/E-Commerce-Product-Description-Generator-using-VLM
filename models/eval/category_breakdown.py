"""
Per-category ROUGE-L breakdown — C2 substitute for confusion matrix.

Loads whichever results files exist and prints a table: rows = categories,
columns = models. Saves models/results/category_breakdown.json.

Run:
    python -m models.eval.category_breakdown
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

from rouge_score import rouge_scorer as rouge_lib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import RESULTS_DIR

RESULT_FILES = {
    "BLIP":      RESULTS_DIR / "blip_results.jsonl",
    "CLIP-GPT2": RESULTS_DIR / "clip_gpt2_results.jsonl",
    "Two-Stage": RESULTS_DIR / "two_stage_results.jsonl",
}

CAT_W  = 26
COL_W  = 14
OUT    = RESULTS_DIR / "category_breakdown.json"


def _load(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _rouge_l_mean(pairs: list[tuple[str, str]]) -> float:
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    vals = [scorer.score(ref, hyp)["rougeL"].fmeasure for hyp, ref in pairs]
    return round(sum(vals) / len(vals) * 100, 2) if vals else 0.0


def _extract_hyp(rec: dict, model_key: str) -> str:
    if model_key == "Two-Stage":
        return rec.get("description_stage1", rec.get("generated", "")).strip()
    return rec.get("generated", "").strip()


def main() -> None:
    model_cats: dict[str, dict[str, list[tuple[str, str]]]] = {}

    for model_name, path in RESULT_FILES.items():
        if not path.exists():
            print(f"  [~] Skipping (not found): {path.name}")
            continue
        records = _load(path)
        cat_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for rec in records:
            cat = rec.get("category", "unknown") or "unknown"
            hyp = _extract_hyp(rec, model_name)
            ref = rec.get("reference", "").strip()
            if hyp and ref:
                cat_map[cat].append((hyp, ref))
        model_cats[model_name] = cat_map
        print(f"  Loaded {len(records)} records — {model_name}")

    if not model_cats:
        print("  [!] No result files found. Run evaluation scripts first.")
        return

    models     = list(model_cats.keys())
    categories = sorted({c for cm in model_cats.values() for c in cm})

    # sample count per category (max across models)
    cat_n: dict[str, int] = {}
    for cat in categories:
        cat_n[cat] = max(len(model_cats[m].get(cat, [])) for m in models)

    scores: dict[str, dict[str, float | None]] = {m: {} for m in models}
    for m in models:
        for cat in categories:
            pairs = model_cats[m].get(cat, [])
            scores[m][cat] = _rouge_l_mean(pairs) if pairs else None

    header = f"{'Category':<{CAT_W}}{'N':>{COL_W}}" + "".join(f"{m:>{COL_W}}" for m in models)
    div    = "=" * len(header)
    sep    = "-" * len(header)

    print(f"\n  Per-Category ROUGE-L F1")
    print(div)
    print(header)
    print(sep)
    for cat in categories:
        row = f"{cat:<{CAT_W}}{cat_n[cat]:>{COL_W}}"
        for m in models:
            v = scores[m][cat]
            row += f"{v:>{COL_W}.2f}" if v is not None else f"{'—':>{COL_W}}"
        print(row)
    print(div + "\n")

    output = {
        "models": models,
        "categories": {
            cat: {"n": cat_n[cat], **{m: scores[m][cat] for m in models}}
            for cat in categories
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {OUT}")


if __name__ == "__main__":
    main()
