"""
Evaluation metrics for generated product descriptions.

Computes: BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr
All functions expect plain string lists (not tokenized).

This module runs locally on CPU — no GPU required.
"""

import json
from pathlib import Path
from typing import Optional

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rouge_lib

# Download NLTK data on first run
for resource in ["wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


# ── Individual metric functions ───────────────────────────────────────────────

def bleu_scores(hypotheses: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus-level BLEU-1 and BLEU-4."""
    refs_tok  = [[r.split()] for r in references]
    hyps_tok  = [h.split() for h in hypotheses]
    smooth    = SmoothingFunction().method4
    return {
        "BLEU-1": round(corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=smooth) * 100, 2),
        "BLEU-4": round(corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth) * 100, 2),
    }


def rouge_l_score(hypotheses: list[str], references: list[str]) -> float:
    """Compute mean ROUGE-L F1 score."""
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
    return round(sum(scores) / len(scores) * 100, 2)


def meteor_scores(hypotheses: list[str], references: list[str]) -> float:
    """Compute mean METEOR score."""
    scores = [
        meteor_score([r.split()], h.split())
        for r, h in zip(references, hypotheses)
    ]
    return round(sum(scores) / len(scores) * 100, 2)


def cider_score(hypotheses: list[str], references: list[str]) -> float:
    """
    Compute CIDEr score using pycocoevalcap.
    Falls back to 0.0 if pycocoevalcap is not installed.
    """
    try:
        from pycocoevalcap.cider.cider import Cider
        gts = {i: [r] for i, r in enumerate(references)}
        res = {i: [h] for i, h in enumerate(hypotheses)}
        score, _ = Cider().compute_score(gts, res)
        return round(score * 100, 2)
    except ImportError:
        print("  [!] pycocoevalcap not installed — CIDEr skipped. Run: pip install pycocoevalcap")
        return 0.0


# ── All-in-one ────────────────────────────────────────────────────────────────

def compute_all_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute all 5 metrics at once.

    Args:
        hypotheses: List of model-generated descriptions (one per sample)
        references: List of ground-truth descriptions (one per sample)

    Returns:
        Dict with keys: BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr
    """
    assert len(hypotheses) == len(references), "Lists must be equal length"
    assert len(hypotheses) > 0, "Cannot evaluate empty lists"

    bleu = bleu_scores(hypotheses, references)
    return {
        **bleu,
        "ROUGE-L": rouge_l_score(hypotheses, references),
        "METEOR":  meteor_scores(hypotheses, references),
        "CIDEr":   cider_score(hypotheses, references),
    }


# ── Save / display results ────────────────────────────────────────────────────

def print_metrics_table(results: dict[str, dict[str, float]]) -> None:
    """
    Pretty-prints a comparison table of multiple models.

    Args:
        results: {model_name: {metric: value, ...}, ...}

    Example:
        print_metrics_table({
            "Metadata-Only": {"BLEU-1": 5.84, "CIDEr": 0.85, ...},
            "BLIP Fine-tuned": {"BLEU-1": 48.11, "CIDEr": 7.73, ...},
        })
    """
    metrics = ["CIDEr", "BLEU-1", "BLEU-4", "ROUGE-L", "METEOR"]
    col_w = 12

    header = f"{'Model':<30}" + "".join(f"{m:>{col_w}}" for m in metrics)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for model_name, scores in results.items():
        row = f"{model_name:<30}" + "".join(f"{scores.get(m, 0.0):>{col_w}.2f}" for m in metrics)
        print(row)
    print("=" * len(header) + "\n")


def save_metrics(results: dict, out_path: str) -> None:
    """Save metrics dict to a JSON file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved metrics → {out_path}")
