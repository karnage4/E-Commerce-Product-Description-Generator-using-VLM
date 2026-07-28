"""
Experiment N7 — Do the metrics track judged visual grounding?

Additive experiment, CPU only. Reads the two-stage result files that already
contain per-item Gemma-4-31B judge scores; computes nothing new on the GPU.

Motivation
----------
N1-N3 argue that n-gram metrics reward metadata echo rather than visual
grounding, and propose replacements (Visual-Residual Recall, Residual ROUGE-L,
Echo Rate). That argument is only worth anything if the proposed metrics track
something real.

The two-stage pipeline already produced an independent per-item judgement of
visual grounding for all 135 Stage-1 outputs. This script uses it as ground
truth and asks, for every candidate metric, how well it correlates with the
judge.

A metric that measures description quality should correlate positively with
judged visual grounding. A metric that measures metadata echo should not.

Run:
    python -m models.novelty.metric_validity
"""

import json
import re
import sys
from pathlib import Path

import nltk
from nltk.stem.porter import PorterStemmer
from rouge_score import rouge_scorer as rouge_lib
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)

OUT_JSON = RESULTS_DIR / "novelty_metric_validity.json"

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()
WORD_RE = re.compile(r"[a-z0-9]+")


def content_tokens(t):
    return [STEM.stem(w) for w in WORD_RE.findall(t.lower())
            if w not in STOP and len(w) > 1]


def strip_prompt(text, pset):
    return " ".join(w for w in WORD_RE.findall(text.lower()) if STEM.stem(w) not in pset)


def load_records():
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
    recs = {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids:
                recs[r["item_id"]] = r
    return recs


def main():
    print("\n" + "=" * 78)
    print("N7 — METRIC VALIDITY AGAINST THE LLM JUDGE")
    print("=" * 78)

    records = load_records()
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

    rows = []
    for model_tag, fname in (("BLIP", "two_stage_results_blip.jsonl"),
                             ("CLIP-GPT2", "two_stage_results_clip_gpt2.jsonl")):
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sc = d.get("stage1_scores")
            gen = d.get("description_stage1")
            rec = records.get(str(d.get("item_id")))
            if not (isinstance(sc, dict) and sc and gen and rec):
                continue
            ref = (rec.get("description") or "").strip()
            if not ref:
                continue

            prompt = build_metadata_prompt(rec)
            pset = set(content_tokens(prompt))
            gset = set(content_tokens(gen))
            rt = content_tokens(ref)
            vset = {t for t in rt if t not in pset}
            gall = content_tokens(gen)

            rres = strip_prompt(ref, pset)
            gres = strip_prompt(gen, pset)

            rows.append({
                "model": model_tag,
                "item_id": d["item_id"],
                # candidate metrics
                "ROUGE-L": 100 * scorer.score(ref, gen)["rougeL"].fmeasure,
                "Residual ROUGE-L": 100 * scorer.score(rres, gres)["rougeL"].fmeasure if rres.strip() else 0.0,
                "Visual Residual Recall": 100 * len(gset & vset) / max(len(vset), 1),
                "Echo Rate": 100 * sum(1 for t in gall if t in pset) / max(len(gall), 1),
                # judge
                "judge_visual_grounding": float(sc.get("visual_grounding", 0)),
                "judge_fluency": float(sc.get("fluency", 0)),
                "judge_relevance": float(sc.get("relevance", 0)),
                "judge_overall": float(sc.get("overall", 0)),
            })

    print(f"\n  {len(rows)} scored Stage-1 outputs "
          f"({len({r['model'] for r in rows})} models)")

    metrics = ["ROUGE-L", "Residual ROUGE-L", "Visual Residual Recall", "Echo Rate"]
    judges = ["judge_visual_grounding", "judge_fluency", "judge_relevance", "judge_overall"]

    out = {"n": len(rows), "correlations": {}}
    print(f"\n  {'metric':<26}" + "".join(f"{j.replace('judge_',''):>20}" for j in judges))
    print("  " + "-" * (26 + 20 * len(judges)))
    for m in metrics:
        xs = [r[m] for r in rows]
        cells = []
        for j in judges:
            ys = [r[j] for r in rows]
            rho, p = spearmanr(xs, ys)
            out["correlations"].setdefault(m, {})[j] = {
                "spearman_rho": round(float(rho), 3),
                "p_value": round(float(p), 5),
                "pearson_r": round(float(pearsonr(xs, ys)[0]), 3),
            }
            star = "*" if p < 0.05 else " "
            cells.append(f"{rho:>+18.3f}{star}")
        print(f"  {m:<26}" + "".join(cells))
    print("\n  (Spearman rho; * = p < 0.05)")

    OUT_JSON.write_text(json.dumps({"rows": rows, **out}, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
