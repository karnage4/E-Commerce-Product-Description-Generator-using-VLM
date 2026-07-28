"""
Experiment N1 — Metadata-Echo Audit and Metadata-Discounted Evaluation (MDE).

Additive experiment: reads existing result files, writes new ones. Nothing in the
original evaluation path is modified.

Motivation
----------
On Daraz (and on ABO, and on every seller-authored e-commerce corpus we know of)
the "reference" description is written by the same seller who filled in the
structured attribute table. The reference therefore contains a large block of
text that is trivially recoverable from the metadata prompt we feed the model.
Any n-gram metric computed against that reference pays the model for copying its
own input.

This script quantifies that and proposes a decomposition that does not.

For every test item we split the reference's content tokens into:
    M-set : tokens that also occur in the metadata prompt  (metadata-derivable)
    V-set : tokens that do NOT occur in the metadata prompt (visual / novel)

and report, per system:
    Metadata Recall (MR)         = |gen ∩ M| / |M|
    Visual-Residual Recall (VRR) = |gen ∩ V| / |V|
    Echo Rate (ER)               = |gen ∩ prompt| / |gen|
    Residual ROUGE-L             = ROUGE-L(gen \ prompt, reference \ prompt)

Run:
    python -m models.novelty.echo_audit
"""

import json
import re
from collections import Counter
from pathlib import Path

import nltk
from nltk.stem.porter import PorterStemmer
from rouge_score import rouge_scorer as rouge_lib

from models.shared.config import (
    METADATA_FILE, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt
)

OUT_JSON = RESULTS_DIR / "novelty_echo_audit.json"

# ── Tokenisation helpers ──────────────────────────────────────────────────────

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()
WORD_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, drop stopwords and 1-char tokens, stem."""
    toks = WORD_RE.findall(text.lower())
    return [STEM.stem(t) for t in toks if t not in STOP and len(t) > 1]


def strip_prompt_tokens(text: str, prompt_set: set[str]) -> str:
    """Remove every token of `text` whose stem occurs in the metadata prompt."""
    kept = [
        t for t in WORD_RE.findall(text.lower())
        if STEM.stem(t) not in prompt_set
    ]
    return " ".join(kept)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_records() -> dict[str, dict]:
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


def load_system(path: Path, gen_key: str) -> dict[str, str]:
    """Return {item_id: generated_text} from a results jsonl."""
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            g = d.get(gen_key)
            if g:
                out[str(d["item_id"])] = g
    return out


# ── Core measurement ──────────────────────────────────────────────────────────

def evaluate_system(name, gens, records, verbose=True):
    """Compute MR / VRR / ER / residual ROUGE-L for one system."""
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

    mr_num = mr_den = vr_num = vr_den = 0
    echo_num = echo_den = 0
    res_rouge, raw_rouge = [], []
    n = 0

    for item_id, rec in records.items():
        gen = gens.get(item_id)
        if not gen:
            continue
        ref = (rec.get("description") or "").strip()
        if not ref:
            continue

        prompt = build_metadata_prompt(rec)
        p_set = set(content_tokens(prompt))
        g_set = set(content_tokens(gen))
        r_toks = content_tokens(ref)

        m_set = {t for t in r_toks if t in p_set}      # metadata-derivable
        v_set = {t for t in r_toks if t not in p_set}  # visual / novel

        mr_num += len(g_set & m_set); mr_den += len(m_set)
        vr_num += len(g_set & v_set); vr_den += len(v_set)

        g_all = content_tokens(gen)
        echo_num += sum(1 for t in g_all if t in p_set)
        echo_den += len(g_all)

        raw_rouge.append(scorer.score(ref, gen)["rougeL"].fmeasure)

        # residual: delete every metadata token from BOTH sides
        g_res = strip_prompt_tokens(gen, p_set)
        r_res = strip_prompt_tokens(ref, p_set)
        if r_res.strip():
            res_rouge.append(scorer.score(r_res, g_res)["rougeL"].fmeasure)
        n += 1

    if n == 0:
        return None

    out = {
        "n":                    n,
        "ROUGE-L":              round(100 * sum(raw_rouge) / len(raw_rouge), 2),
        "Residual ROUGE-L":     round(100 * sum(res_rouge) / len(res_rouge), 2),
        "Metadata Recall":      round(100 * mr_num / max(mr_den, 1), 2),
        "Visual Residual Recall": round(100 * vr_num / max(vr_den, 1), 2),
        "Echo Rate":            round(100 * echo_num / max(echo_den, 1), 2),
    }
    if verbose:
        print(f"  {name:<34} " + "  ".join(f"{k}={v}" for k, v in out.items() if k != "n"))
    return out


def reference_decomposition(records):
    """How much of the average reference is recoverable from metadata alone?"""
    frac, lens, mlens = [], [], []
    for rec in records.values():
        ref = (rec.get("description") or "").strip()
        if not ref:
            continue
        p_set = set(content_tokens(build_metadata_prompt(rec)))
        r_toks = content_tokens(ref)
        if not r_toks:
            continue
        frac.append(sum(1 for t in r_toks if t in p_set) / len(r_toks))
        lens.append(len(r_toks))
        mlens.append(len(p_set))
    return {
        "n_items": len(frac),
        "mean_pct_of_reference_derivable_from_metadata": round(100 * sum(frac) / len(frac), 2),
        "median_pct": round(100 * sorted(frac)[len(frac) // 2], 2),
        "mean_reference_content_tokens": round(sum(lens) / len(lens), 1),
        "mean_metadata_content_tokens": round(sum(mlens) / len(mlens), 1),
    }


def main():
    print("\n" + "=" * 78)
    print("N1 — METADATA-ECHO AUDIT / METADATA-DISCOUNTED EVALUATION")
    print("=" * 78)

    records = load_test_records()
    print(f"\nLoaded {len(records)} test records")

    decomp = reference_decomposition(records)
    print("\n[Reference decomposition]")
    for k, v in decomp.items():
        print(f"  {k:<48} {v}")

    # ── Systems under test ────────────────────────────────────────────────────
    systems = {}

    # The metadata prompt itself, submitted verbatim as a "description".
    systems["Metadata-Only (template)"] = {
        i: build_metadata_prompt(r) for i, r in records.items()
    }
    systems["BLIP fine-tuned"] = load_system(RESULTS_DIR / "blip_results.jsonl", "generated")
    systems["CLIP-GPT2 fine-tuned"] = load_system(RESULTS_DIR / "clip_gpt2_results.jsonl", "generated")
    systems["Two-Stage BLIP (stage2)"] = load_system(
        RESULTS_DIR / "two_stage_results_blip.jsonl", "description_stage2")
    systems["Two-Stage BLIP (stage1)"] = load_system(
        RESULTS_DIR / "two_stage_results_blip.jsonl", "description_stage1")
    systems["Two-Stage CLIP-GPT2 (stage2)"] = load_system(
        RESULTS_DIR / "two_stage_results_clip_gpt2.jsonl", "description_stage2")

    print("\n[Per-system decomposition]  (all values %)")
    results = {}
    for name, gens in systems.items():
        if not gens:
            print(f"  {name:<34} (no results file — skipped)")
            continue
        r = evaluate_system(name, gens, records)
        if r:
            results[name] = r

    payload = {"reference_decomposition": decomp, "systems": results}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
