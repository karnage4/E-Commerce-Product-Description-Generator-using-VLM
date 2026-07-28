"""
Experiment N15 — Confidence intervals on the headline numbers.

The central claim rests on a difference between two percentages measured on
135 and 400 test items. A reviewer will ask whether the difference is inside
the noise. This computes bias-corrected bootstrap confidence intervals for the
image-counterfactual effect on each corpus, and a permutation test for the
difference between corpora.

Run:
    python -m models.novelty.bootstrap_ci
"""

import json
import random
import sys
from pathlib import Path

from rouge_score import rouge_scorer as rouge_lib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import RESULTS_DIR, build_metadata_prompt
from models.novelty.visual_lexicon import load_records
from models.novelty.prompt_prefix_audit import strip_prefix

N_BOOT = 2000
SEED = 42
OUT = RESULTS_DIR / "novelty_bootstrap_ci.json"


def per_item_rouge(hyps, refs):
    sc = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    return [100 * sc.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]


def boot_relative_drop(real, blank, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the statistic 100*(mean(real)-mean(blank))/mean(real)."""
    rng = random.Random(seed)
    n = len(real)
    idx = list(range(n))
    point = 100 * (sum(real) / n - sum(blank) / n) / (sum(real) / n)
    draws = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in range(n)]
        mr = sum(real[i] for i in s) / n
        mb = sum(blank[i] for i in s) / n
        if mr > 0:
            draws.append(100 * (mr - mb) / mr)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    return round(point, 2), round(lo, 2), round(hi, 2)


def boot_difference(realA, blankA, realB, blankB, n_boot=N_BOOT, seed=SEED):
    """Bootstrap CI for (relative drop on A) - (relative drop on B).

    A permutation test is inappropriate here: the two corpora have very
    different absolute score scales, so pooling and reshuffling raw per-item
    differences produces a statistic dominated by the scale mismatch rather
    than by the effect. Instead we resample each corpus independently and
    bootstrap the difference of the two relative drops directly. If the
    resulting interval excludes zero, the corpora differ.
    """
    rng = random.Random(seed)
    nA, nB = len(realA), len(realB)
    iA, iB = list(range(nA)), list(range(nB))

    def rel(real, blank, sample):
        mr = sum(real[i] for i in sample) / len(sample)
        mb = sum(blank[i] for i in sample) / len(sample)
        return 100 * (mr - mb) / mr if mr > 0 else 0.0

    obs = rel(realA, blankA, iA) - rel(realB, blankB, iB)
    draws = []
    for _ in range(n_boot):
        sA = [rng.choice(iA) for _ in range(nA)]
        sB = [rng.choice(iB) for _ in range(nB)]
        draws.append(rel(realA, blankA, sA) - rel(realB, blankB, sB))
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    # Two-sided bootstrap p: proportion of draws on the far side of zero.
    frac = sum(1 for d in draws if d <= 0) / len(draws)
    p = 2 * min(frac, 1 - frac)
    return round(obs, 2), round(lo, 2), round(hi, 2), round(max(p, 1.0 / n_boot), 5)


def daraz_pairs():
    """Prefix-stripped real/blank ROUGE-L per item for the Daraz BLIP."""
    recs = load_records()
    rows = [json.loads(l) for l in
            open(RESULTS_DIR / "novelty_visual_sensitivity_generations.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["model"] == "BLIP" and r["prompt"] == "full"]
    real, blank, refs = [], [], []
    for r in rows:
        rec = recs.get(str(r["item_id"]))
        if not rec:
            continue
        p = build_metadata_prompt(rec)
        a, _, _ = strip_prefix(r["real"], p)
        b, _, _ = strip_prefix(r["blank"], p)
        real.append(a.strip() or ".")
        blank.append(b.strip() or ".")
        refs.append(rec["description"].strip())
    return per_item_rouge(real, refs), per_item_rouge(blank, refs)


def abo_pairs(tag="_873"):
    """ABO real/blank per-item ROUGE-L, recomputed from saved generations if present."""
    p = RESULTS_DIR / f"novelty_abo_generations{tag}.jsonl"
    if not p.exists():
        return None, None
    real, blank, refs = [], [], []
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        if d.get("blank") is None:
            continue
        real.append(d["real"])
        blank.append(d["blank"])
        refs.append(d["reference"])
    return per_item_rouge(real, refs), per_item_rouge(blank, refs)


def main():
    print("\n" + "=" * 78)
    print("N15 — BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 78)

    out = {}

    dr, db = daraz_pairs()
    pt, lo, hi = boot_relative_drop(dr, db)
    out["DPD (Daraz) BLIP, prefix-stripped"] = {
        "n": len(dr), "relative_drop_pct": pt, "ci95": [lo, hi]}
    print(f"\n  Daraz  n={len(dr)}  image-removal cost = {pt}%  95% CI [{lo}, {hi}]")

    ar, ab = abo_pairs("_873")
    if ar:
        pt2, lo2, hi2 = boot_relative_drop(ar, ab)
        out["ABO 873, prefix-stripped"] = {
            "n": len(ar), "relative_drop_pct": pt2, "ci95": [lo2, hi2]}
        print(f"  ABO873 n={len(ar)}  image-removal cost = {pt2}%  95% CI [{lo2}, {hi2}]")

        obs, lo3, hi3, p = boot_difference(ar, ab, dr, db)
        out["bootstrap_difference_ABO873_minus_DPD"] = {
            "difference_pct_points": obs, "ci95": [lo3, hi3], "p_value": p}
        print(f"\n  ABO − DPD difference: {obs} percentage points, 95% CI [{lo3}, {hi3}], p = {p}")
        if lo3 > 0:
            print("  → The interval excludes zero. The corpora differ; this is not sampling noise.")
        else:
            print("  → The interval includes zero. Do not claim a corpus difference from this alone.")
    else:
        print("\n  [!] ABO per-item generations not saved — rerun abo_head2head with")
        print("      generation logging to get a CI for the ABO side.")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
