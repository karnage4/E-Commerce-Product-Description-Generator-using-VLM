"""
Experiment N21 — Does the system solve the seller's actual problem?

The problem this project exists to solve is not "score well on BLEU". It is the
one a Daraz seller has:

    my listing's description is a copy of my competitor's, even though my
    product is slightly different, and it says nothing about what my product
    actually looks like

N17 confirmed that is real and widespread: 56.4% of listings sit in a multi-item
duplicate group, and copied *descriptions* were the dominant edge type (2,220 of
them, against 511 title edges).

So the two things a useful system must do are:

    DIFFERENTIATION   produce text that is not a copy of some other seller's
                      listing for a similar product
    SPECIFICITY       produce text that actually describes this product's
                      appearance

Neither is what BLEU/ROUGE measures. Both are measurable directly, without a
reference, which is the point.

Metrics (all 0-100, computed against the whole corpus, not just the test split):

    CompetitorSim   max similarity between a text and any *other* listing's
                    seller-written description. High = it reads like a copy of
                    something already on the platform. Lower is better.
    SelfSim         max similarity between a text and the *other generated*
                    texts. High = the model emits boilerplate for everything.
                    Lower is better.
    VAD             visual attribute density (validated in N8). Higher is better.

The seller's own reference is scored on the same scales, so it acts as the
incumbent baseline: whatever the system produces has to beat what is on the
site today.

Run:
    python -m models.novelty.differentiation
"""

import json
import sys
from pathlib import Path

from rapidfuzz import fuzz, process
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)
from models.novelty.visual_lexicon import density, VISUAL
from models.novelty.prompt_prefix_audit import strip_prefix

OUT = RESULTS_DIR / "novelty_differentiation.json"
TRUNC = 400          # characters compared; descriptions are long and noisy
NEAR_DUP = 70        # rapidfuzz score above which two texts read as the same copy


def load_all():
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
            if (r.get("description") or "").strip():
                recs[r["item_id"]] = r
    return recs


def load_system(fname, key):
    out = {}
    p = RESULTS_DIR / fname
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            v = d.get(key)
            if v:
                out[str(d["item_id"])] = v
    return out


def max_sim_against(queries, corpus, exclude_self_idx=None):
    """For each query, the highest similarity to anything in corpus."""
    m = process.cdist(queries, corpus, scorer=fuzz.token_sort_ratio, workers=-1)
    out = []
    for i, row in enumerate(m):
        best = 0
        for j, v in enumerate(row):
            if exclude_self_idx is not None and exclude_self_idx[i] == j:
                continue
            if v > best:
                best = v
        out.append(best)
    return out


def main():
    print("\n" + "=" * 78)
    print("N21 — DIFFERENTIATION AND SPECIFICITY: THE SELLER'S ACTUAL PROBLEM")
    print("=" * 78)

    recs = load_all()
    test_ids = [i for i in Path(TEST_SPLIT).read_text(encoding="utf-8").split() if i in recs]
    all_ids = list(recs)
    corpus = [(recs[i]["description"] or "")[:TRUNC] for i in all_ids]
    pos = {i: k for k, i in enumerate(all_ids)}
    print(f"\n  corpus: {len(all_ids)} seller descriptions | test items: {len(test_ids)}")

    systems = {}
    systems["Seller's own listing (incumbent)"] = {i: recs[i]["description"] for i in test_ids}
    systems["Metadata template"] = {i: build_metadata_prompt(recs[i]) for i in test_ids}

    blind = load_system("novelty_blind_baseline_results.jsonl", "generated")
    if blind:
        systems["Blind GPT-2 (no image)"] = blind

    blip = load_system("blip_results.jsonl", "generated")
    if blip:
        systems["BLIP (full prompt, prefix stripped)"] = {
            i: strip_prefix(g, build_metadata_prompt(recs[i]))[0]
            for i, g in blip.items() if i in recs}

    s1 = load_system("two_stage_results_blip.jsonl", "description_stage1")
    if s1:
        systems["BLIP Stage-1 (image-forced)"] = s1
    s2 = load_system("two_stage_results_blip.jsonl", "description_stage2")
    if s2:
        systems["Two-Stage (Stage-1 + refiner)"] = s2

    clip = load_system("clip_gpt2_results.jsonl", "generated")
    if clip:
        systems["CLIP-GPT2"] = clip

    results = {}
    print(f"\n  {'system':<38}{'CompetitorSim':>14}{'%near-dup':>11}{'SelfSim':>9}{'VAD':>7}")
    print("  " + "-" * 79)

    for name, gens in systems.items():
        ids = [i for i in test_ids if i in gens and (gens[i] or "").strip()]
        if len(ids) < 20:
            continue
        texts = [gens[i][:TRUNC] for i in ids]
        # similarity to other sellers' listings (exclude the item's own listing)
        excl = [pos[i] for i in ids]
        comp = max_sim_against(texts, corpus, exclude_self_idx=excl)
        # similarity to the other texts this system produced
        self_excl = list(range(len(texts)))
        selfsim = max_sim_against(texts, texts, exclude_self_idx=self_excl)
        vad = [density(t, VISUAL) for t in texts]

        near = 100 * sum(1 for c in comp if c >= NEAR_DUP) / len(comp)
        row = {
            "n": len(ids),
            "CompetitorSim": round(float(sum(comp)) / len(comp), 1),
            "pct_near_duplicate_of_another_listing": round(float(near), 1),
            "SelfSim": round(float(sum(selfsim)) / len(selfsim), 1),
            "VAD": round(float(sum(vad)) / len(vad), 2),
        }
        results[name] = row
        print(f"  {name:<38}{row['CompetitorSim']:>14.1f}{near:>10.1f}%"
              f"{row['SelfSim']:>9}{row['VAD']:>7}")

    print("\n  CompetitorSim: max similarity to another seller's listing (lower = more differentiated)")
    print("  %near-dup    : share scoring >= 70 against some other listing")
    print("  SelfSim      : max similarity to this system's other outputs (lower = less boilerplate)")
    print("  VAD          : visual attribute density (higher = more about the product's appearance)")

    OUT.write_text(json.dumps({"near_dup_threshold": NEAR_DUP,
                               "systems": results}, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
