"""
Experiment N16 — What kind of duplicate is it?

N2 found that 44 of 135 test products have a pixel-identical image somewhere in
the training split. That is only a leakage problem if the *item* is the same.
Two genuinely different products that happen to share a stock photograph are a
nuisance; the same product listed twice across the split boundary is train/test
contamination and invalidates the evaluation.

This script separates the two by comparing, for every image-duplicate pair, how
similar the titles and the descriptions are:

    SAME ITEM      image matches AND title/description are near-identical
                   -> genuine leakage, must be fixed before any number is trusted
    SHARED PHOTO   image matches but the text differs substantially
                   -> different products reusing a catalogue image; not leakage

Run:
    python -m models.novelty.duplicate_diagnosis
"""

import json
import sys
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import METADATA_FILE, RESULTS_DIR

AUDIT = RESULTS_DIR / "novelty_duplicate_audit.json"
OUT = RESULTS_DIR / "novelty_duplicate_diagnosis.json"

TITLE_SAME = 75      # rapidfuzz token_sort_ratio above this = same product
DESC_SAME = 70


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
            recs[r["item_id"]] = r
    return recs


def main():
    print("\n" + "=" * 78)
    print("N16 — SAME ITEM, OR JUST THE SAME PHOTOGRAPH?")
    print("=" * 78)

    if not AUDIT.exists():
        raise SystemExit("Run models.novelty.duplicate_audit first.")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    recs = load_all()
    thresh = audit["hamming_threshold"]

    rows, same_item, shared_photo = [], 0, 0
    for tid, info in audit["neighbours"].items():
        if info["hamming"] > thresh:
            continue
        nid = info["nearest_train_id"]
        a, b = recs.get(tid), recs.get(nid)
        if not a or not b:
            continue
        ta = (a.get("item_name") or "").strip()
        tb = (b.get("item_name") or "").strip()
        da = (a.get("description") or "").strip()
        db = (b.get("description") or "").strip()
        t_sim = fuzz.token_sort_ratio(ta, tb)
        d_sim = fuzz.token_sort_ratio(da[:600], db[:600])
        verdict = "SAME ITEM" if (t_sim >= TITLE_SAME or d_sim >= DESC_SAME) else "SHARED PHOTO"
        if verdict == "SAME ITEM":
            same_item += 1
        else:
            shared_photo += 1
        rows.append({"test_id": tid, "train_id": nid, "hamming": info["hamming"],
                     "title_sim": round(t_sim, 1), "desc_sim": round(d_sim, 1),
                     "verdict": verdict, "test_title": ta[:90], "train_title": tb[:90]})

    n = len(rows)
    print(f"\n  image-duplicate pairs examined (pHash <= {thresh}): {n}")
    print(f"    SAME ITEM     {same_item:>3}  ({100*same_item/max(n,1):.1f}%)  -> real train/test leakage")
    print(f"    SHARED PHOTO  {shared_photo:>3}  ({100*shared_photo/max(n,1):.1f}%)  -> different products, same image")

    hard = [r for r in rows if r["hamming"] == 0]
    hard_same = sum(1 for r in hard if r["verdict"] == "SAME ITEM")
    print(f"\n  of the {len(hard)} pixel-identical (Hamming 0) pairs, "
          f"{hard_same} are the same item ({100*hard_same/max(len(hard),1):.1f}%)")

    print("\n  [Examples judged SAME ITEM]")
    for r in [x for x in rows if x["verdict"] == "SAME ITEM"][:4]:
        print(f"    title_sim={r['title_sim']:<6} desc_sim={r['desc_sim']:<6} h={r['hamming']}")
        print(f"      test : {r['test_title']}")
        print(f"      train: {r['train_title']}")

    print("\n  [Examples judged SHARED PHOTO]")
    for r in [x for x in rows if x["verdict"] == "SHARED PHOTO"][:4]:
        print(f"    title_sim={r['title_sim']:<6} desc_sim={r['desc_sim']:<6} h={r['hamming']}")
        print(f"      test : {r['test_title']}")
        print(f"      train: {r['train_title']}")

    OUT.write_text(json.dumps(
        {"n_pairs": n, "same_item": same_item, "shared_photo": shared_photo,
         "hamming0_pairs": len(hard), "hamming0_same_item": hard_same,
         "title_threshold": TITLE_SAME, "desc_threshold": DESC_SAME,
         "pairs": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
