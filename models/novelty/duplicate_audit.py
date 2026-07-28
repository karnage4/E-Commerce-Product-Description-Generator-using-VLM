"""
Experiment N2 — Catalogue Near-Duplicate Contamination Audit.

Additive experiment: writes new result files only.

Motivation
----------
Milestone 2 reported a MobileNetV2 + cosine k-NN *retrieval* baseline scoring
BLEU-4 21.33, an order of magnitude above every fine-tuned generative model.
The report attributed this to cross-seller near-duplicate listings that survived
the two-pass dedup pipeline. This script tests that claim directly and turns it
into a measurement.

Method
------
1. Perceptual-hash every image of every train and test product.
2. For each test product, find its nearest train product (min Hamming distance
   over all image pairs).
3. Partition the test set into CONTAMINATED (a visually near-identical product
   exists in train) and CLEAN.
4. Build a pHash retrieval baseline that simply copies the nearest train
   product's description.
5. Score retrieval, BLIP, and CLIP-GPT2 separately on each partition.

If the retrieval score collapses on CLEAN while the generative scores hold, the
retrieval baseline was measuring catalogue duplication, not description quality.

Run:
    python -m models.novelty.duplicate_audit
"""

import json
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TRAIN_SPLIT, TEST_SPLIT, RESULTS_DIR
)
from models.shared.metrics import compute_all_metrics

HAMMING_THRESHOLD = 8          # same threshold the project's dedup pass uses
CACHE = RESULTS_DIR / "novelty_phash_cache.json"
OUT_JSON = RESULTS_DIR / "novelty_duplicate_audit.json"


def load_records(split_file):
    ids = set(Path(split_file).read_text(encoding="utf-8").split())
    out = {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids and r.get("images"):
                out[r["item_id"]] = r
    return out


def hash_product(rec) -> list[str]:
    """pHash every available image of a product."""
    hashes = []
    for rel in rec["images"]:
        p = IMAGES_DIR.parent / rel
        if not p.exists():
            continue
        try:
            hashes.append(str(imagehash.phash(Image.open(p).convert("RGB"))))
        except Exception:
            continue
    return hashes


def build_hashes(records, label, cache):
    out = {}
    for item_id, rec in tqdm(records.items(), desc=f"  hashing {label}"):
        if item_id in cache:
            out[item_id] = cache[item_id]
        else:
            h = hash_product(rec)
            cache[item_id] = h
            out[item_id] = h
    return out


def min_distance(test_hashes, train_hashes):
    """Return (nearest_train_id, min_hamming) for one test product."""
    best_id, best_d = None, 999
    th = [imagehash.hex_to_hash(h) for h in test_hashes]
    if not th:
        return None, 999
    for tid, hlist in train_hashes.items():
        for h in hlist:
            hh = imagehash.hex_to_hash(h)
            for t in th:
                d = t - hh
                if d < best_d:
                    best_d, best_id = d, tid
    return best_id, best_d


def load_system(path, gen_key):
    out = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get(gen_key):
                    out[str(d["item_id"])] = d[gen_key]
    return out


def score_subset(gens, refs, ids):
    ids = [i for i in ids if i in gens and i in refs]
    if len(ids) < 3:
        return None
    return compute_all_metrics([gens[i] for i in ids], [refs[i] for i in ids])


def main():
    print("\n" + "=" * 78)
    print("N2 — CATALOGUE NEAR-DUPLICATE CONTAMINATION AUDIT")
    print("=" * 78)

    train = load_records(TRAIN_SPLIT)
    test = load_records(TEST_SPLIT)
    print(f"\ntrain={len(train)}  test={len(test)}")

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    train_h = build_hashes(train, "train", cache)
    test_h = build_hashes(test, "test", cache)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")

    print("\n  matching each test product against the training catalogue...")
    neighbours = {}
    for tid, hl in tqdm(test_h.items(), desc="  nearest-neighbour"):
        nid, d = min_distance(hl, train_h)
        neighbours[tid] = {"nearest_train_id": nid, "hamming": int(d)}

    contaminated = [i for i, v in neighbours.items() if v["hamming"] <= HAMMING_THRESHOLD]
    clean = [i for i in test_h if i not in contaminated]

    print(f"\n[Contamination]")
    print(f"  test items with a near-duplicate in train (pHash<= {HAMMING_THRESHOLD}): "
          f"{len(contaminated)}/{len(test_h)} ({100*len(contaminated)/len(test_h):.1f}%)")
    print(f"  clean test items: {len(clean)}")

    dist = {}
    for v in neighbours.values():
        b = "0 (identical)" if v["hamming"] == 0 else (
            "1-4" if v["hamming"] <= 4 else
            "5-8" if v["hamming"] <= 8 else
            "9-16" if v["hamming"] <= 16 else "17+")
        dist[b] = dist.get(b, 0) + 1
    print("  hamming distribution:", dict(sorted(dist.items())))

    refs = {i: r["description"].strip() for i, r in test.items() if r.get("description")}

    # pHash retrieval baseline: copy the nearest train product's description
    retrieval = {}
    for tid, v in neighbours.items():
        nid = v["nearest_train_id"]
        if nid and train.get(nid, {}).get("description"):
            retrieval[tid] = train[nid]["description"].strip()

    systems = {
        "pHash Retrieval (copy nearest train desc)": retrieval,
        "BLIP fine-tuned":      load_system(RESULTS_DIR / "blip_results.jsonl", "generated"),
        "CLIP-GPT2 fine-tuned": load_system(RESULTS_DIR / "clip_gpt2_results.jsonl", "generated"),
    }

    results = {}
    print("\n[Scores by partition]")
    for name, gens in systems.items():
        if not gens:
            continue
        row = {
            "ALL":          score_subset(gens, refs, list(test_h)),
            "CONTAMINATED": score_subset(gens, refs, contaminated),
            "CLEAN":        score_subset(gens, refs, clean),
        }
        results[name] = row
        print(f"\n  {name}")
        for part, sc in row.items():
            if sc:
                print(f"    {part:<14} " + "  ".join(f"{k}={v}" for k, v in sc.items()))
            else:
                print(f"    {part:<14} (too few samples)")

    payload = {
        "hamming_threshold": HAMMING_THRESHOLD,
        "n_test": len(test_h),
        "n_contaminated": len(contaminated),
        "n_clean": len(clean),
        "hamming_distribution": dist,
        "contaminated_ids": contaminated,
        "neighbours": neighbours,
        "scores": results,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
