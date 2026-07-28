"""
Experiment N17 — A leakage-free split.

N16 established that only about half of the image-duplicate pairs are the same
product; the rest are different products reusing a catalogue photograph. So the
fix is not "drop everything with a matching image" — that would throw away
legitimate data. The fix is to identify *item* duplicates and keep every member
of an item group inside one split.

Grouping rule. Two listings are the same item if any of:
    strong text     title similarity >= 82        (the project's own threshold)
    copied text     description similarity >= 70
    image + text    pHash Hamming <= 4 AND title similarity >= 60

Note the third clause: an image match alone is never sufficient, because N16
showed that is only ~50% precise. Text agreement is what distinguishes a
re-listed product from a shared stock photo.

Connected components over those edges become groups; groups are then assigned
whole to train/val/test, largest first, greedily filling category quotas so the
category distribution is preserved.

Writes new split files to data/data/processed/splits_clean/. The original
splits are untouched and remain the Milestone-2 baseline.

Run:
    python -m models.novelty.clean_split
    python -m models.novelty.clean_split --verify
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from rapidfuzz import fuzz, process
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import METADATA_FILE, IMAGES_DIR, SPLITS_DIR, RESULTS_DIR

OUT_DIR = SPLITS_DIR.parent / "splits_clean"
REPORT = RESULTS_DIR / "novelty_clean_split.json"
PHASH_CACHE = RESULTS_DIR / "novelty_phash_cache_all.json"

# The v2 corpus lives under the pipeline's own paths (data/processed), separate
# from the Milestone-2 dataset at data/data/processed. --v2 switches to it.
_V2_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"
META_FILE = METADATA_FILE
IMG_ROOT = IMAGES_DIR

TITLE_STRONG = 82
DESC_COPIED = 70
IMG_HAMMING = 4
IMG_TITLE_MIN = 60
SEED = 42
RATIOS = (0.80, 0.10, 0.10)


def load_records():
    recs = []
    with open(META_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("images") and (r.get("description") or "").strip():
                recs.append(r)
    return recs


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def hash_images(recs):
    cache = json.loads(PHASH_CACHE.read_text(encoding="utf-8")) if PHASH_CACHE.exists() else {}
    out = {}
    for r in tqdm(recs, desc="  hashing"):
        iid = r["item_id"]
        if iid in cache:
            out[iid] = cache[iid]
            continue
        hs = []
        for rel in r["images"]:
            p = IMG_ROOT.parent / rel
            if p.exists():
                try:
                    hs.append(str(imagehash.phash(Image.open(p).convert("RGB"))))
                except Exception:
                    pass
        cache[iid] = hs
        out[iid] = hs
    PHASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    PHASH_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    return out


def build_groups(recs, hashes):
    n = len(recs)
    titles = [(r.get("item_name") or "")[:200] for r in recs]
    descs = [(r.get("description") or "")[:600] for r in recs]
    dsu = DSU(n)
    edges = {"title": 0, "desc": 0, "image_text": 0}

    print("  text similarity (titles)...")
    tm = process.cdist(titles, titles, scorer=fuzz.token_sort_ratio,
                       score_cutoff=TITLE_STRONG, workers=-1)
    print("  text similarity (descriptions)...")
    dm = process.cdist(descs, descs, scorer=fuzz.token_sort_ratio,
                       score_cutoff=DESC_COPIED, workers=-1)

    for i in range(n):
        for j in range(i + 1, n):
            if tm[i][j] >= TITLE_STRONG:
                dsu.union(i, j); edges["title"] += 1
            elif dm[i][j] >= DESC_COPIED:
                dsu.union(i, j); edges["desc"] += 1

    print("  image + weak-text edges...")
    hobj = {r["item_id"]: [imagehash.hex_to_hash(h) for h in hashes.get(r["item_id"], [])]
            for r in recs}
    ids = [r["item_id"] for r in recs]
    for i in tqdm(range(n), desc="  image pairs"):
        hi = hobj[ids[i]]
        if not hi:
            continue
        for j in range(i + 1, n):
            if dsu.find(i) == dsu.find(j):
                continue
            hj = hobj[ids[j]]
            if not hj:
                continue
            close = any((a - b) <= IMG_HAMMING for a in hi for b in hj)
            if close and fuzz.token_sort_ratio(titles[i], titles[j]) >= IMG_TITLE_MIN:
                dsu.union(i, j); edges["image_text"] += 1

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)
    return list(groups.values()), edges


def assign(groups, recs):
    """Assign whole groups to splits, preserving category proportions."""
    import random
    rng = random.Random(SEED)

    cat_total = defaultdict(int)
    for r in recs:
        cat_total[r.get("category", "?")] += 1
    quota = {c: [n * RATIOS[0], n * RATIOS[1], n * RATIOS[2]] for c, n in cat_total.items()}
    filled = {c: [0, 0, 0] for c in cat_total}

    groups = sorted(groups, key=len, reverse=True)
    rng.shuffle_ = None
    out = [[], [], []]
    for g in groups:
        cats = defaultdict(int)
        for i in g:
            cats[recs[i].get("category", "?")] += 1
        # place where the largest category is furthest below quota
        best, best_deficit = 0, None
        for s in range(3):
            deficit = sum(quota[c][s] - filled[c][s] for c in cats)
            if best_deficit is None or deficit > best_deficit:
                best, best_deficit = s, deficit
        out[best].extend(g)
        for c, k in cats.items():
            filled[c][best] += k
    return out


def verify(splits, recs, hashes):
    """Confirm no item group straddles two splits."""
    idx = {i: s for s, ids in enumerate(splits) for i in ids}
    titles = [(r.get("item_name") or "")[:200] for r in recs]
    descs = [(r.get("description") or "")[:600] for r in recs]
    bad = 0
    tr = set(splits[0]); te = set(splits[2])
    for i in te:
        for j in tr:
            if fuzz.token_sort_ratio(titles[i], titles[j]) >= TITLE_STRONG or \
               fuzz.token_sort_ratio(descs[i], descs[j]) >= DESC_COPIED:
                bad += 1
                break
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--v2", action="store_true",
                    help="run over the expanded corpus in data/processed")
    args = ap.parse_args()

    if args.v2:
        globals()["META_FILE"] = _V2_ROOT / "metadata" / "listings_final.jsonl"
        globals()["IMG_ROOT"] = _V2_ROOT / "images"
        globals()["OUT_DIR"] = _V2_ROOT / "splits_clean"
        globals()["REPORT"] = RESULTS_DIR / "novelty_clean_split_v2.json"
        globals()["PHASH_CACHE"] = RESULTS_DIR / "novelty_phash_cache_v2.json"
        print("  corpus: v2 (data/processed)")

    print("\n" + "=" * 78)
    print("N17 — LEAKAGE-FREE SPLIT")
    print("=" * 78)

    recs = load_records()
    print(f"\n  {len(recs)} listings with an image and a description")
    hashes = hash_images(recs)
    groups, edges = build_groups(recs, hashes)

    sizes = defaultdict(int)
    for g in groups:
        sizes[len(g)] += 1
    multi = sum(1 for g in groups if len(g) > 1)
    absorbed = sum(len(g) for g in groups if len(g) > 1)
    print(f"\n  groups: {len(groups)}  (multi-item groups: {multi}, "
          f"covering {absorbed} listings)")
    print(f"  edges — title {edges['title']}, description {edges['desc']}, image+text {edges['image_text']}")
    print(f"  group size distribution: {dict(sorted(sizes.items())[:8])}")

    splits = assign(groups, recs)
    names = ["train", "val", "test"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s, name in enumerate(names):
        ids = [recs[i]["item_id"] for i in splits[s]]
        (OUT_DIR / f"{name}.txt").write_text("\n".join(ids), encoding="utf-8")
        cats = defaultdict(int)
        for i in splits[s]:
            cats[recs[i].get("category", "?")] += 1
        print(f"  {name:<6} {len(ids):>5}  {dict(sorted(cats.items()))}")

    leaks = verify(splits, recs, hashes)
    print(f"\n  residual test items with a same-item match in train: {leaks}")
    if leaks == 0:
        print("  → split is leakage-free under the grouping rule.")

    REPORT.write_text(json.dumps({
        "n_records": len(recs), "n_groups": len(groups),
        "multi_item_groups": multi, "listings_in_multi_groups": absorbed,
        "edges": edges, "residual_leaks": leaks,
        "sizes": {k: len(v) for k, v in zip(names, splits)},
        "thresholds": {"title": TITLE_STRONG, "desc": DESC_COPIED,
                       "img_hamming": IMG_HAMMING, "img_title_min": IMG_TITLE_MIN},
    }, indent=2), encoding="utf-8")
    print(f"\n  splits → {OUT_DIR}")
    print(f"  report → {REPORT}")


if __name__ == "__main__":
    main()
