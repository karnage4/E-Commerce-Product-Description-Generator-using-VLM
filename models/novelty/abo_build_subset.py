"""
Builds a controlled ABO subset so the blind-vs-vision experiment can be repeated
on Amazon's catalogue rather than only on Daraz.

Samples English ABO listings that have a description and a main image, resolves
each image to its S3 path, downloads the small (max-256px) rendition, and writes
train/val/test splits in the same JSONL shape the DPD pipeline uses — so the
existing training and evaluation code runs on it unchanged.

Requires the listing shards and images.csv.gz already in data/abo/ (see
models/novelty/abo_replication.py for the download commands).

Run:
    python -m models.novelty.abo_build_subset --n 15000
"""

import argparse
import csv
import gzip
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.novelty.abo_replication import abo_metadata_prompt, abo_reference, en_values

ROOT = Path(__file__).resolve().parents[2]
ABO_DIR = ROOT / "data" / "abo"
OUT_DIR = ROOT / "data" / "abo_subset"
IMG_DIR = OUT_DIR / "images"
S3 = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"
SEED = 42


def load_image_index():
    idx = {}
    with gzip.open(ABO_DIR / "images.csv.gz", "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx[row["image_id"]] = row["path"]
    return idx


def sample_records(n, img_index):
    pool = []
    for fp in tqdm(sorted(ABO_DIR.glob("listings_*.json.gz")), desc="  scanning shards"):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mid = d.get("main_image_id")
                if not mid or mid not in img_index:
                    continue
                ref = abo_reference(d)
                if len(ref.split()) < 10:
                    continue
                prompt = abo_metadata_prompt(d)
                if not prompt:
                    continue
                pool.append({
                    "item_id": d["item_id"],
                    "item_name": en_values(d.get("item_name")),
                    "brand": en_values(d.get("brand")),
                    "category": en_values(d.get("product_type")).lower(),
                    "subcategory": en_values(d.get("style")),
                    "description": ref,
                    "_prompt": prompt,
                    "_img": img_index[mid],
                })
    print(f"  eligible listings: {len(pool)}")
    rng = random.Random(SEED)
    rng.shuffle(pool)
    return pool[:n]


def fetch(rec):
    dest = IMG_DIR / rec["item_id"] / "0.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(S3 + rec["_img"], timeout=25)
        if r.status_code == 200 and r.content:
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("BUILDING CONTROLLED ABO SUBSET")
    print("=" * 78)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  loading image index...")
    img_index = load_image_index()
    print(f"  {len(img_index)} images indexed")

    recs = sample_records(args.n, img_index)
    print(f"  sampled {len(recs)}")

    print("\n  downloading images...")
    ok = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec, good in tqdm(zip(recs, ex.map(fetch, recs)), total=len(recs), desc="  images"):
            if good:
                ok.append(rec)
    print(f"  downloaded {len(ok)}/{len(recs)}")

    # Write in the DPD record shape so existing loaders work unchanged.
    meta_dir = OUT_DIR / "metadata"
    split_dir = OUT_DIR / "splits"
    meta_dir.mkdir(exist_ok=True)
    split_dir.mkdir(exist_ok=True)

    with open(meta_dir / "listings_final.jsonl", "w", encoding="utf-8") as f:
        for r in ok:
            out = {k: v for k, v in r.items() if not k.startswith("_")}
            out["images"] = [f"images/{r['item_id']}/0.jpg"]
            out["abo_metadata_prompt"] = r["_prompt"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    rng = random.Random(SEED)
    ids = [r["item_id"] for r in ok]
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(0.8 * n), int(0.9 * n)
    for name, chunk in (("train", ids[:a]), ("val", ids[a:b]), ("test", ids[b:])):
        (split_dir / f"{name}.txt").write_text("\n".join(chunk), encoding="utf-8")
        print(f"  {name}: {len(chunk)}")

    print(f"\nSaved → {OUT_DIR}")


if __name__ == "__main__":
    main()
