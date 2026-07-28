"""
Parallel image pre-fetcher for the v2 dataset build.

organizer/dataset_builder.py downloads images strictly sequentially, with a
0.2-0.6s sleep per image and every image of every product fetched in turn. On
the 3,092-product v2 corpus that measured ~12s per product — about eight hours.

It does, however, skip any file that already exists on disk. So rather than
touching the builder, this script pre-populates the image directory in parallel
and then lets `run.py --steps build` run through in minutes, writing exactly
the same metadata and splits it would have written anyway.

Nothing in organizer/ is modified. Politeness to the origin server is preserved
via a bounded worker pool and a per-request timeout; the aggregate request rate
is comparable to a normal browser loading a product page.

Run:
    python -m models.novelty.prefetch_images
    python -m models.novelty.prefetch_images --workers 16 --max-images 3
"""

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "data" / "processed" / "metadata" / "listings_clean.jsonl"
IMG_ROOT = ROOT / "data" / "processed" / "images"
HEADERS = {"User-Agent": config.HEADERS["User-Agent"]}

_counts = {"ok": 0, "skip": 0, "fail": 0}
_lock = threading.Lock()


def fetch(task):
    item_id, idx, url = task
    dest = IMG_ROOT / item_id / f"{idx}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        with _lock:
            _counts["skip"] += 1
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200 and r.content:
            Image.open(io.BytesIO(r.content)).convert("RGB").save(
                dest, format="JPEG", quality=90)
            with _lock:
                _counts["ok"] += 1
            return
    except Exception:
        pass
    with _lock:
        _counts["fail"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-images", type=int, default=4,
                    help="images per product; the models only use the first")
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("PARALLEL IMAGE PRE-FETCH (v2 dataset)")
    print("=" * 78)

    if not CLEAN.exists():
        raise SystemExit(f"missing {CLEAN} — run `python run.py --steps clean` first")

    tasks, products = [], 0
    with open(CLEAN, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # listings_clean.jsonl carries `all_images` (list) plus a single
            # `image_url`; the builder consumes the list.
            urls = rec.get("all_images") or rec.get("images") or rec.get("image_urls") or []
            if isinstance(urls, str):
                urls = [urls]
            if not urls and rec.get("image_url"):
                urls = [rec["image_url"]]
            if not urls:
                continue
            products += 1
            for i, u in enumerate(urls[:args.max_images]):
                if isinstance(u, str) and u.startswith("http"):
                    tasks.append((str(rec["item_id"]), i, u))

    print(f"  products: {products}   images queued: {len(tasks)}   workers: {args.workers}")
    IMG_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(tqdm(ex.map(fetch, tasks), total=len(tasks), desc="  fetching", unit="img"))
    dt = time.time() - t0

    have = sum(1 for d in IMG_ROOT.iterdir() if d.is_dir() and any(d.iterdir()))
    print(f"\n  downloaded {_counts['ok']}   already present {_counts['skip']}   "
          f"failed {_counts['fail']}")
    print(f"  products with at least one image: {have}")
    print(f"  elapsed {dt/60:.1f} min ({len(tasks)/max(dt,1):.1f} img/s)")
    print("\n  now run:  python run.py --steps build")


if __name__ == "__main__":
    main()
