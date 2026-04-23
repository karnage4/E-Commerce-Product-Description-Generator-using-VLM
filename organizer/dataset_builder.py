"""
Dataset builder — organises deduplicated products into an ABO-style structure:

data/processed/
├── images/
│   └── {item_id}/
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
├── metadata/
│   ├── listings.jsonl       ← one JSON object per line (full metadata)
│   └── listings_clean.jsonl ← same but description-only subset (for eval)
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt

Each listings.jsonl record:
{
  "item_id":     "123456",
  "item_name":   "...",
  "category":    "electronics",
  "brand":       "...",
  "seller":      "...",
  "price_pkr":   1200.0,
  "rating":      4.2,
  "num_reviews": 87,
  "description": "...",
  "attributes":  {"Color": "Black", "Material": "Plastic", ...},
  "images":      ["images/123456/0.jpg", "images/123456/1.jpg"]
}
"""

import io
import json
import os
import random
import time
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

import config

_HEADERS = {"User-Agent": config.HEADERS["User-Agent"]}


# ── Image downloading ─────────────────────────────────────────────────────────

def _download_image(url: str, dest: Path) -> bool:
    """Download a single image to `dest`. Returns True on success."""
    if not url:
        return False
    try:
        time.sleep(random.uniform(0.2, 0.6))
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        if resp.status_code != 200:
            return False
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img.save(dest, format="JPEG", quality=90)
        return True
    except Exception:
        return False


def _download_product_images(item_id: str, image_urls: list[str]) -> list[str]:
    """
    Downloads all images for a product into images/{item_id}/.
    Returns relative paths to saved images.
    """
    img_dir = Path(config.IMAGES_DIR) / item_id
    img_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, url in enumerate(image_urls):
        dest = img_dir / f"{i}.jpg"
        if dest.exists():
            saved_paths.append(str(dest.relative_to(config.PROCESSED_DIR)))
            continue
        if _download_image(url, dest):
            saved_paths.append(str(dest.relative_to(config.PROCESSED_DIR)))

    return saved_paths


# ── Dataset assembly ──────────────────────────────────────────────────────────

def build_dataset(products: list[dict] = None) -> None:
    """
    Main entry point. For each product:
      1. Download images
      2. Write final metadata record to listings_final.jsonl
    Then produce train/val/test splits.

    If `products` is None, reads from listings_clean.jsonl (output of the
    clean step).
    """
    Path(config.METADATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.SPLITS_DIR).mkdir(parents=True, exist_ok=True)

    # Default: read the cleaned data
    if products is None:
        clean_path = Path(config.METADATA_DIR) / "listings_clean.jsonl"
        if not clean_path.exists():
            print("  [!] No listings_clean.jsonl found. Run the clean step first.")
            return
        products = []
        with open(clean_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    products.append(json.loads(line))
        print(f"  Loaded {len(products)} cleaned products")

    final_path = Path(config.METADATA_DIR) / "listings_final.jsonl"

    # Load already-processed item_ids to allow resumption
    processed_ids = set()
    if final_path.exists():
        with open(final_path, encoding="utf-8") as f:
            for line in f:
                try:
                    processed_ids.add(json.loads(line)["item_id"])
                except Exception:
                    pass
        print(f"  Resuming — {len(processed_ids)} already downloaded")

    with open(final_path, "a", encoding="utf-8") as f_out:
        for product in tqdm(products, desc="Downloading images", unit="product"):
            item_id = product["item_id"]
            if item_id in processed_ids:
                continue

            image_urls = product.get("all_images") or [product.get("image_url", "")]
            image_paths = _download_product_images(item_id, image_urls)

            if not image_paths:
                continue  # Skip products where all image downloads failed

            record = {
                "item_id":      item_id,
                "item_name":    product.get("name", ""),
                "category":     product.get("category", ""),
                "subcategory":  product.get("subcategory", "Other"),
                "brand":        product.get("brand_name", product.get("brand", "")),
                "seller":       product.get("seller_name", product.get("seller", "")),
                "price_pkr":    product.get("price_pkr", 0.0),
                "rating":       product.get("rating", 0.0),
                "num_reviews":  product.get("num_reviews", 0),
                "description":  product.get("description", ""),
                "attributes":   product.get("attributes", {}),
                "images":       image_paths,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_ids.add(item_id)

    print(f"\n  {len(processed_ids)} products with images → {final_path}")
    _write_splits(final_path)


# ── Train / val / test splits ─────────────────────────────────────────────────

def _write_splits(clean_path: Path) -> None:
    """
    Reads listings_clean.jsonl, shuffles, and writes item_id lists to
    splits/train.txt, splits/val.txt, splits/test.txt.
    Stratified by category so each split has similar category distribution.
    """
    from collections import defaultdict

    by_category: dict[str, list[str]] = defaultdict(list)
    with open(clean_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                by_category[rec["category"]].append(rec["item_id"])
            except Exception:
                pass

    train_ids, val_ids, test_ids = [], [], []

    for cat, ids in by_category.items():
        random.shuffle(ids)
        n = len(ids)
        n_val  = max(1, int(n * config.VAL_RATIO))
        n_test = max(1, int(n * config.TEST_RATIO))

        test_ids  += ids[:n_test]
        val_ids   += ids[n_test:n_test + n_val]
        train_ids += ids[n_test + n_val:]

    splits = {
        "train": train_ids,
        "val":   val_ids,
        "test":  test_ids,
    }

    splits_dir = Path(config.SPLITS_DIR)
    for split_name, ids in splits.items():
        path = splits_dir / f"{split_name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids) + "\n")
        print(f"  Split [{split_name}]: {len(ids)} items → {path}")
