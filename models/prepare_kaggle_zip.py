"""
Bundle the Daraz dataset into a single ZIP for upload to Kaggle as a Dataset.

What it bundles
---------------
  - metadata/listings_augmented.jsonl   (preferred — has rich descriptions)
    or metadata/listings_final.jsonl    (fallback if augmentation not done)
  - splits/{train,val,test}.txt
  - images/<item_id>/*.jpg              (all product images)

Output
------
  daraz_dataset_kaggle.zip at project root.

Path normalization
------------------
The metadata records sometimes contain Windows-style image paths like
"images\\781567707\\0.jpg". This script rewrites them to forward slashes
INSIDE THE ZIP so the Kaggle notebook (Linux) can use them directly.
A normalized copy of the metadata file is written into the zip; the
on-disk source is untouched.

Upload flow
-----------
  1. python -m models.prepare_kaggle_zip
  2. Go to https://www.kaggle.com/datasets → New Dataset → upload the zip
  3. Title it e.g. "daraz-vlm-augmented"
  4. In your training notebook: Add Data → search for your dataset → Add
  5. Files appear at /kaggle/input/<your-dataset-slug>/...
"""

import json
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT     = PROJECT_ROOT / "data" / "processed"
IMAGES_DIR    = DATA_ROOT / "images"
SPLITS_DIR    = DATA_ROOT / "splits"
METADATA_DIR  = DATA_ROOT / "metadata"

AUGMENTED_FILE = METADATA_DIR / "listings_augmented.jsonl"
ORIGINAL_FILE  = METADATA_DIR / "listings_final.jsonl"

OUTPUT_ZIP = PROJECT_ROOT / "daraz_dataset_kaggle.zip"


def normalize_record(rec: dict) -> dict:
    """Rewrite Windows-style image paths to forward slashes."""
    if "images" in rec and isinstance(rec["images"], list):
        rec = dict(rec)
        rec["images"] = [p.replace("\\", "/") for p in rec["images"]]
    return rec


def merge_metadata() -> list[str]:
    """
    Build the merged metadata for the zip:
      - Start from listings_final.jsonl (ALL 1370 records: train + val + test)
      - For records present in listings_augmented.jsonl, overlay the
        `description_augmented` and `description_original` fields.

    This fixes a bug where bundling only the augmented file dropped val/test
    records (which were never augmented), causing the val DataLoader to be
    empty and the trainer to ZeroDivisionError on val_loss / len(val_loader).

    Returns a list of JSON strings (one per record), with normalized image paths.
    """
    if not ORIGINAL_FILE.exists():
        sys.exit(f"[!] {ORIGINAL_FILE} not found — can't build dataset")

    # 1. Load ALL records from the original file, keyed by item_id
    all_records: dict[str, dict] = {}
    with open(ORIGINAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = rec.get("item_id")
            if item_id:
                all_records[item_id] = rec
    print(f"  Loaded {len(all_records)} records from {ORIGINAL_FILE.name}")

    # 2. Overlay augmented descriptions onto matching records
    n_overlay = 0
    if AUGMENTED_FILE.exists():
        with open(AUGMENTED_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    aug = json.loads(line)
                except json.JSONDecodeError:
                    continue
                item_id = aug.get("item_id")
                if item_id and item_id in all_records:
                    target = all_records[item_id]
                    aug_text = (aug.get("description_augmented") or "").strip()
                    if aug_text:
                        target["description_augmented"] = aug_text
                        # Preserve the original description as a separate field
                        # so eval can score against the human-written reference.
                        target["description_original"] = (
                            aug.get("description_original")
                            or target.get("description", "")
                        )
                        n_overlay += 1
        print(f"  Overlaid augmented descriptions onto {n_overlay} records")
    else:
        print(f"  [!] {AUGMENTED_FILE.name} not found — bundling original only")

    # 3. Normalize image paths and serialize
    out_lines: list[str] = []
    for rec in all_records.values():
        rec = normalize_record(rec)
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    return out_lines


def main():
    # Build merged + path-normalized metadata in memory
    normalized_lines = merge_metadata()

    # Collect splits
    split_files = sorted(SPLITS_DIR.glob("*.txt"))
    if not split_files:
        sys.exit(f"[!] No split files in {SPLITS_DIR}")

    # Collect images
    print("  Scanning images...")
    image_files = list(IMAGES_DIR.rglob("*.jpg"))
    print(f"  Found {len(image_files)} images")

    total_mb = (
        sum(p.stat().st_size for p in split_files) +
        sum(p.stat().st_size for p in image_files) +
        sum(len(s.encode("utf-8")) for s in normalized_lines)
    ) / (1024 ** 2)
    print(f"  Total raw size: {total_mb:.0f} MB → zipping...")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Metadata (normalized)
        zf.writestr(
            "metadata/listings.jsonl",
            "\n".join(normalized_lines) + "\n",
        )

        # Splits
        for sp in split_files:
            zf.write(sp, arcname=f"splits/{sp.name}")

        # Images
        for ip in tqdm(image_files, desc="  Zipping images", unit="file"):
            arcname = ip.relative_to(DATA_ROOT)            # images/<id>/0.jpg
            zf.write(ip, arcname=str(arcname).replace("\\", "/"))

    zip_mb = OUTPUT_ZIP.stat().st_size / (1024 ** 2)
    print(f"\n  Done. {OUTPUT_ZIP.name}  ({zip_mb:.0f} MB)")
    print(f"\n  Inside the zip:")
    print(f"    metadata/listings.jsonl")
    print(f"    splits/{{train,val,test}}.txt")
    print(f"    images/<item_id>/*.jpg")
    print(f"\n  Next: upload to https://www.kaggle.com/datasets as a new dataset")


if __name__ == "__main__":
    main()
