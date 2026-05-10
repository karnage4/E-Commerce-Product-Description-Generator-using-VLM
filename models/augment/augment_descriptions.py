"""
Caption augmentation via NVIDIA NIM (meta/llama-3.2-11b-vision-instruct).

Why this exists
---------------
The raw Daraz descriptions in listings_final.jsonl are mostly metadata
echoes (often the description == the item_name). Training a VLM on these
teaches it to ignore the image and copy the metadata back. This script
uses a strong vision-language model to rewrite each training description
so it actually mentions the visible features of the product, giving
fine-tuning a real visual grounding signal.

Inputs
------
  data/processed/metadata/listings_final.jsonl   (full corpus)
  data/processed/splits/train.txt                (which item_ids to augment)
  data/processed/images/<item_id>/0.jpg          (first image per product)

Output
------
  data/processed/metadata/listings_augmented.jsonl

Each output row is the original record PLUS:
  - description_augmented  : the new rich, visually-grounded description
  - description_original   : copy of the source description (for reference)

Resumable: re-running skips item_ids already present in the output file.

Setup
-----
  1. Add NVIDIA_API_KEY to .env (get one at https://build.nvidia.com/)
  2. pip install openai python-dotenv pillow tqdm
  3. python -m models.augment.augment_descriptions

Tuning knobs (see CONFIG block):
  - MODEL              : NIM model identifier
  - RATE_LIMIT_DELAY   : seconds between requests (raise if you hit 429)
  - MAX_IMAGE_DIM      : longest image side after resize (smaller = faster, smaller payload)
  - JPEG_QUALITY       : initial JPEG quality (drops to 60 if payload > limit)
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_ROOT     = PROJECT_ROOT / "data" / "processed"
METADATA_FILE = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_ROOT   = DATA_ROOT / "images"
TRAIN_SPLIT   = DATA_ROOT / "splits" / "train.txt"
OUTPUT_FILE   = DATA_ROOT / "metadata" / "listings_augmented.jsonl"


# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL              = "meta/llama-3.2-11b-vision-instruct"
BASE_URL           = "https://integrate.api.nvidia.com/v1"
RATE_LIMIT_DELAY   = 0.5      # seconds between requests
MAX_IMAGE_DIM      = 512      # longest side after resize
JPEG_QUALITY       = 80
MAX_PAYLOAD_KB     = 150      # NIM's inline image limit is ~180 KB
MAX_OUTPUT_TOKENS  = 400
TEMPERATURE        = 0.3
MAX_RETRIES        = 3


# ── Augmentation prompt ───────────────────────────────────────────────────────
# Notes on prompt design:
#   - Asks the model to ground in visible features (color, material, design,
#     ports, branding visible on the product, accessories shown in the photo).
#   - Hard constraints to suppress hallucination ("do NOT invent specs").
#   - Excludes price (we don't want models to learn to predict price; price
#     should come from metadata at inference, not be hallucinated).
#   - Asks for a flowing prose paragraph, not bullets — matches Daraz style
#     and avoids the model getting stuck in list-format ruts.
AUGMENTATION_PROMPT = """\
You are writing a product description for Daraz Pakistan, an e-commerce site. \
You will see ONE product image and structured metadata.

Write a 70-120 word product description that:
- Describes specific visual features visible in the image: color, material or \
finish, shape, design elements, visible branding, ports or buttons, attachments \
or accessories shown.
- Incorporates relevant facts from the metadata (product type, category, brand \
if it is a real brand and not "No Brand").
- Uses an engaging, factual tone suitable for Pakistani online shoppers.
- Does NOT invent specifications that are not visible in the image or stated \
in the metadata.
- Does NOT mention price, ratings, or shipping.
- Writes in flowing prose, no bullet points, no markdown headings.
- Starts with the product itself (e.g. "This <product>..."), not "The image \
shows" or "I can see".

METADATA:
{metadata}

DESCRIPTION:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_metadata_str(rec: dict) -> str:
    """Compact metadata representation for the prompt."""
    parts = []
    if rec.get("item_name"):
        parts.append(f"Name: {rec['item_name']}")
    brand = rec.get("brand")
    if brand and brand.strip().lower() not in {"no brand", "", "n/a"}:
        parts.append(f"Brand: {brand}")
    if rec.get("category"):
        parts.append(f"Category: {rec['category']}")
    if rec.get("subcategory"):
        parts.append(f"Subcategory: {rec['subcategory']}")
    attrs = rec.get("attributes") or {}
    for k, v in list(attrs.items())[:6]:
        if k and v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts) if parts else "(no metadata)"


def normalize_image_path(rel: str) -> Path:
    """Handle Windows-style backslashes in image paths."""
    rel = rel.replace("\\", "/")
    return DATA_ROOT / rel


def load_first_image_b64(rec: dict) -> str | None:
    """
    Load first available image, resize, JPEG-encode, base64-encode.
    Returns None if no readable image is found.
    """
    for rel in rec.get("images", []):
        path = normalize_image_path(rel)
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)

            # Encode at default quality
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            data = buf.getvalue()

            # Re-encode at lower quality if over the inline-image limit
            if len(data) > MAX_PAYLOAD_KB * 1024:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=60)
                data = buf.getvalue()

            return base64.b64encode(data).decode()
        except Exception:
            continue
    return None


def augment_one(client: OpenAI, rec: dict) -> str | None:
    """
    Call NIM for one record. Returns the augmented description, or None if
    the call failed after MAX_RETRIES or no image could be loaded.
    """
    image_b64 = load_first_image_b64(rec)
    if image_b64 is None:
        return None

    metadata_str = build_metadata_str(rec)
    prompt_text  = AUGMENTATION_PROMPT.format(metadata=metadata_str)

    # NIM's vision Llama models expect the image inline as an HTML <img> tag
    # within the user message string, NOT as a separate image_url block.
    user_content = (
        f'{prompt_text} <img src="data:image/jpeg;base64,{image_b64}" />'
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
                top_p=0.9,
                stream=False,
            )
            text = resp.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 5  # 5s, 10s
                tqdm.write(f"  retry {attempt+1}/{MAX_RETRIES} after error: {e}")
                time.sleep(wait)
            else:
                tqdm.write(f"  ! gave up on {rec.get('item_id')}: {e}")
                return None
    return None


def load_records_to_augment() -> list[dict]:
    train_ids = set(TRAIN_SPLIT.read_text(encoding="utf-8").strip().splitlines())
    out = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("item_id") not in train_ids:
                continue
            if not rec.get("images"):
                continue
            out.append(rec)
    return out


def load_done_ids() -> set[str]:
    if not OUTPUT_FILE.exists():
        return set()
    done = set()
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["item_id"])
            except Exception:
                pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit(
            "[!] NVIDIA_API_KEY not set.\n"
            "    1. Get a key at https://build.nvidia.com/\n"
            "    2. Add  NVIDIA_API_KEY=nvapi-...  to .env at project root"
        )

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    records  = load_records_to_augment()
    done_ids = load_done_ids()
    todo     = [r for r in records if r["item_id"] not in done_ids]

    print(f"  Train records with images : {len(records)}")
    print(f"  Already augmented         : {len(done_ids)}")
    print(f"  Remaining                 : {len(todo)}")
    if not todo:
        print("  Nothing to do.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    failed = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc="Augmenting", unit="item"):
            augmented = augment_one(client, rec)
            if augmented is None:
                failed += 1
                continue

            new_rec = dict(rec)
            new_rec["description_original"]  = rec.get("description", "")
            new_rec["description_augmented"] = augmented
            out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            time.sleep(RATE_LIMIT_DELAY)

    print(f"\n  Done. Failed: {failed}/{len(todo)}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
