"""
Caption augmentation via Gemma 4 31B (OpenRouter, multimodal).

Rewrites raw Daraz descriptions to focus on visible product features,
giving BLIP/CLIP-GPT2 a real visual grounding signal during fine-tuning.

Output: data/data/processed/metadata/listings_augmented.jsonl
Each row = original record + description_augmented + description_original

Resumable — re-running skips already-processed item_ids.

Run:
    python -m models.augment.augment_descriptions
"""

import base64
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
DATA_ROOT     = PROJECT_ROOT / "data" / "data" / "processed"
METADATA_FILE = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_ROOT   = DATA_ROOT / "images"
TRAIN_SPLIT   = DATA_ROOT / "splits" / "train.txt"
OUTPUT_FILE   = DATA_ROOT / "metadata" / "listings_augmented.jsonl"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL             = "google/gemma-4-31b-it"
BASE_URL          = "https://openrouter.ai/api/v1"
MAX_WORKERS       = 5      # parallel API calls — raise if OpenRouter allows more
RATE_LIMIT_DELAY  = 0.3    # seconds between request submissions (throttle)
MAX_IMAGE_DIM     = 768
JPEG_QUALITY      = 82
MAX_OUTPUT_TOKENS = 250
TEMPERATURE       = 0.4
MAX_RETRIES       = 3


# ── Category-specific visual feature hints ────────────────────────────────────
# Shown in the prompt to steer Gemma toward the features that matter per category.
CATEGORY_HINTS = {
    "smartphones": (
        "Look for: screen-to-body ratio / bezel thickness, notch or punch-hole style, "
        "number and arrangement of rear cameras, camera bump shape, fingerprint sensor "
        "location (side / rear / under-display), color and finish (glossy / matte / "
        "frosted glass), edge curvature (flat / curved), visible ports (USB-C / "
        "headphone jack), branding on the rear panel."
    ),
    "tablets": (
        "Look for: screen size impression, bezel thickness, front / rear camera "
        "placement, slim profile, color and finish, any attached keyboard or stylus, "
        "charging port type, speaker grilles, stand or case design."
    ),
    "consumer-electronics": (
        "Look for: overall form factor and size, color and material finish, "
        "button / knob layout, display or indicator lights, visible ports and "
        "connectors, cable or accessory included in the shot, brand logo placement."
    ),
    "home-appliances": (
        "Look for: color and surface finish (stainless / white plastic / matte black), "
        "control panel style (dial / digital / touch), door or drum design, "
        "handle shape, size impression, visible vents or coils, brand badge."
    ),
    "womens-fashion": (
        "Look for: color and color pattern (solid / print / floral / geometric), "
        "fabric texture (chiffon / cotton / silk / embroidered), neckline style "
        "(V-neck / round / square / collar), sleeve length and style, any embroidery "
        "or embellishment, hemline length, drape or silhouette."
    ),
    "mens-fashion": (
        "Look for: color and pattern (solid / striped / checked / printed), "
        "fabric texture, collar style (polo / spread / mandarin / none), "
        "button or zip closure, pocket placement, fit impression (slim / relaxed), "
        "cuffs or hem detail, visible brand label."
    ),
    "beauty-health": (
        "Look for: packaging color and shape (bottle / tube / jar / pump), "
        "lid or cap design, label color scheme, visible texture (cream / gel / "
        "liquid), applicator type (brush / dropper / roller), product color if "
        "visible, size impression."
    ),
    "travel-bags": (
        "Look for: color and material (leather / faux-leather / nylon / canvas / "
        "hard-shell), number and placement of pockets / compartments, wheel type "
        "(2-wheel / 4-spinner), handle style (retractable / top / shoulder strap), "
        "zipper color, size impression (carry-on / cabin / large), branding."
    ),
    "toys": (
        "Look for: dominant colors, character or theme design, material impression "
        "(plastic / plush / wood), size relative to hands or packaging, moving parts "
        "or accessories visible, age-group cues, any box / packaging artwork."
    ),
}
DEFAULT_HINTS = (
    "Look for: dominant colors, materials and finish, shape and form factor, "
    "visible branding, any accessories or components shown in the image."
)


# ── Prompt ────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """\
You are a product copywriter for Daraz Pakistan. Study the product image carefully.

Category: {category}
Visual features to focus on: {hints}

Metadata (use for context, do NOT just copy it):
{metadata}

Write a 70-120 word product description that:
- Leads with specific things you can SEE in the image (color, finish, shape, \
design details, visible components, branding on the product itself).
- Naturally weaves in 1-2 relevant facts from the metadata.
- Sounds engaging and natural for Pakistani online shoppers.
- Does NOT mention price, ratings, shipping, or specs you cannot see.
- Writes as flowing prose — no bullet points, no markdown.
- Starts with the product (e.g. "This sleek black smartphone..."), \
not "The image shows" or "I can see".

Description:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_metadata_str(rec: dict) -> str:
    parts = []
    if rec.get("item_name"):
        parts.append(f"Name: {rec['item_name']}")
    brand = rec.get("brand", "")
    if brand and brand.strip().lower() not in {"no brand", "", "n/a"}:
        parts.append(f"Brand: {brand}")
    if rec.get("subcategory"):
        parts.append(f"Subcategory: {rec['subcategory']}")
    attrs = rec.get("attributes") or {}
    for k, v in list(attrs.items())[:5]:
        if k and v:
            parts.append(f"{k}: {v}")
    return "\n".join(parts) if parts else "(no metadata)"


def load_image_b64(rec: dict) -> str | None:
    for rel in rec.get("images", []):
        path = DATA_ROOT / rel.replace("\\", "/")
        if not path.exists():
            # also try relative to DATA_ROOT parent
            path = DATA_ROOT.parent / rel.replace("\\", "/")
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            data = buf.getvalue()
            if len(data) > 200 * 1024:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=65)
                data = buf.getvalue()
            return base64.b64encode(data).decode()
        except Exception:
            continue
    return None


def augment_one(client: OpenAI, rec: dict) -> str | None:
    image_b64 = load_image_b64(rec)
    if image_b64 is None:
        return None

    category = rec.get("category", "")
    hints    = CATEGORY_HINTS.get(category, DEFAULT_HINTS)
    prompt   = PROMPT_TEMPLATE.format(
        category=category or "general",
        hints=hints,
        metadata=build_metadata_str(rec),
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    }]

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            )
            text = resp.choices[0].message.content
            return text.strip() if text else None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 5 * (2 ** attempt)
                tqdm.write(f"  retry {attempt+1}/{MAX_RETRIES} in {wait}s — {e}")
                time.sleep(wait)
            else:
                tqdm.write(f"  ! gave up on {rec.get('item_id')}: {e}")
    return None


def load_records() -> list[dict]:
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
    done: set[str] = set()
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
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            "[!] OPENROUTER_API_KEY not set.\n"
            "    Add  OPENROUTER_API_KEY=sk-or-...  to .env at project root."
        )

    # Each thread gets its own client (OpenAI client is not thread-safe to share)
    def make_client():
        return OpenAI(base_url=BASE_URL, api_key=api_key)

    records  = load_records()
    done_ids = load_done_ids()
    todo     = [r for r in records if r["item_id"] not in done_ids]

    print(f"  Train records : {len(records)}")
    print(f"  Already done  : {len(done_ids)}")
    print(f"  Remaining     : {len(todo)}")
    print(f"  Workers       : {MAX_WORKERS}")
    if not todo:
        print("  Nothing to do.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    write_lock = threading.Lock()
    failed_count = [0]  # mutable container for thread-safe increment
    pbar = tqdm(total=len(todo), desc="Augmenting", unit="item")

    def process(rec: dict):
        client    = make_client()
        augmented = augment_one(client, rec)
        if augmented is None:
            failed_count[0] += 1
        return rec, augmented

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {}
            for rec in todo:
                fut = ex.submit(process, rec)
                futures[fut] = rec["item_id"]
                time.sleep(RATE_LIMIT_DELAY)  # stagger submissions to avoid burst

            for fut in as_completed(futures):
                rec, augmented = fut.result()
                pbar.update(1)
                if augmented is None:
                    continue
                new_rec = dict(rec)
                new_rec["description_original"]  = rec.get("description", "")
                new_rec["description_augmented"] = augmented
                with write_lock:
                    out_f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
                    out_f.flush()

    pbar.close()
    print(f"\n  Done. Failed: {failed_count[0]}/{len(todo)}")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
