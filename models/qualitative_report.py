"""
Build a styled HTML qualitative report from one or more model results JSONL files.

Output mirrors the team's reference format: dark cards with the product image,
metadata input, gold-bordered reference description, and color-coded model outputs.

Inputs
------
  data/processed/metadata/listings_final.jsonl     (for image paths + full metadata)
  models/results/blip_results.jsonl                (BLIP outputs — required)
  models/results/clip_gpt2_results.jsonl           (optional, added if file exists)

Each results JSONL row should contain at least:
  {"item_id": ..., "generated": ..., "reference": ..., "category": ...}

Output
------
  qualitative_report.html  (at project root)

Usage
-----
  # Default — picks 8 samples spread across categories
  python -m models.qualitative_report

  # Custom sample count
  python -m models.qualitative_report --num 12

  # Custom seed (deterministic)
  python -m models.qualitative_report --num 6 --seed 42

After running, open `qualitative_report.html` in a browser.
"""

import argparse
import base64
import html
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
METADATA_FILE = PROJECT_ROOT / "data" / "processed" / "metadata" / "listings_final.jsonl"
DATA_ROOT     = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "models" / "results"
OUTPUT_HTML   = PROJECT_ROOT / "qualitative_report.html"


# ── Model registry ────────────────────────────────────────────────────────────
# To add another model later, drop a results JSONL into models/results/ and
# append a row here. Models with missing files are skipped silently.
MODELS = [
    {
        "name":        "BLIP Fine-tuned (Augmented)",
        "icon":        "🔵",
        "results":     RESULTS_DIR / "blip_results.jsonl",
        "label_color": "#4fc3f7",
        "bg_color":    "#162447",
    },
    {
        "name":        "CLIP-GPT2 Fine-tuned",
        "icon":        "🟢",
        "results":     RESULTS_DIR / "clip_gpt2_results.jsonl",
        "label_color": "#a5d6a7",
        "bg_color":    "#1b2838",
    },
    # Add another row here when you train BLIP-2 or any other model.
]


# ── Display config ────────────────────────────────────────────────────────────
MAX_IMAGE_DIM    = 320     # px — embedded base64 image longest side
MAX_REF_CHARS    = 500     # truncate reference descriptions for readability
MAX_MODEL_CHARS  = 700     # truncate model outputs (some loop or run long)
TRUNC_SUFFIX     = " ..."


# ── HTML chrome ───────────────────────────────────────────────────────────────
HEAD = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qualitative Evaluation — Daraz VLM</title>
  <style>
    body { background: #0d0d1a; color: #eee; font-family: monospace; padding: 30px; }
    h1   { color: #00d4ff; text-align: center; }
    h2   { color: #aaa;    text-align: center; font-size: 14px; margin-top: -10px; }
  </style>
</head>
<body>
  <h1>Qualitative Evaluation — Product Description Generation</h1>
  <h2>Daraz.pk Dataset &nbsp;|&nbsp; {subtitle}</h2>
"""

CARD_STYLE = (
    "border:2px solid #333; border-radius:10px; margin:20px 0; padding:20px; "
    "font-family:monospace; background:#1a1a2e; color:#eee;"
)

FOOT = """</body>
</html>
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + TRUNC_SUFFIX


def safe(s: str) -> str:
    """HTML-escape and convert newlines to <br> for inline rendering."""
    return html.escape(s or "").replace("\n", "<br>")


def load_metadata_index() -> dict:
    idx = {}
    if not METADATA_FILE.exists():
        sys.exit(f"[!] Metadata file not found: {METADATA_FILE}")
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx[rec["item_id"]] = rec
            except Exception:
                continue
    return idx


def load_results(path: Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def encode_image(rec: dict) -> str:
    """Find first usable image, resize for embed, return base64 PNG (no prefix)."""
    for rel in rec.get("images", []):
        rel = rel.replace("\\", "/")
        path = DATA_ROOT / rel
        if not path.exists():
            continue
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            continue
    return ""


def build_metadata_line(rec: dict) -> str:
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    return ". ".join(parts)


# ── Sample selection: one per category, then fill ─────────────────────────────

def select_diverse_samples(records: list, n: int, seed: int) -> list:
    rng = random.Random(seed)
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r.get("category", "unknown")].append(r)

    cats = list(by_cat.keys())
    rng.shuffle(cats)

    chosen = []
    seen_ids = set()

    # Pass 1: one sample per category
    for cat in cats:
        if len(chosen) >= n:
            break
        pick = rng.choice(by_cat[cat])
        if pick["item_id"] not in seen_ids:
            chosen.append(pick)
            seen_ids.add(pick["item_id"])

    # Pass 2: fill remaining slots with random others
    if len(chosen) < n:
        pool = [r for r in records if r["item_id"] not in seen_ids]
        rng.shuffle(pool)
        for r in pool:
            if len(chosen) >= n:
                break
            chosen.append(r)

    return chosen[:n]


# ── HTML card builder ─────────────────────────────────────────────────────────

def build_card(
    idx: int,
    sample: dict,
    metadata_index: dict,
    models: list,
    results_by_model: dict,
) -> str:
    item_id  = sample["item_id"]
    category = sample.get("category", "uncategorised")
    meta_rec = metadata_index.get(item_id, {})

    img_b64    = encode_image(meta_rec) if meta_rec else ""
    meta_line  = build_metadata_line(meta_rec) if meta_rec else "(metadata not found)"
    reference  = truncate(sample.get("reference", ""), MAX_REF_CHARS)

    parts = [f'<div style="{CARD_STYLE}">']
    parts.append(
        f'<h3 style="color:#00d4ff; margin:0 0 12px 0;">'
        f'Sample {idx} — {html.escape(category)}'
        f'</h3>'
    )
    parts.append('<div style="display:flex; gap:20px; align-items:flex-start;">')

    # Image column
    if img_b64:
        parts.append(
            f'<div style="flex-shrink:0;">'
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="width:240px; border-radius:8px; border:1px solid #555;" />'
            f'</div>'
        )

    # Text column
    parts.append('<div style="flex:1; font-size:13px; line-height:1.7;">')

    # Metadata input
    parts.append(
        '<div style="margin-bottom:10px;">'
        '<span style="color:#aaa; font-weight:bold;">📋 METADATA INPUT</span><br>'
        f'<span style="color:#ccc;">{safe(meta_line)}</span>'
        '</div>'
    )

    # Reference (gold)
    parts.append(
        '<div style="margin-bottom:10px; background:#0f3460; padding:10px; border-radius:6px;">'
        '<span style="color:#ffd700; font-weight:bold;">📖 REFERENCE DESCRIPTION</span><br>'
        f'<span style="color:#e0e0e0;">{safe(reference)}</span>'
        '</div>'
    )

    # Model outputs — same colour scheme as the example
    for m in models:
        match = next(
            (r for r in results_by_model[m["name"]] if r["item_id"] == item_id),
            None,
        )
        if not match:
            continue
        gen = truncate(match.get("generated") or match.get("raw") or "", MAX_MODEL_CHARS)
        parts.append(
            f'<div style="margin-bottom:10px; background:{m["bg_color"]}; '
            f'padding:10px; border-radius:6px;">'
            f'<span style="color:{m["label_color"]}; font-weight:bold;">'
            f'{m["icon"]} {html.escape(m["name"])}'
            f'</span><br>'
            f'<span style="color:#e0e0e0;">{safe(gen)}</span>'
            f'</div>'
        )

    parts.append('</div>')   # text column
    parts.append('</div>')   # flex container
    parts.append('</div>')   # card
    return "\n".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num",    type=int, default=8,  help="how many samples to include")
    ap.add_argument("--seed",   type=int, default=7,  help="random seed for sample selection")
    ap.add_argument("--output", type=Path, default=OUTPUT_HTML, help="output HTML path")
    args = ap.parse_args()

    print(f"  Loading metadata index from {METADATA_FILE.name}...")
    metadata_index = load_metadata_index()
    print(f"    {len(metadata_index)} records")

    # Load results for each configured model that has a file on disk
    active_models = []
    results_by_model: dict[str, list] = {}
    for m in MODELS:
        if not m["results"].exists():
            print(f"  [skip] {m['name']} — no file at {m['results']}")
            continue
        recs = load_results(m["results"])
        if not recs:
            print(f"  [skip] {m['name']} — empty results file")
            continue
        active_models.append(m)
        results_by_model[m["name"]] = recs
        print(f"  Loaded {len(recs)} predictions for {m['name']}")

    if not active_models:
        sys.exit(
            "[!] No model results found. Expected at least:\n"
            f"    {MODELS[0]['results']}\n"
            "    (download blip_results.jsonl from Kaggle into models/results/)"
        )

    # Pick samples using the FIRST model's results as the base set
    base_recs = results_by_model[active_models[0]["name"]]
    samples = select_diverse_samples(base_recs, args.num, args.seed)
    print(f"\n  Selected {len(samples)} diverse samples (seed={args.seed})")

    # Subtitle reflects which models are present
    subtitle = " &nbsp;vs&nbsp; ".join(m["name"] for m in active_models)

    # Build HTML
    parts = [HEAD.format(subtitle=html.escape(subtitle))]
    for i, s in enumerate(samples, 1):
        parts.append(build_card(i, s, metadata_index, active_models, results_by_model))
    parts.append(FOOT)

    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"\n  Wrote {args.output}")
    print(f"  Open in a browser to view.")


if __name__ == "__main__":
    main()
