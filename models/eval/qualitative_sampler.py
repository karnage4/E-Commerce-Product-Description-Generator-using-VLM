import argparse
import base64
import html
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image
from rouge_score import rouge_scorer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import METADATA_FILE, IMAGES_DIR, RESULTS_DIR

RESULTS_FILES = {
    "blip":       RESULTS_DIR / "blip_results.jsonl",
    "clip_gpt2":  RESULTS_DIR / "clip_gpt2_results.jsonl",
    "two_stage":  RESULTS_DIR / "two_stage_results.jsonl",
}

OUTPUT_JSON = RESULTS_DIR / "qualitative_samples.json"
OUTPUT_HTML = RESULTS_DIR / "qualitative_report.html"

MAX_IMAGE_DIM = 300
MAX_REF_CHARS = 600
MAX_GEN_CHARS = 700
TRUNC_SUFFIX  = " ..."

CARD_STYLE = (
    "border:2px solid #333; border-radius:10px; margin:20px 0; padding:20px; "
    "font-family:monospace; background:#1a1a2e; color:#eee;"
)

HEAD = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qualitative Samples — Daraz VLM</title>
  <style>
    body  {{ background:#0d0d1a; color:#eee; font-family:monospace; padding:30px; }}
    h1    {{ color:#00d4ff; text-align:center; }}
    h2    {{ color:#aaa; text-align:center; font-size:14px; margin-top:-10px; }}
    h3.sec{{ color:#ffd700; border-bottom:1px solid #444; padding-bottom:6px; margin-top:40px; }}
    .score{{ color:#aaa; font-size:12px; margin-top:6px; }}
  </style>
</head>
<body>
  <h1>Qualitative Samples — Product Description Generation</h1>
  <h2>Daraz.pk Dataset &nbsp;|&nbsp; Model: {model_name}</h2>
"""

FOOT = "</body>\n</html>\n"


def load_metadata_index() -> dict[str, dict]:
    idx: dict[str, dict] = {}
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


def load_results(path: Path, model_key: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if model_key == "two_stage":
                rec["generated"] = rec.get("description_stage1", "")
            out.append(rec)
    return out


def compute_rouge_scores(records: list[dict]) -> list[dict]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    enriched = []
    for rec in records:
        gen = (rec.get("generated") or "").strip()
        ref = (rec.get("reference") or "").strip()
        if not ref:
            continue
        result = scorer.score(ref, gen)["rougeL"]
        enriched.append({
            **rec,
            "rouge_p": round(result.precision, 4),
            "rouge_r": round(result.recall,    4),
            "rouge_f": round(result.fmeasure,  4),
        })
    return enriched


def select_buckets(scored: list[dict]) -> dict[str, list[dict]]:
    by_f_desc   = sorted(scored, key=lambda r: r["rouge_f"], reverse=True)
    by_p_asc    = sorted(scored, key=lambda r: r["rouge_p"])
    by_r_asc    = sorted(scored, key=lambda r: r["rouge_r"])

    true_positives  = by_f_desc[:5]
    tp_ids = {r["item_id"] for r in true_positives}

    false_positives = []
    for rec in by_p_asc:
        if rec["item_id"] not in tp_ids:
            false_positives.append(rec)
        if len(false_positives) >= 5:
            break
    fp_ids = {r["item_id"] for r in false_positives}

    excluded_fn = tp_ids | fp_ids
    false_negatives = []
    for rec in by_r_asc:
        if rec["item_id"] not in excluded_fn:
            false_negatives.append(rec)
        if len(false_negatives) >= 5:
            break
    fn_ids = {r["item_id"] for r in false_negatives}

    excluded_hard = tp_ids | fp_ids | fn_ids
    hard_candidates = [r for r in reversed(by_f_desc) if r["item_id"] not in excluded_hard]

    seen_cats: set[str] = set()
    hard: list[dict] = []
    for rec in hard_candidates:
        cat = rec.get("category", "unknown")
        if cat not in seen_cats:
            seen_cats.add(cat)
            hard.append(rec)
        if len(hard) >= 3:
            break

    if len(hard) < 3:
        for rec in hard_candidates:
            if rec not in hard:
                hard.append(rec)
            if len(hard) >= 3:
                break

    return {
        "true_positives":  true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hard_cases":      hard,
    }


def encode_image(meta_rec: dict) -> str:
    data_root = IMAGES_DIR.parent
    for rel in meta_rec.get("images", []):
        path = data_root / rel.replace("\\", "/")
        if not path.exists():
            item_id = meta_rec.get("item_id", "")
            fallback = IMAGES_DIR / item_id / "0.jpg"
            if fallback.exists():
                path = fallback
            else:
                continue
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            continue

    return (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            b'<svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">'
            b'<rect width="240" height="240" fill="#555"/>'
            b'<text x="120" y="125" font-size="14" fill="#aaa" '
            b'text-anchor="middle" font-family="monospace">no image</text>'
            b"</svg>"
        ).decode()
    )


def _trunc(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + TRUNC_SUFFIX


def _safe(s: str) -> str:
    return html.escape(s or "").replace("\n", "<br>")


def _build_card(rec: dict, meta_rec: dict, idx: int) -> str:
    img_src   = encode_image(meta_rec) if meta_rec else encode_image({})
    category  = html.escape(rec.get("category", "unknown"))
    generated = _safe(_trunc(rec.get("generated", ""), MAX_GEN_CHARS))
    reference = _safe(_trunc(rec.get("reference", ""), MAX_REF_CHARS))
    prec = rec.get("rouge_p", 0.0)
    rec_ = rec.get("rouge_r", 0.0)
    f1   = rec.get("rouge_f", 0.0)
    score_line = f"ROUGE-L  precision={prec:.3f}  recall={rec_:.3f}  F1={f1:.3f}"

    return f"""
<div style="{CARD_STYLE}">
  <h3 style="color:#00d4ff; margin:0 0 12px 0;">{idx}. {category}</h3>
  <div style="display:flex; gap:20px; align-items:flex-start;">
    <div style="flex-shrink:0;">
      <img src="{img_src}" style="width:240px; border-radius:8px; border:1px solid #555;" />
    </div>
    <div style="flex:1; font-size:13px; line-height:1.7;">
      <div style="margin-bottom:10px; background:#0f3460; padding:10px; border-radius:6px;">
        <span style="color:#ffd700; font-weight:bold;">REFERENCE</span><br>
        <span style="color:#e0e0e0;">{reference}</span>
      </div>
      <div style="margin-bottom:10px; background:#162447; padding:10px; border-radius:6px;">
        <span style="color:#4fc3f7; font-weight:bold;">GENERATED</span><br>
        <span style="color:#e0e0e0;">{generated}</span>
      </div>
      <div class="score">{html.escape(score_line)}</div>
    </div>
  </div>
</div>"""


def _build_section(title: str, color: str, records: list[dict], metadata_index: dict) -> str:
    parts = [f'<h3 class="sec" style="color:{color};">{html.escape(title)}</h3>']
    for i, rec in enumerate(records, 1):
        meta_rec = metadata_index.get(rec.get("item_id", ""), {})
        parts.append(_build_card(rec, meta_rec, i))
    return "\n".join(parts)


def build_html(buckets: dict[str, list[dict]], meta_index: dict, model_name: str) -> str:
    sections = [
        ("True Positives — highest ROUGE-L F1 (model got it right)",
         "#00e676", buckets["true_positives"]),
        ("False Positives — lowest ROUGE-L precision (hallucinations / off-topic)",
         "#ff7043", buckets["false_positives"]),
        ("False Negatives — lowest ROUGE-L recall (missed reference content)",
         "#ffb300", buckets["false_negatives"]),
        ("Hard Cases — lowest F1, diverse categories",
         "#ce93d8", buckets["hard_cases"]),
    ]
    body_parts = []
    for title, color, records in sections:
        body_parts.append(_build_section(title, color, records, meta_index))

    return HEAD.format(model_name=html.escape(model_name)) + "\n".join(body_parts) + FOOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RESULTS_FILES), default="blip")
    args = ap.parse_args()

    results_path = RESULTS_FILES[args.model]
    if not results_path.exists():
        sys.exit(
            f"[!] Results file not found: {results_path}\n"
            f"    Run the corresponding evaluate.py first."
        )

    print(f"  Loading results from {results_path.name}...")
    records = load_results(results_path, args.model)
    print(f"    {len(records)} records loaded")

    print("  Computing ROUGE-L scores...")
    scored = compute_rouge_scores(records)
    print(f"    {len(scored)} scored (skipped {len(records) - len(scored)} with empty reference)")

    if len(scored) < 5:
        sys.exit(f"[!] Not enough valid records ({len(scored)}) to fill all buckets.")

    print("  Selecting samples...")
    buckets = select_buckets(scored)
    for key, recs in buckets.items():
        print(f"    {key}: {len(recs)} items")

    print(f"  Loading metadata index from {METADATA_FILE.name}...")
    meta_index = load_metadata_index()
    print(f"    {len(meta_index)} records")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "true_positives":  buckets["true_positives"],
        "false_positives": buckets["false_positives"],
        "false_negatives": buckets["false_negatives"],
        "hard_cases":      buckets["hard_cases"],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved JSON  -> {OUTPUT_JSON}")

    html_str = build_html(buckets, meta_index, model_name=args.model)
    OUTPUT_HTML.write_text(html_str, encoding="utf-8")
    print(f"  Saved HTML  -> {OUTPUT_HTML}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
