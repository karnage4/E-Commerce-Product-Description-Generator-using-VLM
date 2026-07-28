"""
Experiment N25 — Frontier-model baseline for the Pareto claim.

The deployment argument (paper §8) currently bounds cost but not quality-per-
cost: we claim the local 224M model delivers +11% headroom at ~$0.01/1k
descriptions, but have never measured what a hosted frontier VLM delivers at
~$0.85/1k. This closes that gap.

Runs a Gemini-Flash-class multimodal model over the DPD test set via OpenRouter
(image + full metadata prompt, same prompt template as the local models), then
scores it on the same axes as everything else:

    CLIPScore + headroom over the incumbent seller text
    CompetitorSim / near-duplicate rate against the live corpus
    output length, latency, and measured API cost

Resumable: already-generated items are skipped on re-run.

Run:
    python -m models.novelty.frontier_baseline --max-items 135
    python -m models.novelty.frontier_baseline --score-only
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests as rq
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)

MODEL = "google/gemini-2.0-flash-001"
OUT_JSONL = RESULTS_DIR / "novelty_frontier_generations.jsonl"
OUT_JSON = RESULTS_DIR / "novelty_frontier_baseline.json"

PROMPT = """You are a professional e-commerce copywriter for Daraz Pakistan.
Given the product image and the following metadata, write an informative,
engaging product description in 3-5 sentences. Describe what the product
actually looks like — colour, material, shape, visible features. Do NOT
mention price. Use natural English.

Metadata:
{metadata}

Description:"""


def load_test():
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
    recs = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids and r.get("images") and (r.get("description") or "").strip():
                recs.append(r)
    return recs


def image_b64(rec, max_side=512):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((max_side, max_side))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode()
            except Exception:
                continue
    return None


def generate(recs):
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set")
    done = {}
    if OUT_JSONL.exists():
        for line in open(OUT_JSONL, encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                done[d["item_id"]] = d
        print(f"  resuming — {len(done)} already generated")

    total_cost = 0.0
    with open(OUT_JSONL, "a", encoding="utf-8") as out:
        for rec in tqdm(recs, desc="  frontier", unit="item"):
            if rec["item_id"] in done:
                continue
            b64 = image_b64(rec)
            if not b64:
                continue
            body = {
                "model": MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": PROMPT.format(metadata=build_metadata_prompt(rec))},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                }],
                "max_tokens": 300,
                "usage": {"include": True},
            }
            t0 = time.time()
            try:
                r = rq.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json=body, timeout=90)
                j = r.json()
                text = j["choices"][0]["message"]["content"].strip()
                cost = float(j.get("usage", {}).get("cost", 0.0) or 0.0)
            except Exception as e:
                print(f"    [{rec['item_id']}] failed: {str(e)[:80]}")
                time.sleep(3)
                continue
            total_cost += cost
            out.write(json.dumps({
                "item_id": rec["item_id"], "category": rec.get("category", ""),
                "generated": text, "latency_s": round(time.time() - t0, 2),
                "cost_usd": cost}, ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(0.4)
    print(f"  session API cost: ${total_cost:.4f}")


def score():
    from rapidfuzz import fuzz, process
    from models.novelty.headroom import clipscore

    recs = {r["item_id"]: r for r in load_test()}
    rows = [json.loads(l) for l in open(OUT_JSONL, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["item_id"] in recs]
    print(f"  scoring {len(rows)} frontier generations")

    def img(rec):
        for rel in rec.get("images", []):
            p = IMAGES_DIR.parent / rel
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    continue
        return None

    imgs = {r["item_id"]: img(recs[r["item_id"]]) for r in rows}
    gen_scores = clipscore([(imgs[r["item_id"]], r["generated"]) for r in rows], "  CLIP gen")
    inc_scores = clipscore([(imgs[r["item_id"]], recs[r["item_id"]]["description"]) for r in rows],
                           "  CLIP incumbent")

    ok = [(g, i) for g, i in zip(gen_scores, inc_scores) if g is not None and i is not None]
    mg = sum(g for g, _ in ok) / len(ok)
    mi = sum(i for _, i in ok) / len(ok)

    # differentiation vs the live corpus
    all_recs = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                d = json.loads(line)
                if (d.get("description") or "").strip():
                    all_recs.append((d["item_id"], d["description"][:400]))
    corpus = [t for _, t in all_recs]
    pos = {i: k for k, (i, _) in enumerate(all_recs)}
    texts = [r["generated"][:400] for r in rows]
    m = process.cdist(texts, corpus, scorer=fuzz.token_sort_ratio, workers=-1)
    comp = []
    for k, r in enumerate(rows):
        row = list(m[k])
        if r["item_id"] in pos:
            row[pos[r["item_id"]]] = 0
        comp.append(max(row))
    near = 100 * sum(1 for c in comp if c >= 70) / len(comp)

    lat = [r["latency_s"] for r in rows]
    cost = [r.get("cost_usd", 0.0) for r in rows]
    res = {
        "model": MODEL, "n": len(rows),
        "CLIPScore_generated": round(mg, 4),
        "CLIPScore_incumbent": round(mi, 4),
        "headroom_pct": round(100 * (mg - mi) / mi, 2),
        "CompetitorSim": round(float(sum(comp)) / len(comp), 1),
        "pct_near_duplicate": round(near, 1),
        "mean_latency_s": round(sum(lat) / len(lat), 2),
        "measured_cost_per_1k_usd": round(1000 * sum(cost) / max(len(cost), 1), 3),
        "mean_words": round(sum(len(r["generated"].split()) for r in rows) / len(rows), 1),
    }
    OUT_JSON.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.items():
        print(f"    {k:<28} {v}")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-items", type=int, default=135)
    ap.add_argument("--score-only", action="store_true")
    a = ap.parse_args()
    if not a.score_only:
        generate(load_test()[:a.max_items])
    score()
