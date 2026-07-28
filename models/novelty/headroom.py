"""
Experiment N24 — Headroom: how much can a generator actually add?

Every paper in this space scores a generated description by its similarity to
the seller's existing description. That silently defines success as reproducing
the incumbent. It is a reasonable definition on a curated catalogue, where the
incumbent is good. It is the wrong definition on a marketplace, where the
incumbent is the problem being solved.

The deployment-relevant quantity is not "how close is the output to the existing
listing" but "how much better is it than the existing listing":

    headroom(corpus) = CLIPScore(model output, image) - CLIPScore(seller text, image)

CLIPScore is used rather than the lexicon metric because N23 showed it tracks
judged visual grounding better (rho +0.52 vs +0.42) and is a standard, citable
measure.

Two corpora are compared:

    ABO    incumbent = Amazon's own bullet points, professionally written
    DPD    incumbent = marketplace seller text, 56.8% of it re-listed from
           another seller

If headroom is large on DPD and small on ABO, then the same model has very
different deployment value in the two markets — and the market with the worse
incumbent is the one where automation is worth the most, which is the opposite
of where the field does its work.

Run:
    python -m models.novelty.headroom
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import METADATA_FILE, IMAGES_DIR, RESULTS_DIR, build_metadata_prompt
from models.novelty.prompt_prefix_audit import strip_prefix

ROOT = Path(__file__).resolve().parents[2]
ABO_DIR = ROOT / "data" / "abo_subset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL = "openai/clip-vit-base-patch32"
OUT = RESULTS_DIR / "novelty_headroom.json"

_model = _proc = None


def clip_init():
    global _model, _proc
    if _model is None:
        _model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
        _proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    return _model, _proc


@torch.no_grad()
def clipscore(pairs, desc="  scoring"):
    """pairs: list of (PIL image, text). Returns list of 2.5*max(cos,0)."""
    model, proc = clip_init()
    out = []
    for img, text in tqdm(pairs, desc=desc, unit="item", leave=False):
        if img is None or not text.strip():
            out.append(None)
            continue
        inp = proc(text=[text[:300]], images=img, return_tensors="pt",
                   padding=True, truncation=True, max_length=77)
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        o = model(**inp)
        ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        out.append(2.5 * max(float((ie * te).sum()), 0.0))
    return out


def mean(xs):
    v = [x for x in xs if x is not None]
    return round(sum(v) / len(v), 4) if v else None


# ── Daraz ─────────────────────────────────────────────────────────────────────

def daraz():
    recs = {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if (r.get("description") or "").strip() and r.get("images"):
                    recs[r["item_id"]] = r

    def img(rec):
        for rel in rec.get("images", []):
            p = IMAGES_DIR.parent / rel
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    continue
        return None

    def load(fname, key):
        out, p = {}, RESULTS_DIR / fname
        if not p.exists():
            return out
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get(key):
                    out[str(d["item_id"])] = d[key]
        return out

    blip = load("blip_results.jsonl", "generated")
    s1 = load("two_stage_results_blip.jsonl", "description_stage1")
    s2 = load("two_stage_results_blip.jsonl", "description_stage2")
    ids = [i for i in blip if i in recs and i in s1 and i in s2]
    print(f"\n  DPD items with all systems: {len(ids)}")

    images = {i: img(recs[i]) for i in ids}
    out = {}
    out["incumbent (seller text)"] = mean(clipscore(
        [(images[i], recs[i]["description"]) for i in ids], "  DPD seller"))
    out["BLIP (prefix stripped)"] = mean(clipscore(
        [(images[i], strip_prefix(blip[i], build_metadata_prompt(recs[i]))[0]) for i in ids],
        "  DPD blip"))
    out["Stage-1 (image-forced)"] = mean(clipscore(
        [(images[i], s1[i]) for i in ids], "  DPD stage1"))
    out["two-stage"] = mean(clipscore(
        [(images[i], s2[i]) for i in ids], "  DPD two-stage"))
    out["_n"] = len(ids)
    return out


# ── ABO ───────────────────────────────────────────────────────────────────────

def abo():
    gen_path = RESULTS_DIR / "novelty_abo_generations.jsonl"
    if not gen_path.exists():
        print("\n  [skip] no ABO generations on disk")
        return None
    rows = [json.loads(l) for l in open(gen_path, encoding="utf-8")]
    pairs_ref, pairs_gen = [], []
    kept = 0
    for r in rows:
        p = ABO_DIR / "images" / str(r["item_id"]) / "0.jpg"
        if not p.exists():
            continue
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        pairs_ref.append((im, r["reference"]))
        pairs_gen.append((im, r["real"]))
        kept += 1
    print(f"\n  ABO items with images: {kept}")
    if kept == 0:
        return None
    return {"incumbent (Amazon bullets)": mean(clipscore(pairs_ref, "  ABO reference")),
            "ABO-trained BLIP": mean(clipscore(pairs_gen, "  ABO blip")),
            "_n": kept}


def main():
    print("\n" + "=" * 78)
    print("N24 — HEADROOM: VALUE ADDED OVER THE INCUMBENT LISTING")
    print("=" * 78)

    d = daraz()
    a = abo()

    print("\n  [Daraz — marketplace, poor incumbent]")
    base = d["incumbent (seller text)"]
    for k, v in d.items():
        if k.startswith("_"):
            continue
        delta = "" if k.startswith("incumbent") else f"   headroom {v-base:+.4f}  ({100*(v-base)/base:+.1f}%)"
        print(f"    {k:<28}CLIPScore {v}{delta}")

    if a:
        print("\n  [ABO — curated catalogue, strong incumbent]")
        abase = a["incumbent (Amazon bullets)"]
        for k, v in a.items():
            if k.startswith("_"):
                continue
            delta = "" if k.startswith("incumbent") else f"   headroom {v-abase:+.4f}  ({100*(v-abase)/abase:+.1f}%)"
            print(f"    {k:<28}CLIPScore {v}{delta}")

        best_d = max(v for k, v in d.items() if not k.startswith("_") and not k.startswith("incumbent"))
        print("\n  [Verdict]")
        print(f"    best achievable gain over incumbent — Daraz: {100*(best_d-base)/base:+.1f}%"
              f"   Amazon: {100*(a['ABO-trained BLIP']-abase)/abase:+.1f}%")

    OUT.write_text(json.dumps({"daraz": d, "abo": a}, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
