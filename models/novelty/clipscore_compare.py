"""
Experiment N23 — Does CLIPScore already do what VAD does?

N8 proposed Visual Attribute Density, a lexicon-based reference-free measure,
and validated it against the Gemma judge at rho = +0.42 while ROUGE-L managed
-0.15. The obvious reviewer question is why we built a word list at all, when
CLIPScore (Hessel et al., EMNLP 2021) is the standard reference-free captioning
metric and directly measures image-text agreement in CLIP space.

Three outcomes, all useful:

    CLIPScore correlates better   -> drop VAD, use CLIPScore, cite it. The paper
                                     loses a weak contribution and gains a solid
                                     baseline.
    CLIPScore correlates similarly-> report both; VAD is a cheap, interpretable
                                     alternative that needs no GPU.
    CLIPScore fails too           -> the strongest outcome: even the standard
                                     reference-free metric cannot track grounding
                                     on this data, which motivates the whole
                                     evaluation section.

CLIPScore here is the standard formulation: 2.5 * max(0, cos(image, text)).

Run:
    python -m models.novelty.clipscore_compare
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from scipy.stats import spearmanr
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import METADATA_FILE, IMAGES_DIR, RESULTS_DIR
from models.novelty.visual_lexicon import density, VISUAL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_MODEL = "openai/clip-vit-base-patch32"
OUT = RESULTS_DIR / "novelty_clipscore_compare.json"


def load_recs():
    recs = {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs[r["item_id"]] = r
    return recs


def load_image(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return None


def main():
    print("\n" + "=" * 78)
    print("N23 — CLIPSCORE vs VISUAL ATTRIBUTE DENSITY, AGAINST THE JUDGE")
    print("=" * 78)

    recs = load_recs()
    rows = []
    for tag, fname in (("BLIP", "two_stage_results_blip.jsonl"),
                       ("CLIP-GPT2", "two_stage_results_clip_gpt2.jsonl")):
        p = RESULTS_DIR / fname
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sc, gen = d.get("stage1_scores"), d.get("description_stage1")
            rec = recs.get(str(d.get("item_id")))
            if not (isinstance(sc, dict) and sc and gen and rec):
                continue
            rows.append({"model": tag, "item_id": str(d["item_id"]), "text": gen,
                         "vg": float(sc.get("visual_grounding", 0)),
                         "fl": float(sc.get("fluency", 0)),
                         "rel": float(sc.get("relevance", 0)),
                         "ov": float(sc.get("overall", 0))})

    print(f"\n  scored Stage-1 outputs: {len(rows)}")
    print(f"  loading {CLIP_MODEL} on {DEVICE}")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)

    img_cache = {}
    kept = []
    with torch.no_grad():
        for r in tqdm(rows, desc="  CLIPScore", unit="item"):
            iid = r["item_id"]
            if iid not in img_cache:
                img_cache[iid] = load_image(recs[iid])
            img = img_cache[iid]
            if img is None:
                continue
            inp = proc(text=[r["text"][:300]], images=img, return_tensors="pt",
                       padding=True, truncation=True, max_length=77)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            out = model(**inp)
            ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            cos = float((ie * te).sum())
            r["clipscore"] = 2.5 * max(cos, 0.0)
            r["vad"] = density(r["text"], VISUAL)
            kept.append(r)

    print(f"  usable items (image found): {len(kept)}")

    judges = [("visual_grounding", "vg"), ("fluency", "fl"),
              ("relevance", "rel"), ("overall", "ov")]
    metrics = [("CLIPScore", "clipscore"), ("Visual Attribute Density", "vad")]

    print(f"\n  {'metric':<28}" + "".join(f"{j[0]:>20}" for j in judges))
    print("  " + "-" * (28 + 20 * len(judges)))
    res = {}
    for mname, mkey in metrics:
        xs = [r[mkey] for r in kept]
        cells = []
        for jname, jkey in judges:
            ys = [r[jkey] for r in kept]
            rho, p = spearmanr(xs, ys)
            res.setdefault(mname, {})[jname] = {
                "spearman_rho": round(float(rho), 3), "p_value": round(float(p), 6)}
            cells.append(f"{rho:>+18.3f}{'*' if p < 0.05 else ' '}")
        print(f"  {mname:<28}" + "".join(cells))

    rho, p = spearmanr([r["clipscore"] for r in kept], [r["vad"] for r in kept])
    res["_clipscore_vs_vad"] = {"spearman_rho": round(float(rho), 3),
                                "p_value": round(float(p), 6)}
    res["_n"] = len(kept)
    print(f"\n  CLIPScore vs VAD agreement: rho = {rho:+.3f} (p={p:.3g})")
    print(f"  mean CLIPScore {sum(r['clipscore'] for r in kept)/len(kept):.3f}   "
          f"mean VAD {sum(r['vad'] for r in kept)/len(kept):.2f}")

    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
