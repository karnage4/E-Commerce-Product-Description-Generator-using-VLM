"""
Experiment N8 — Visual Attribute Density, a reference-free grounding proxy.

Additive experiment, CPU only.

Motivation
----------
N7 produced an uncomfortable result. Correlated against the Gemma-4-31B judge's
per-item visual-grounding score:

    ROUGE-L                  rho = -0.15
    Residual ROUGE-L         rho = -0.20
    Visual-Residual Recall   rho = -0.28

Every reference-based metric we have, including the corrected ones proposed in
N1, is *negatively* related to judged visual grounding. The obvious explanation
is that the references themselves are not visual: once the metadata-derivable
part is stripped out, what remains is warranty text, dimensions, shipping terms
and marketing boilerplate — not descriptions of what the product looks like.

If that is true then no reference-based metric can measure visual grounding on
this corpus, and the fix is not a better reference-based metric but a
reference-free one.

This script tests the explanation and proposes the alternative:

    Visual Attribute Density (VAD)
        the share of a text's content tokens that name a visually observable
        property — colour, material, texture, shape, or a visible part.

VAD needs no reference. We measure it for the references, the metadata prompts
and every system's output, and validate it against the judge.

Run:
    python -m models.novelty.visual_lexicon
"""

import json
import re
import sys
from pathlib import Path

import nltk
from nltk.stem.porter import PorterStemmer
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)

OUT_JSON = RESULTS_DIR / "novelty_visual_lexicon.json"

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()
WORD_RE = re.compile(r"[a-z0-9]+")

# ── Lexicons ──────────────────────────────────────────────────────────────────
# Deliberately conservative: only words whose referent is visible in a product
# photograph. Sizes in millimetres, warranty terms and spec numbers are not.

COLOUR = """black white red blue green yellow orange purple pink brown grey gray
silver gold golden beige navy maroon teal cream ivory turquoise violet bronze
copper khaki mustard burgundy lavender peach coral tan charcoal"""

MATERIAL = """cotton leather plastic metal metallic steel aluminium aluminum wood
wooden glass rubber silicone denim silk wool linen mesh fabric velvet suede
ceramic canvas nylon polyester chiffon lace satin jersey fleece marble"""

TEXTURE = """matte glossy shiny glossy smooth rough soft textured transparent
translucent opaque striped floral printed embroidered patterned checked plaid
polka ribbed quilted glitter shimmer brushed polished"""

SHAPE = """round square rectangular slim thin thick compact curved flat oval
cylindrical foldable slender bulky tapered rounded narrow wide chunky"""

PARTS = """strap zipper button pocket collar sleeve neckline hood handle wheel
lid buckle stand grille port lens camera bezel screen display keypad hem cuff
frame knob dial band clasp seam stitch panel edge corner base rim"""

BRIGHT = """bright dark light pale deep vivid vibrant muted neutral"""

VISUAL = set()
for block in (COLOUR, MATERIAL, TEXTURE, SHAPE, PARTS, BRIGHT):
    VISUAL |= {STEM.stem(w) for w in block.split()}

# Commercial / logistics boilerplate, for contrast.
COMMERCIAL = {STEM.stem(w) for w in """warranty guarantee delivery shipping return
refund replacement genuine original authentic brand seller stock order buy
purchase price discount offer free gift packaging box sealed inspected condition
customer service policy days months year quality best premium high""".split()}


def content_tokens(t):
    return [STEM.stem(w) for w in WORD_RE.findall(t.lower())
            if w not in STOP and len(w) > 1]


def density(text, lexicon):
    toks = content_tokens(text)
    if not toks:
        return 0.0
    return 100 * sum(1 for t in toks if t in lexicon) / len(toks)


def load_records():
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
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
            if r.get("item_id") in ids:
                recs[r["item_id"]] = r
    return recs


def load_system(path, key):
    out = {}
    if not Path(path).exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            if d.get(key):
                out[str(d["item_id"])] = d[key]
    return out


def strip_prompt(text, pset):
    return " ".join(w for w in WORD_RE.findall(text.lower()) if STEM.stem(w) not in pset)


def main():
    print("\n" + "=" * 78)
    print("N8 — VISUAL ATTRIBUTE DENSITY (reference-free)")
    print("=" * 78)

    records = load_records()
    print(f"\n  {len(records)} test records | visual lexicon = {len(VISUAL)} stems")

    # ── 1. Are the references visual at all? ─────────────────────────────────
    ref_vad, ref_com, resid_vad, resid_com, prompt_vad = [], [], [], [], []
    for r in records.values():
        ref = (r.get("description") or "").strip()
        if not ref:
            continue
        prompt = build_metadata_prompt(r)
        pset = set(content_tokens(prompt))
        ref_vad.append(density(ref, VISUAL))
        ref_com.append(density(ref, COMMERCIAL))
        resid = strip_prompt(ref, pset)
        resid_vad.append(density(resid, VISUAL))
        resid_com.append(density(resid, COMMERCIAL))
        prompt_vad.append(density(prompt, VISUAL))

    def avg(x):
        return round(sum(x) / len(x), 2)

    corpus = {
        "reference_visual_density": avg(ref_vad),
        "reference_commercial_density": avg(ref_com),
        "reference_residual_visual_density": avg(resid_vad),
        "reference_residual_commercial_density": avg(resid_com),
        "metadata_prompt_visual_density": avg(prompt_vad),
    }
    print("\n  [What the text is made of, % of content tokens]")
    for k, v in corpus.items():
        print(f"    {k:<44} {v}")

    # ── 2. VAD of every system's output ──────────────────────────────────────
    systems = {
        "Reference (seller-written)": {i: r["description"] for i, r in records.items()
                                       if r.get("description")},
        "Metadata-Only (template)": {i: build_metadata_prompt(r) for i, r in records.items()},
        "BLIP fine-tuned": load_system(RESULTS_DIR / "blip_results.jsonl", "generated"),
        "CLIP-GPT2 fine-tuned": load_system(RESULTS_DIR / "clip_gpt2_results.jsonl", "generated"),
        "BLIP Stage-1 (category-only prompt)": load_system(
            RESULTS_DIR / "two_stage_results_blip.jsonl", "description_stage1"),
        "CLIP-GPT2 Stage-1 (category-only)": load_system(
            RESULTS_DIR / "two_stage_results_clip_gpt2.jsonl", "description_stage1"),
        "Two-Stage BLIP (stage2)": load_system(
            RESULTS_DIR / "two_stage_results_blip.jsonl", "description_stage2"),
    }

    print("\n  [Visual Attribute Density by system]")
    sys_vad = {}
    for name, gens in systems.items():
        if not gens:
            continue
        vals = [density(g, VISUAL) for g in gens.values()]
        coms = [density(g, COMMERCIAL) for g in gens.values()]
        sys_vad[name] = {"VAD": avg(vals), "CommercialDensity": avg(coms), "n": len(vals)}
        print(f"    {name:<40} VAD={avg(vals):>6}   Commercial={avg(coms):>6}   n={len(vals)}")

    # ── 3. Does VAD track the judge? ─────────────────────────────────────────
    print("\n  [Validation: VAD vs Gemma-4-31B judge, Stage-1 outputs]")
    xs, vg, fl, rel, ov = [], [], [], [], []
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
            if not (isinstance(sc, dict) and sc and gen):
                continue
            xs.append(density(gen, VISUAL))
            vg.append(float(sc.get("visual_grounding", 0)))
            fl.append(float(sc.get("fluency", 0)))
            rel.append(float(sc.get("relevance", 0)))
            ov.append(float(sc.get("overall", 0)))

    corr = {}
    for label, ys in (("visual_grounding", vg), ("fluency", fl),
                      ("relevance", rel), ("overall", ov)):
        rho, p = spearmanr(xs, ys)
        corr[label] = {"spearman_rho": round(float(rho), 3), "p_value": round(float(p), 5)}
        star = "*" if p < 0.05 else " "
        print(f"    VAD vs judge {label:<20} rho = {rho:+.3f}{star}   (p={p:.4g})")
    print(f"    n = {len(xs)}")

    OUT_JSON.write_text(json.dumps(
        {"corpus": corpus, "systems": sys_vad, "vad_vs_judge": corr, "n": len(xs)},
        indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
