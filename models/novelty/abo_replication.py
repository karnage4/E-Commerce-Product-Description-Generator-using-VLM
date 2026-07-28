"""
Experiment N10 — Does the confound replicate on Amazon Berkeley Objects?

This is the experiment that decides whether the Daraz findings are a property of
one small scraped corpus or a property of the task.

ABO is the dataset used by the Stanford CS231N paper this project replicates, and
by a large slice of the e-commerce description-generation literature. It is
Amazon's own catalogue: 147,702 listings, professionally maintained, two orders
of magnitude larger than DPD. If seller-authored references on ABO are also
largely recoverable from ABO metadata, then "the reference cannot measure visual
grounding" is a statement about the benchmark, not about Daraz.

Measures, on ABO and DPD side by side:
    - share of the reference's content tokens that also occur in the metadata
    - ROUGE-L of the metadata prompt submitted verbatim as the description
      (the blind template baseline: no model, no image)
    - Visual Attribute Density of the references

Data:
    python -c "import urllib.request as u; [u.urlretrieve(
        f'https://amazon-berkeley-objects.s3.amazonaws.com/listings/metadata/listings_{i}.json.gz',
        f'data/abo/listings_{i}.json.gz') for i in '0123456789abcdef']"

Run:
    python -m models.novelty.abo_replication
    python -m models.novelty.abo_replication --max-items 20000
"""

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import nltk
from nltk.stem.porter import PorterStemmer
from rouge_score import rouge_scorer as rouge_lib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)
from models.novelty.visual_lexicon import density, VISUAL, COMMERCIAL

ABO_DIR = Path(__file__).resolve().parents[2] / "data" / "abo"
OUT_JSON = RESULTS_DIR / "novelty_abo_replication.json"

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()
WORD_RE = re.compile(r"[a-z0-9]+")


def content_tokens(t):
    return [STEM.stem(w) for w in WORD_RE.findall(t.lower())
            if w not in STOP and len(w) > 1]


# ── ABO field handling ────────────────────────────────────────────────────────

def en_values(field, want_list=False):
    """Pull English values out of an ABO [{language_tag, value}, ...] field."""
    if not field:
        return [] if want_list else ""
    vals = []
    for entry in field:
        if not isinstance(entry, dict):
            continue
        tag = entry.get("language_tag", "")
        # product_type and similar have no language tag at all
        if tag and not tag.startswith("en"):
            continue
        v = entry.get("value")
        if v:
            vals.append(str(v))
    return vals if want_list else (vals[0] if vals else "")


def abo_metadata_prompt(rec):
    """The metadata a model would be conditioned on, mirroring DPD's prompt.

    Same fields the Stanford setup describes: parsed JSON metadata with
    non-essential keys removed.
    """
    parts = []
    name = en_values(rec.get("item_name"))
    if name:
        parts.append(f"Product: {name}")
    brand = en_values(rec.get("brand"))
    if brand:
        parts.append(f"Brand: {brand}")
    ptype = en_values(rec.get("product_type"))
    if ptype:
        parts.append(f"Category: {ptype}")
    for key, label in (("color", "Color"), ("style", "Style"),
                       ("material", "Material"), ("pattern", "Pattern"),
                       ("fabric_type", "Fabric"), ("model_name", "Model"),
                       ("model_number", "Model number"), ("model_year", "Year")):
        v = en_values(rec.get(key))
        if v:
            parts.append(f"{label}: {v}")
    kws = en_values(rec.get("item_keywords"), want_list=True)
    if kws:
        parts.append("Keywords: " + ", ".join(kws[:8]))
    return ". ".join(parts)


def abo_reference(rec):
    """The generation target: the bullet-point description block."""
    bullets = en_values(rec.get("bullet_point"), want_list=True)
    return " ".join(bullets).strip()


def load_abo(max_items):
    recs = []
    files = sorted(ABO_DIR.glob("listings_*.json.gz"))
    if not files:
        raise SystemExit(f"No ABO shards in {ABO_DIR} — see the docstring for the download command.")
    for fp in tqdm(files, desc="  reading ABO shards"):
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Mirror the Stanford filter: English, has image, has description
                if not d.get("main_image_id"):
                    continue
                ref = abo_reference(d)
                if len(ref.split()) < 10:
                    continue
                prompt = abo_metadata_prompt(d)
                if not prompt:
                    continue
                recs.append((prompt, ref))
                if max_items and len(recs) >= max_items:
                    return recs
    return recs


def load_dpd():
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
    out = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids and (r.get("description") or "").strip():
                out.append((build_metadata_prompt(r), r["description"].strip()))
    return out


# ── Measurement ───────────────────────────────────────────────────────────────

def measure(pairs, label):
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    frac, rl, vad, com, reflen = [], [], [], [], []
    for prompt, ref in tqdm(pairs, desc=f"  measuring {label}"):
        pset = set(content_tokens(prompt))
        rt = content_tokens(ref)
        if not rt:
            continue
        frac.append(sum(1 for t in rt if t in pset) / len(rt))
        rl.append(scorer.score(ref, prompt)["rougeL"].fmeasure)
        vad.append(density(ref, VISUAL))
        com.append(density(ref, COMMERCIAL))
        reflen.append(len(rt))

    def avg(x):
        return round(sum(x) / len(x), 2)

    med = sorted(frac)[len(frac) // 2]
    return {
        "n": len(frac),
        "pct_of_reference_recoverable_from_metadata": round(100 * sum(frac) / len(frac), 2),
        "median_pct": round(100 * med, 2),
        "blind_metadata_template_ROUGE_L": round(100 * sum(rl) / len(rl), 2),
        "reference_visual_density": avg(vad),
        "reference_commercial_density": avg(com),
        "mean_reference_content_tokens": avg(reflen),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-items", type=int, default=40000)
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("N10 — DOES THE CONFOUND REPLICATE ON AMAZON BERKELEY OBJECTS?")
    print("=" * 78)

    abo = load_abo(args.max_items)
    dpd = load_dpd()
    print(f"\n  ABO English listings with image + description: {len(abo)}")
    print(f"  DPD test listings: {len(dpd)}")

    results = {"ABO (Amazon)": measure(abo, "ABO"),
               "DPD (Daraz)": measure(dpd, "DPD")}

    print("\n" + "-" * 78)
    keys = [
        ("pct_of_reference_recoverable_from_metadata", "% of reference recoverable from metadata"),
        ("median_pct", "   (median)"),
        ("blind_metadata_template_ROUGE_L", "Blind metadata template, ROUGE-L"),
        ("reference_visual_density", "Reference visual density"),
        ("reference_commercial_density", "Reference commercial density"),
        ("mean_reference_content_tokens", "Mean reference length (content tokens)"),
        ("n", "n"),
    ]
    print(f"  {'':<44}{'ABO':>12}{'DPD':>12}")
    print("  " + "-" * 68)
    for k, label in keys:
        print(f"  {label:<44}{results['ABO (Amazon)'][k]:>12}{results['DPD (Daraz)'][k]:>12}")
    print("-" * 78)

    a = results["ABO (Amazon)"]["pct_of_reference_recoverable_from_metadata"]
    d = results["DPD (Daraz)"]["pct_of_reference_recoverable_from_metadata"]
    print(f"\n  VERDICT: ABO {a}% vs DPD {d}% of the reference recoverable from metadata alone.")
    if a >= 0.7 * d:
        print("  → The confound REPLICATES on Amazon's catalogue. It is a property of the")
        print("    task, not of the Daraz scrape. This generalises.")
    else:
        print("  → The confound is substantially WEAKER on ABO. The Daraz finding may be")
        print("    driven by corpus size or seller-writing habits; scope claims accordingly.")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
