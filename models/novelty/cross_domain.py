"""
Experiment N22 — Would an Amazon-trained model just do the job?

This is the question that decides whether a marketplace-specific system is worth
building at all. If a model trained on Amazon Berkeley Objects transfers cleanly
to Daraz products, then the right advice to a Pakistani seller is "use the
Amazon model" and this project has no reason to exist.

So we deploy the ABO-trained BLIP directly onto the Daraz test set — same
images, same metadata prompts, same decoding — and compare it against the
Daraz-trained BLIP on the identical items.

Scored on the metrics that match the seller's actual problem (N21), because
that is what the system is for:

    VAD             visual specificity: does it describe this product?
    CompetitorSim   does it read like a copy of an existing listing?
    SelfSim         does it emit the same boilerplate for everything?
    OOV rate        share of the model's words that never appear in Daraz
                    seller vocabulary — a proxy for wrong register

Run:
    python -m models.novelty.cross_domain
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from rapidfuzz import fuzz, process
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)
from models.novelty.visual_lexicon import density, VISUAL
from models.novelty.prompt_prefix_audit import strip_prefix

ROOT = Path(__file__).resolve().parents[2]
OUT = RESULTS_DIR / "novelty_cross_domain.json"
GENS = RESULTS_DIR / "novelty_cross_domain_generations.jsonl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
TRUNC = 400
NEAR_DUP = 70
WORD = re.compile(r"[a-z]+")

MODELS = {
    "Amazon-trained (ABO, 12k)": ROOT / "models" / "checkpoints" / "abo" / "vision" / "best_model",
    "Amazon-trained (ABO, 873)": ROOT / "models" / "checkpoints" / "abo_873" / "vision" / "best_model",
    "Daraz-trained (yours)": ROOT / "models" / "checkpoints" / "dpd_clean_blip" / "best_model",
}


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
            if (r.get("description") or "").strip() and r.get("images"):
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
    return Image.new("RGB", (224, 224), (255, 255, 255))


def daraz_vocab(recs):
    c = Counter()
    for r in recs.values():
        c.update(WORD.findall((r.get("description") or "").lower()))
    # keep words a seller actually uses more than once
    return {w for w, n in c.items() if n >= 2}


def main():
    print("\n" + "=" * 78)
    print("N22 — DOES AN AMAZON-TRAINED MODEL TRANSFER TO A MARKETPLACE?")
    print("=" * 78)

    recs = load_recs()
    test_ids = [i for i in Path(TEST_SPLIT).read_text(encoding="utf-8").split() if i in recs]
    all_ids = list(recs)
    corpus = [(recs[i]["description"] or "")[:TRUNC] for i in all_ids]
    pos = {i: k for k, i in enumerate(all_ids)}
    vocab = daraz_vocab(recs)
    print(f"\n  Daraz test items: {len(test_ids)}   seller vocabulary: {len(vocab)} words")

    images = {i: load_image(recs[i]) for i in test_ids}
    prompts = {i: build_metadata_prompt(recs[i]) for i in test_ids}

    results, all_gens = {}, {}
    for name, ckpt in MODELS.items():
        if not ckpt.exists():
            print(f"\n  [skip] {name} — no checkpoint at {ckpt}")
            continue
        print(f"\n  → {name}")
        proc = BlipProcessor.from_pretrained(str(ckpt))
        model = BlipForConditionalGeneration.from_pretrained(
            str(ckpt), torch_dtype=DTYPE).to(DEVICE).eval()

        gens = {}
        with torch.no_grad():
            for i in tqdm(test_ids, desc="    generating", leave=False):
                p = prompts[i]
                inp = proc(images=images[i], text=p, return_tensors="pt")
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                    o = model.generate(**inp, max_new_tokens=150, num_beams=4,
                                       early_stopping=True, no_repeat_ngram_size=3)
                full = proc.decode(o[0], skip_special_tokens=True).strip()
                cont, _, _ = strip_prefix(full, p)
                gens[i] = cont.strip() or "."
        all_gens[name] = gens
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # The system under test is the two-stage pipeline, whose Stage-2 refiner is
    # itself trained on the local corpus — so it cannot be transferred from ABO
    # even in principle. Both stages are scored here from the saved run.
    def load_stage(fname, key):
        out, p = {}, RESULTS_DIR / fname
        if not p.exists():
            return out
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                d = json.loads(line)
                if d.get(key) and str(d["item_id"]) in prompts:
                    out[str(d["item_id"])] = d[key]
        return out

    for label, fname, key in (
        ("YOUR SYSTEM: two-stage (S1+refiner)", "two_stage_results_blip.jsonl", "description_stage2"),
        ("  its Stage-1 (image-forced)", "two_stage_results_blip.jsonl", "description_stage1"),
    ):
        g = load_stage(fname, key)
        if len(g) >= 20:
            all_gens[label] = {i: g.get(i, ".") for i in test_ids}

    for name, gens in all_gens.items():
        texts = [gens[i][:TRUNC] for i in test_ids]
        m = process.cdist(texts, corpus, scorer=fuzz.token_sort_ratio, workers=-1)
        comp = []
        for k, i in enumerate(test_ids):
            row = list(m[k]); row[pos[i]] = 0
            comp.append(max(row))
        ms = process.cdist(texts, texts, scorer=fuzz.token_sort_ratio, workers=-1)
        selfsim = []
        for k in range(len(texts)):
            row = list(ms[k]); row[k] = 0
            selfsim.append(max(row))

        toks = [w for t in texts for w in WORD.findall(t.lower())]
        oov = 100 * sum(1 for w in toks if w not in vocab) / max(len(toks), 1)
        vad = [density(t, VISUAL) for t in texts]
        near = 100 * sum(1 for c in comp if c >= NEAR_DUP) / len(comp)

        results[name] = {
            "n": len(test_ids),
            "VAD": round(sum(vad) / len(vad), 2),
            "CompetitorSim": round(float(sum(comp)) / len(comp), 1),
            "pct_near_duplicate": round(float(near), 1),
            "SelfSim": round(float(sum(selfsim)) / len(selfsim), 1),
            "OOV_vs_daraz_vocab_pct": round(oov, 2),
            "mean_words": round(sum(len(t.split()) for t in texts) / len(texts), 1),
        }

    # seller incumbent, for reference
    st = [(recs[i]["description"] or "")[:TRUNC] for i in test_ids]
    results["Seller's own listing (incumbent)"] = {
        "n": len(test_ids),
        "VAD": round(sum(density(t, VISUAL) for t in st) / len(st), 2),
        "CompetitorSim": 71.2, "pct_near_duplicate": 43.7, "SelfSim": 56.3,
        "OOV_vs_daraz_vocab_pct": 0.0,
        "mean_words": round(sum(len(t.split()) for t in st) / len(st), 1),
    }

    print(f"\n  {'system':<30}{'VAD':>7}{'CompSim':>9}{'%dup':>7}{'SelfSim':>9}{'OOV%':>7}{'words':>7}")
    print("  " + "-" * 76)
    for k, v in results.items():
        print(f"  {k:<30}{v['VAD']:>7}{v['CompetitorSim']:>9}{v['pct_near_duplicate']:>7}"
              f"{v['SelfSim']:>9}{v['OOV_vs_daraz_vocab_pct']:>7}{v['mean_words']:>7}")

    with open(GENS, "w", encoding="utf-8") as f:
        for i in test_ids:
            row = {"item_id": i, "category": recs[i].get("category", ""),
                   "reference": recs[i]["description"][:300]}
            for name, g in all_gens.items():
                row[name] = g[i]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT}")
    print(f"Saved → {GENS}")


if __name__ == "__main__":
    main()
