"""
Experiment N6 — Echo Dynamics Across Training.

Additive experiment: the five per-epoch BLIP checkpoints already on disk are
loaded read-only. Nothing is retrained and nothing is overwritten.

Motivation
----------
Milestone 2 reported that BLIP's validation loss is minimised at epoch 1 and
rises monotonically afterwards, and attributed this to overfitting caused by
data scarcity. That explanation assumes the model is learning the right thing
and simply running out of examples.

N1 suggests a different reading. If the cheapest way to reduce loss on this
corpus is to copy the metadata prompt into the output, then continued training
should make the model echo *more* and ground *less* — regardless of dataset
size. Adding data would not fix that; it would accelerate it.

The two explanations make opposite predictions, and the epoch checkpoints let us
test them for free:

    data scarcity      Echo Rate flat,     Visual Residual Recall flat or up
    objective misfit   Echo Rate rises,    Visual Residual Recall falls

Run:
    python -m models.novelty.epoch_dynamics
    python -m models.novelty.epoch_dynamics --max-samples 60
"""

import argparse
import json
import re
import sys
from pathlib import Path

import nltk
import torch
from PIL import Image
from nltk.stem.porter import PorterStemmer
from rouge_score import rouge_scorer as rouge_lib
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)
from models.shared.metrics import compute_all_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
OUT_JSON = RESULTS_DIR / "novelty_epoch_dynamics.json"
BLANK = Image.new("RGB", (224, 224), (128, 128, 128))

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


def strip_prompt(text, pset):
    return " ".join(w for w in WORD_RE.findall(text.lower()) if STEM.stem(w) not in pset)


def load_records(n):
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
            if r.get("item_id") in ids and r.get("images") and r.get("description", "").strip():
                recs.append(r)
    return recs[:n]


def load_image(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return Image.new("RGB", (224, 224), (255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=135)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--vss-samples", type=int, default=40)
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("N6 — ECHO DYNAMICS ACROSS TRAINING EPOCHS")
    print("=" * 78)

    records = load_records(args.max_samples)
    images = [load_image(r) for r in records]
    prompts = [build_metadata_prompt(r) for r in records]
    refs = [r["description"].strip() for r in records]
    print(f"  {len(records)} test items")

    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    ckpt_root = RESULTS_DIR.parent / "checkpoints" / "blip"
    epochs = sorted([p for p in ckpt_root.glob("epoch_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[1]))
    print(f"  checkpoints: {[p.name for p in epochs]}")

    results = {}
    for ck in epochs:
        print(f"\n  → {ck.name}")
        proc = BlipProcessor.from_pretrained(str(ck))
        model = BlipForConditionalGeneration.from_pretrained(
            str(ck), torch_dtype=DTYPE).to(DEVICE).eval()

        @torch.no_grad()
        def gen(img, prompt):
            inputs = proc(images=img, text=prompt, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     num_beams=4, early_stopping=True,
                                     no_repeat_ngram_size=3)
            return proc.decode(ids[0], skip_special_tokens=True).strip()

        gens, blanks = [], []
        for i in tqdm(range(len(records)), desc=f"    {ck.name}", unit="item"):
            gens.append(gen(images[i], prompts[i]))
            if i < args.vss_samples:
                blanks.append(gen(BLANK, prompts[i]))

        cell = compute_all_metrics(gens, refs)

        vr_num = vr_den = mr_num = mr_den = echo_num = echo_den = 0
        res = []
        for g, ref, p in zip(gens, refs, prompts):
            pset = set(content_tokens(p))
            gset = set(content_tokens(g))
            rt = content_tokens(ref)
            vset = {t for t in rt if t not in pset}
            mset = {t for t in rt if t in pset}
            vr_num += len(gset & vset); vr_den += len(vset)
            mr_num += len(gset & mset); mr_den += len(mset)
            ga = content_tokens(g)
            echo_num += sum(1 for t in ga if t in pset); echo_den += len(ga)
            rr = strip_prompt(ref, pset)
            if rr.strip():
                res.append(scorer.score(rr, strip_prompt(g, pset))["rougeL"].fmeasure)

        cell["Residual ROUGE-L"] = round(100 * sum(res) / len(res), 2)
        cell["Visual Residual Recall"] = round(100 * vr_num / max(vr_den, 1), 2)
        cell["Metadata Recall"] = round(100 * mr_num / max(mr_den, 1), 2)
        cell["Echo Rate"] = round(100 * echo_num / max(echo_den, 1), 2)
        sims = [scorer.score(a, b)["rougeL"].fmeasure for a, b in zip(gens[:len(blanks)], blanks)]
        cell["VSS"] = round(100 - 100 * sum(sims) / len(sims), 2) if sims else None
        cell["mean_gen_words"] = round(sum(len(g.split()) for g in gens) / len(gens), 1)

        results[ck.name] = cell
        for k, v in cell.items():
            print(f"      {k:<26} {v}")
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "-" * 78)
    print(f"  {'epoch':<10}{'ROUGE-L':>10}{'Resid-RL':>10}{'VisResRec':>11}{'MetaRec':>9}{'EchoRate':>10}{'VSS':>8}")
    print("-" * 78)
    for e, c in results.items():
        print(f"  {e:<10}{c['ROUGE-L']:>10}{c['Residual ROUGE-L']:>10}"
              f"{c['Visual Residual Recall']:>11}{c['Metadata Recall']:>9}"
              f"{c['Echo Rate']:>10}{c['VSS'] if c['VSS'] is not None else '-':>8}")
    print("-" * 78)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
