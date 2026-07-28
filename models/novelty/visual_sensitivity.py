"""
Experiment N3 — Visual Counterfactual Sensitivity.

Additive experiment: loads the existing checkpoints read-only and writes new
result files. Training code and original evaluators are untouched.

Motivation
----------
BLEU/ROUGE against a seller-authored reference cannot tell you whether a model
looked at the product photograph. A model that ignores the image entirely and
paraphrases its own metadata prompt can still score well (see N1).

This script measures image dependence directly, by counterfactual. For each test
product we generate a description three times:

    real      the product's own photograph
    blank     a uniform mid-grey image (no visual information at all)
    mismatch  the photograph of a *different* product from a *different* category

and cross both against two prompt conditions:

    full      the full metadata prompt (name, brand, price, attributes...)
    stage1    category + subcategory only (the Stage-1 prompt)

Reported per (model, prompt, image) cell:

    ROUGE-L         quality against the reference
    SelfSim         ROUGE-L(generation_real, generation_counterfactual)
    IdenticalRate   % of items where the counterfactual output is byte-identical

and the headline diagnostic:

    VSS (Visual Sensitivity Score) = 100 - SelfSim(real, blank)

VSS = 0 means removing the image changed nothing: the model is functionally
blind. VSS = 100 means the output is entirely image-driven.

Run:
    python -m models.novelty.visual_sensitivity --max-samples 135
    python -m models.novelty.visual_sensitivity --max-samples 20 --models blip
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from rouge_score import rouge_scorer as rouge_lib
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR,
    build_metadata_prompt, build_stage1_prompt,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
OUT_JSON = RESULTS_DIR / "novelty_visual_sensitivity.json"
OUT_JSONL = RESULTS_DIR / "novelty_visual_sensitivity_generations.jsonl"

BLANK = Image.new("RGB", (224, 224), (128, 128, 128))
SEED = 42


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_records(max_samples):
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
    recs = recs[:max_samples]
    print(f"  Loaded {len(recs)} test records")
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


def build_mismatch_map(records):
    """Pair each record with an image from a different category (deterministic)."""
    rng = random.Random(SEED)
    by_cat = {}
    for i, r in enumerate(records):
        by_cat.setdefault(r.get("category", ""), []).append(i)
    mapping = {}
    for i, r in enumerate(records):
        others = [j for c, idxs in by_cat.items() if c != r.get("category", "") for j in idxs]
        mapping[i] = rng.choice(others) if others else (i + 1) % len(records)
    return mapping


# ── Model wrappers ────────────────────────────────────────────────────────────

class BlipRunner:
    name = "BLIP"

    def __init__(self, max_new_tokens=120):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        ckpt = RESULTS_DIR.parent / "checkpoints" / "blip" / "best_model"
        print(f"  Loading BLIP from {ckpt}")
        self.proc = BlipProcessor.from_pretrained(str(ckpt))
        self.model = BlipForConditionalGeneration.from_pretrained(
            str(ckpt), torch_dtype=DTYPE).to(DEVICE).eval()
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, image, prompt):
        inputs = self.proc(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
            ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                num_beams=4, early_stopping=True, no_repeat_ngram_size=3)
        return self.proc.decode(ids[0], skip_special_tokens=True).strip()


class ClipGpt2Runner:
    name = "CLIP-GPT2"

    def __init__(self, max_new_tokens=120):
        from models.clip_gpt2.evaluate import (
            ClipGPT2Model, CLIP_TRANSFORM, MAX_TEXT_LEN, GPT2_MODEL,
        )
        from transformers import GPT2Tokenizer
        ckpt = RESULTS_DIR.parent / "checkpoints" / "clip_gpt2" / "best_model"
        print(f"  Loading CLIP-GPT2 from {ckpt}")
        self.tok = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
        self.tok.pad_token = self.tok.eos_token
        self.model = ClipGPT2Model()
        state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(DEVICE).eval()
        self.tf = CLIP_TRANSFORM
        self.max_text_len = MAX_TEXT_LEN
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def generate(self, image, prompt):
        px = self.tf(image).unsqueeze(0).to(DEVICE)
        enc = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=self.max_text_len, padding="max_length")
        ids = self.model.generate(
            pixel_values=px,
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
            max_new_tokens=self.max_new_tokens,
            num_beams=4, no_repeat_ngram_size=3,
        )
        return self.tok.decode(ids[0], skip_special_tokens=True).strip()


# ── Experiment ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=135)
    ap.add_argument("--models", nargs="+", default=["blip", "clip_gpt2"])
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("N3 — VISUAL COUNTERFACTUAL SENSITIVITY")
    print("=" * 78)
    print(f"  device={DEVICE}")

    records = load_test_records(args.max_samples)
    mismatch = build_mismatch_map(records)
    images = [load_image(r) for r in records]

    prompt_builders = {"full": build_metadata_prompt, "stage1": build_stage1_prompt}
    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)

    runners = []
    if "blip" in args.models:
        runners.append(BlipRunner(args.max_new_tokens))
    if "clip_gpt2" in args.models:
        runners.append(ClipGpt2Runner(args.max_new_tokens))

    all_results = {}
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    gen_log = open(OUT_JSONL, "w", encoding="utf-8")

    for runner in runners:
        for pname, pbuild in prompt_builders.items():
            gens = {"real": [], "blank": [], "mismatch": []}
            t0 = time.time()
            desc = f"{runner.name}/{pname}"
            for i, rec in enumerate(tqdm(records, desc=f"  {desc}", unit="item")):
                prompt = pbuild(rec)
                conds = {
                    "real": images[i],
                    "blank": BLANK,
                    "mismatch": images[mismatch[i]],
                }
                row = {"model": runner.name, "prompt": pname,
                       "item_id": rec["item_id"], "category": rec.get("category", "")}
                for cname, img in conds.items():
                    out = runner.generate(img, prompt)
                    gens[cname].append(out)
                    row[cname] = out
                gen_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                gen_log.flush()
            elapsed = time.time() - t0

            refs = [r["description"].strip() for r in records]
            cell = {"seconds": round(elapsed, 1),
                    "sec_per_item_3conds": round(elapsed / max(len(records), 1), 2)}

            for cname in ("real", "blank", "mismatch"):
                rl = [scorer.score(ref, g)["rougeL"].fmeasure
                      for ref, g in zip(refs, gens[cname])]
                cell[f"ROUGE-L_{cname}"] = round(100 * sum(rl) / len(rl), 2)

            for cname in ("blank", "mismatch"):
                sims = [scorer.score(a, b)["rougeL"].fmeasure
                        for a, b in zip(gens["real"], gens[cname])]
                ident = sum(1 for a, b in zip(gens["real"], gens[cname]) if a.strip() == b.strip())
                cell[f"SelfSim_real_vs_{cname}"] = round(100 * sum(sims) / len(sims), 2)
                cell[f"IdenticalRate_{cname}"] = round(100 * ident / len(sims), 2)

            cell["VSS_blank"] = round(100 - cell["SelfSim_real_vs_blank"], 2)
            cell["VSS_mismatch"] = round(100 - cell["SelfSim_real_vs_mismatch"], 2)
            cell["DeltaROUGE_real_minus_blank"] = round(
                cell["ROUGE-L_real"] - cell["ROUGE-L_blank"], 2)

            key = f"{runner.name}|{pname}"
            all_results[key] = cell
            print(f"\n  [{key}]")
            for k, v in cell.items():
                print(f"    {k:<32} {v}")

            OUT_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

        del runner
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    gen_log.close()
    print(f"\nSaved → {OUT_JSON}")
    print(f"Saved → {OUT_JSONL}")


if __name__ == "__main__":
    main()
