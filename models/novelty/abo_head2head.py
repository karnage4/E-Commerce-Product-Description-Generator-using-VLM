"""
Experiment N12 — Blind vs. vision, controlled, on Amazon Berkeley Objects.

Day-2 experiment. Repeats the DPD head-to-head on Amazon's catalogue so every
claim in the paper rests on two corpora rather than one small scrape.

Two models are trained on an identical ABO split, with identical prompts,
identical targets, identical loss masking (loss on description tokens only),
identical optimiser settings and identical decoding:

    BLIND    GPT-2, metadata -> description. The image is never loaded.
    VISION   BLIP, image + metadata -> description.

The only difference between the two conditions is whether the model can see the
product photograph. Both are then scored on the same held-out items, with the
prompt prefix stripped from the output (see N9) so neither is credited for
restating its own input.

Also runs the blank-image counterfactual on the trained BLIP, extending N3 to ABO.

Run:
    python -m models.novelty.abo_head2head --train-blind --train-vision --evaluate
    python -m models.novelty.abo_head2head --evaluate --eval-n 600
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, BlipProcessor,
    BlipForConditionalGeneration, get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import RESULTS_DIR
from models.shared.metrics import compute_all_metrics
from models.novelty.visual_lexicon import density, VISUAL
from models.novelty.prompt_prefix_audit import strip_prefix

ROOT = Path(__file__).resolve().parents[2]
ABO = ROOT / "data" / "abo_subset"
META = ABO / "metadata" / "listings_final.jsonl"
SPLITS = ABO / "splits"
CKPT = ROOT / "models" / "checkpoints" / "abo"
OUT = RESULTS_DIR / "novelty_abo_head2head.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
BLANK = Image.new("RGB", (224, 224), (128, 128, 128))

MAX_PROMPT = 64
MAX_TARGET = 128
# Prompt (64) + description (128). Both conditions get the same prompt budget and
# the same description budget, so neither is advantaged by truncation.
BLIP_MAX = MAX_PROMPT + MAX_TARGET
LR = 1e-5
BATCH = 8
GRAD_ACCUM = 1
WARMUP = 100
SEED = 42


TRAIN_N = None   # set by --train-n to match DPD's training size exactly


def load_split(name):
    ids = set((SPLITS / f"{name}.txt").read_text(encoding="utf-8").split())
    out = []
    with open(META, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("item_id") in ids:
                out.append(r)
    if name == "train" and TRAIN_N:
        random.Random(SEED).shuffle(out)
        out = out[:TRAIN_N]
    return out


def trunc_prompt(tokenizer, text, n=MAX_PROMPT):
    """Clip the metadata prompt to n tokens so both conditions see the same budget."""
    ids = tokenizer(text, add_special_tokens=False, truncation=True,
                    max_length=n)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


def image_path(rec):
    return ABO / "images" / rec["item_id"] / "0.jpg"


def load_image(rec):
    p = image_path(rec)
    if p.exists():
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


# ── BLIND: text-only GPT-2 ────────────────────────────────────────────────────

class BlindDS(Dataset):
    def __init__(self, recs, tok):
        self.recs, self.tok = recs, tok

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        p = self.tok(r["abo_metadata_prompt"], truncation=True, max_length=MAX_PROMPT,
                     padding="max_length", return_tensors="pt")
        d = self.tok(r["description"], truncation=True, max_length=MAX_TARGET,
                     padding="max_length", return_tensors="pt")
        labels = d["input_ids"].squeeze(0).clone()
        labels[d["attention_mask"].squeeze(0) == 0] = -100
        return {"pid": p["input_ids"].squeeze(0), "pmask": p["attention_mask"].squeeze(0),
                "labels": labels, "lmask": d["attention_mask"].squeeze(0)}


def blind_loss(model, b):
    pid, pmask = b["pid"].to(DEVICE), b["pmask"].to(DEVICE)
    labels, lmask = b["labels"].to(DEVICE), b["lmask"].to(DEVICE)
    ids = torch.cat([pid, labels.clamp(min=0)], 1)
    attn = torch.cat([pmask, lmask], 1)
    full = torch.cat([torch.full_like(pid, -100), labels], 1)
    return model(input_ids=ids, attention_mask=attn, labels=full).loss


def train_blind(epochs):
    print("\n=== TRAINING BLIND (GPT-2, no image) on ABO ===")
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)

    tr, va = load_split("train"), load_split("val")
    print(f"  train={len(tr)} val={len(va)}")
    tl = DataLoader(BlindDS(tr, tok), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(BlindDS(va, tok), batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, WARMUP, len(tl) * epochs)
    dest = CKPT / "blind" / "best_model"
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for b in tqdm(tl, desc=f"  blind ep{ep}", leave=False):
            loss = blind_loss(model, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            tot += loss.item()
        model.eval(); v = 0.0
        with torch.no_grad():
            for b in tqdm(vl, desc=f"  blind ep{ep} val", leave=False):
                v += blind_loss(model, b).item()
        tr_l, va_l = tot / len(tl), v / len(vl)
        print(f"  blind epoch {ep}: train={tr_l:.4f} val={va_l:.4f}")
        if va_l < best:
            best = va_l
            dest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(dest); tok.save_pretrained(dest)
            print(f"    saved (val {va_l:.4f})")
    del model
    torch.cuda.empty_cache()


# ── VISION: BLIP ──────────────────────────────────────────────────────────────

class BlipDS(Dataset):
    def __init__(self, recs, proc):
        self.recs, self.proc = recs, proc

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        prompt = trunc_prompt(self.proc.tokenizer, r["abo_metadata_prompt"])
        desc = r["description"]
        enc = self.proc(images=load_image(r), text=prompt + " " + desc,
                        return_tensors="pt", padding="max_length",
                        truncation=True, max_length=BLIP_MAX)
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        labels[enc["attention_mask"].squeeze(0) == 0] = -100
        # Mask the metadata prefix so loss falls on description tokens only.
        n_prompt = len(self.proc.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        labels[:min(n_prompt + 1, BLIP_MAX)] = -100
        return {"pixel_values": enc["pixel_values"].squeeze(0),
                "input_ids": ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels}


def train_vision(epochs):
    print("\n=== TRAINING VISION (BLIP, image + metadata) on ABO ===")
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to(DEVICE)
    model.gradient_checkpointing_enable()

    tr, va = load_split("train"), load_split("val")
    print(f"  train={len(tr)} val={len(va)}")
    tl = DataLoader(BlipDS(tr, proc), batch_size=BATCH, shuffle=True, num_workers=4)
    vl = DataLoader(BlipDS(va, proc), batch_size=BATCH, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sch = get_cosine_schedule_with_warmup(opt, WARMUP, len(tl) * epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    dest = CKPT / "vision" / "best_model"
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for b in tqdm(tl, desc=f"  vision ep{ep}", leave=False):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                loss = model(**b).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            tot += loss.item()
        model.eval(); v = 0.0
        with torch.no_grad():
            for b in tqdm(vl, desc=f"  vision ep{ep} val", leave=False):
                b = {k: x.to(DEVICE) for k, x in b.items()}
                with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                    v += model(**b).loss.item()
        tr_l, va_l = tot / len(tl), v / len(vl)
        print(f"  vision epoch {ep}: train={tr_l:.4f} val={va_l:.4f}")
        if va_l < best:
            best = va_l
            dest.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(dest); proc.save_pretrained(dest)
            print(f"    saved (val {va_l:.4f})")
    del model
    torch.cuda.empty_cache()


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(n_eval, counterfactual_n):
    print("\n=== EVALUATING ON ABO TEST SPLIT ===")
    test = load_split("test")
    random.Random(SEED).shuffle(test)
    test = test[:n_eval]
    refs = [r["description"].strip() for r in test]
    prompts = [r["abo_metadata_prompt"] for r in test]
    results = {}

    # BLIND
    bdir = CKPT / "blind" / "best_model"
    if bdir.exists():
        tok = GPT2Tokenizer.from_pretrained(str(bdir))
        tok.pad_token = tok.eos_token
        m = GPT2LMHeadModel.from_pretrained(str(bdir)).to(DEVICE).eval()
        hyp = []
        for p in tqdm(prompts, desc="  blind infer"):
            e = tok(p, return_tensors="pt", truncation=True, max_length=MAX_PROMPT)
            o = m.generate(input_ids=e["input_ids"].to(DEVICE),
                           attention_mask=e["attention_mask"].to(DEVICE),
                           max_new_tokens=120, num_beams=4, no_repeat_ngram_size=3,
                           early_stopping=True, pad_token_id=tok.eos_token_id)
            g = tok.decode(o[0][e["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            hyp.append(g if g else ".")
        r = compute_all_metrics(hyp, refs)
        r["Visual Attribute Density"] = round(sum(density(h, VISUAL) for h in hyp) / len(hyp), 2)
        results["BLIND GPT-2 (no image)"] = r
        print("  BLIND:", r)

        # Persist per-item blind generations so the blind-vs-vision comparison
        # can be given a confidence interval without re-running inference.
        btag = OUT.stem.replace("novelty_abo_head2head", "")
        bpath = RESULTS_DIR / f"novelty_abo_blind_generations{btag}.jsonl"
        with open(bpath, "w", encoding="utf-8") as bf:
            for rec, ref, g in zip(test, refs, hyp):
                bf.write(json.dumps({"item_id": rec["item_id"], "reference": ref,
                                     "blind": g}, ensure_ascii=False) + "\n")
        print(f"  per-item blind generations → {bpath}")
        del m; torch.cuda.empty_cache()

    # VISION
    vdir = CKPT / "vision" / "best_model"
    if vdir.exists():
        proc = BlipProcessor.from_pretrained(str(vdir))
        m = BlipForConditionalGeneration.from_pretrained(
            str(vdir), torch_dtype=DTYPE).to(DEVICE).eval()
        hyp, hyp_blank = [], []
        vprompts = [trunc_prompt(proc.tokenizer, p) for p in prompts]
        for i, (rec, p) in enumerate(tqdm(list(zip(test, vprompts)), desc="  vision infer")):
            def gen(img):
                inp = proc(images=img, text=p, return_tensors="pt")
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                    o = m.generate(**inp, max_new_tokens=120, num_beams=4,
                                   early_stopping=True, no_repeat_ngram_size=3)
                full = proc.decode(o[0], skip_special_tokens=True).strip()
                cont, _, _ = strip_prefix(full, p)
                return cont if cont.strip() else "."
            hyp.append(gen(load_image(rec)))
            if i < counterfactual_n:
                hyp_blank.append(gen(BLANK))
        r = compute_all_metrics(hyp, refs)
        r["Visual Attribute Density"] = round(sum(density(h, VISUAL) for h in hyp) / len(hyp), 2)
        results["VISION BLIP (image + metadata)"] = r
        print("  VISION:", r)

        # Persist per-item generations so confidence intervals can be computed
        # later without re-running inference (see models/novelty/bootstrap_ci.py).
        tag = OUT.stem.replace("novelty_abo_head2head", "")
        gen_path = RESULTS_DIR / f"novelty_abo_generations{tag}.jsonl"
        with open(gen_path, "w", encoding="utf-8") as gf:
            for i, (rec, ref) in enumerate(zip(test, refs)):
                gf.write(json.dumps({
                    "item_id": rec["item_id"],
                    "reference": ref,
                    "real": hyp[i],
                    "blank": hyp_blank[i] if i < len(hyp_blank) else None,
                }, ensure_ascii=False) + "\n")
        print(f"  per-item generations → {gen_path}")

        if hyp_blank:
            rb = compute_all_metrics(hyp_blank, refs[:len(hyp_blank)])
            rb["Visual Attribute Density"] = round(
                sum(density(h, VISUAL) for h in hyp_blank) / len(hyp_blank), 2)
            from rouge_score import rouge_scorer as rl
            sc = rl.RougeScorer(["rougeL"], use_stemmer=True)
            sim = [sc.score(a, b)["rougeL"].fmeasure
                   for a, b in zip(hyp[:len(hyp_blank)], hyp_blank)]
            rb["VSS"] = round(100 - 100 * sum(sim) / len(sim), 2)
            rb["n"] = len(hyp_blank)
            results["VISION BLIP (BLANK image counterfactual)"] = rb
            print("  BLANK:", rb)
        del m; torch.cuda.empty_cache()

    results["_config"] = {"n_eval": len(test), "counterfactual_n": counterfactual_n,
                          "train_n": len(load_split("train"))}
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "-" * 78)
    print(f"  {'System':<40}{'ROUGE-L':>9}{'BLEU-1':>8}{'CIDEr':>8}{'VAD':>8}")
    print("-" * 78)
    for k, v in results.items():
        if k.startswith("_"):
            continue
        print(f"  {k:<40}{v['ROUGE-L']:>9}{v['BLEU-1']:>8}{v['CIDEr']:>8}"
              f"{v['Visual Attribute Density']:>8}")
    print("-" * 78)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-blind", action="store_true")
    ap.add_argument("--train-vision", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=600)
    ap.add_argument("--counterfactual-n", type=int, default=200)
    ap.add_argument("--train-n", type=int, default=None,
                    help="subsample the ABO training split (873 matches DPD exactly)")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for checkpoint and result paths, to keep runs separate")
    a = ap.parse_args()
    if a.train_n:
        TRAIN_N = a.train_n
        globals()["TRAIN_N"] = a.train_n
    if a.tag:
        CKPT = CKPT.parent / f"abo{a.tag}"
        OUT = RESULTS_DIR / f"novelty_abo_head2head{a.tag}.json"
        globals()["CKPT"] = CKPT
        globals()["OUT"] = OUT
    if a.train_blind:
        train_blind(a.epochs)
    if a.train_vision:
        train_vision(a.epochs)
    if a.evaluate:
        evaluate(a.eval_n, a.counterfactual_n)
