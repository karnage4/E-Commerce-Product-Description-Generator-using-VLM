"""
Experiment N11 — The blind baseline: how far do you get with no image at all?

A text-only GPT-2 is fine-tuned on (metadata -> description) with the product
photograph never shown to it. Everything else matches the CLIP-GPT2 setup: same
split, same metadata prompt, same tokenizer, same loss masking (loss on
description tokens only), same decoding parameters.

The comparison is therefore controlled. The only difference between this model
and the CLIP-GPT2 model is the 10-token visual prefix.

If the blind model matches the vision-language models, then the reported
"vision-language" results on this task are not evidence that vision helped, and
every paper that reports BLEU/ROUGE/CIDEr on seller-authored references without
a blind control has an unsupported claim in it.

This is the e-commerce analogue of the language-prior control that Goyal et al.
(CVPR 2017) introduced for VQA.

Run:
    python -m models.novelty.blind_baseline --train
    python -m models.novelty.blind_baseline --evaluate
    python -m models.novelty.blind_baseline --train --evaluate --epochs 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RESULTS_DIR,
    build_metadata_prompt,
)
from models.shared.metrics import compute_all_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = RESULTS_DIR.parent / "checkpoints" / "blind_gpt2"
RESULTS_FILE = RESULTS_DIR / "novelty_blind_baseline_results.jsonl"
METRICS_FILE = RESULTS_DIR / "novelty_blind_baseline_metrics.json"

# Matched to the CLIP-GPT2 configuration in models/clip_gpt2/train_colab.py
GPT2_MODEL = "gpt2"
MAX_PROMPT_LEN = 64
MAX_TARGET_LEN = 128
LR = 1e-5
BATCH_SIZE = 4
GRAD_ACCUM = 2
WEIGHT_DECAY = 0.01
WARMUP = 100


class BlindTextDataset(Dataset):
    """(metadata prompt -> description) pairs. No image is ever loaded."""

    _SPLITS = {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT}

    def __init__(self, split, tokenizer, max_samples=None):
        self.tok = tokenizer
        ids = set(Path(self._SPLITS[split]).read_text(encoding="utf-8").split())
        self.records = []
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
                    self.records.append(r)
                if max_samples and len(self.records) >= max_samples:
                    break
        print(f"  [blind-{split}] {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        prompt = build_metadata_prompt(rec)
        desc = rec["description"].strip()

        p = self.tok(prompt, truncation=True, max_length=MAX_PROMPT_LEN,
                     padding="max_length", return_tensors="pt")
        d = self.tok(desc, truncation=True, max_length=MAX_TARGET_LEN,
                     padding="max_length", return_tensors="pt")

        labels = d["input_ids"].squeeze(0).clone()
        labels[d["attention_mask"].squeeze(0) == 0] = -100

        return {
            "prompt_ids": p["input_ids"].squeeze(0),
            "prompt_mask": p["attention_mask"].squeeze(0),
            "labels": labels,
            "label_mask": d["attention_mask"].squeeze(0),
        }


def forward_loss(model, batch):
    """Loss on description tokens only — prompt positions masked to -100."""
    pid = batch["prompt_ids"].to(DEVICE)
    pmask = batch["prompt_mask"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    lmask = batch["label_mask"].to(DEVICE)

    ids = torch.cat([pid, labels.clamp(min=0)], dim=1)
    attn = torch.cat([pmask, lmask], dim=1)
    ignore = torch.full_like(pid, -100)
    full_labels = torch.cat([ignore, labels], dim=1)

    return model(input_ids=ids, attention_mask=attn, labels=full_labels).loss


def train(epochs, max_samples=None):
    print("\n" + "=" * 78)
    print("N11 — TRAINING THE BLIND (TEXT-ONLY) BASELINE")
    print("=" * 78)
    print(f"  device={DEVICE}   no images are loaded at any point")

    tok = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(GPT2_MODEL).to(DEVICE)

    tr = BlindTextDataset("train", tok, max_samples)
    va = BlindTextDataset("val", tok)
    tl = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
    vl = DataLoader(va, batch_size=BATCH_SIZE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps = (len(tl) // GRAD_ACCUM) * epochs
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, max(steps, 1))

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    history, best = [], float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for i, b in enumerate(tqdm(tl, desc=f"  epoch {ep}/{epochs} [train]", leave=False)):
            loss = forward_loss(model, b) / GRAD_ACCUM
            loss.backward()
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item() * GRAD_ACCUM

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for b in tqdm(vl, desc=f"  epoch {ep}/{epochs} [val]", leave=False):
                vtot += forward_loss(model, b).item()

        tr_l, va_l = tot / len(tl), vtot / len(vl)
        history.append({"epoch": ep, "train_loss": tr_l, "val_loss": va_l})
        print(f"  epoch {ep}: train={tr_l:.4f}  val={va_l:.4f}")

        if va_l < best:
            best = va_l
            model.save_pretrained(CKPT_DIR / "best_model")
            tok.save_pretrained(CKPT_DIR / "best_model")
            print(f"    saved best (val {va_l:.4f})")

    (RESULTS_DIR / "novelty_blind_baseline_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n  Best val loss {best:.4f} — checkpoint at {CKPT_DIR/'best_model'}")


@torch.no_grad()
def evaluate(max_new_tokens=150, num_beams=4):
    print("\n" + "=" * 78)
    print("N11 — EVALUATING THE BLIND BASELINE")
    print("=" * 78)

    ckpt = CKPT_DIR / "best_model"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt} — run with --train first.")
    tok = GPT2Tokenizer.from_pretrained(str(ckpt))
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(str(ckpt)).to(DEVICE).eval()

    ds = BlindTextDataset("test", tok)
    hyps, refs, rows = [], [], []
    t0 = time.time()

    for rec in tqdm(ds.records, desc="  blind inference", unit="item"):
        prompt = build_metadata_prompt(rec)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LEN)
        out = model.generate(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            no_repeat_ngram_size=3, early_stopping=True,
            pad_token_id=tok.eos_token_id,
        )
        # Score only the continuation, never the prompt (see N9).
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ref = rec["description"].strip()
        hyps.append(gen if gen else ".")
        refs.append(ref)
        rows.append({"item_id": rec["item_id"], "category": rec.get("category", ""),
                     "generated": gen, "reference": ref})

    elapsed = time.time() - t0
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    m = compute_all_metrics(hyps, refs)
    from models.novelty.visual_lexicon import density, VISUAL
    m["Visual Attribute Density"] = round(sum(density(h, VISUAL) for h in hyps) / len(hyps), 2)
    m["sec_per_description"] = round(elapsed / len(hyps), 3)
    m["n"] = len(hyps)

    print("\n  BLIND (no image ever seen):")
    for k, v in m.items():
        print(f"    {k:<26} {v}")

    METRICS_FILE.write_text(json.dumps(m, indent=2), encoding="utf-8")
    print(f"\nSaved → {METRICS_FILE}")
    print(f"Saved → {RESULTS_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=None)
    a = ap.parse_args()
    if a.train:
        train(a.epochs, a.max_samples)
    if a.evaluate:
        evaluate()
    if not (a.train or a.evaluate):
        ap.print_help()
