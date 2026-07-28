"""
Experiment N19 — Daraz BLIP on the leakage-free split, using the configuration
that actually works for this corpus.

Why this exists. N14 tried to hold hyperparameters constant across corpora by
training Daraz through the ABO code path. It produced a degenerate model: BLEU-1
of 0.96, output averaging 33 words against 116-word references, and a blank
image out-scoring the real one. Fifteen epochs instead of five did not fix it.

The diagnosis is that "identical hyperparameters" is not the same as "controlled".
ABO's references are short bullet lists (47 content tokens); Daraz's are long
seller prose (114 words). A 64-token prompt cap and a 192-token unified sequence
suit the first and starve the second. Forcing one setting onto both corpora
degrades one of them, which is a confound rather than a control.

So this trains Daraz with the configuration the Milestone-2 pipeline established
for it — full metadata prompt, one 128-token unified sequence, batch 4 with
gradient accumulation 2, image augmentation on train only, cosine schedule,
best-validation checkpointing — and changes only the two things under test:

    the leakage-free split from N17
    the markup-cleaned descriptions from N18

Original checkpoints and result files are untouched.

Run:
    python -m models.novelty.dpd_clean_train --epochs 5
    python -m models.novelty.dpd_clean_train --evaluate-only
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    BlipProcessor, BlipForConditionalGeneration, get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, RESULTS_DIR, build_metadata_prompt,
)
from models.shared.metrics import compute_all_metrics
from models.novelty.visual_lexicon import density, VISUAL
from models.novelty.prompt_prefix_audit import strip_prefix

ROOT = Path(__file__).resolve().parents[2]
# v2 corpus (3,091 listings, 9 categories) lives under the pipeline's own paths
_V2 = ROOT / "data" / "processed"
CLEAN_SPLITS = METADATA_FILE.parent.parent / "splits_clean"
CLEAN_META = METADATA_FILE.parent / "listings_final_textclean.jsonl"
CKPT = ROOT / "models" / "checkpoints" / "dpd_clean_blip"
OUT = RESULTS_DIR / "novelty_dpd_clean_result.json"
GENS = RESULTS_DIR / "novelty_dpd_clean_generations.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
BLANK = Image.new("RGB", (224, 224), (128, 128, 128))

# Mirrors models/blip/train_colab.py
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
GRAD_ACCUM = 2
WARMUP_STEPS = 100
MODEL_NAME = "Salesforce/blip-image-captioning-base"

TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(10),
])


USE_ORIGINAL_SPLIT = False
USE_ORIGINAL_TEXT = False
USE_AUGMENTED = False       # train on LLM-repaired labels instead of seller text
AUG_META = METADATA_FILE.parent / "listings_augmented.jsonl"


def load_split(name):
    split_dir = (METADATA_FILE.parent.parent / "splits") if USE_ORIGINAL_SPLIT else CLEAN_SPLITS
    ids = set((split_dir / f"{name}.txt").read_text(encoding="utf-8").split())
    src = METADATA_FILE if (USE_ORIGINAL_TEXT or not CLEAN_META.exists()) else CLEAN_META
    # Supervision repair: train targets come from the rewritten labels, but the
    # held-out test set is always scored against what the seller actually wrote.
    if USE_AUGMENTED and name in ("train", "val") and AUG_META.exists():
        src = AUG_META
    out = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") not in ids or not r.get("images"):
                continue
            if USE_AUGMENTED and name in ("train", "val"):
                a = (r.get("description_augmented") or "").strip()
                if not a:
                    continue
                r = dict(r, description=a)
            if (r.get("description") or "").strip():
                out.append(r)
    return out


def load_image(rec):
    for rel in rec["images"]:
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return Image.new("RGB", (224, 224), (255, 255, 255))


class DS(Dataset):
    def __init__(self, recs, proc, augment):
        self.recs, self.proc, self.augment = recs, proc, augment

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        img = load_image(rec)
        if self.augment:
            img = TRAIN_AUGMENT(img)
        meta = build_metadata_prompt(rec)
        desc = rec["description"].strip()
        enc = self.proc(images=img, text=f"{meta}. {desc}", padding="max_length",
                        truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        prefix = self.proc.tokenizer(f"{meta}. ", add_special_tokens=False)["input_ids"]
        labels[:min(1 + len(prefix), MAX_SEQ_LENGTH)] = -100
        labels[labels == self.proc.tokenizer.pad_token_id] = -100
        return {"pixel_values": enc["pixel_values"].squeeze(0),
                "input_ids": ids,
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels}


def train(epochs):
    print("\n" + "=" * 78)
    print("N19 — DARAZ BLIP, LEAKAGE-FREE SPLIT + CLEANED TEXT")
    print("=" * 78)
    print(f"  metadata: {CLEAN_META.name if CLEAN_META.exists() else METADATA_FILE.name}")
    print(f"  splits  : {CLEAN_SPLITS.name}")

    proc = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.gradient_checkpointing_enable()

    tr, va = load_split("train"), load_split("val")
    print(f"  train={len(tr)}  val={len(va)}")
    tl = DataLoader(DS(tr, proc, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    vl = DataLoader(DS(va, proc, False), batch_size=BATCH_SIZE, num_workers=4)

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    steps = (len(tl) // GRAD_ACCUM) * epochs
    sch = get_cosine_schedule_with_warmup(opt, WARMUP_STEPS, max(steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best = float("inf")

    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for i, b in enumerate(tqdm(tl, desc=f"  ep{ep}/{epochs}", leave=False)):
            b = {k: v.to(DEVICE) for k, v in b.items()}
            with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                loss = model(**b).loss / GRAD_ACCUM
            scaler.scale(loss).backward()
            if (i + 1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad()
            tot += loss.item() * GRAD_ACCUM
        model.eval(); v = 0.0
        with torch.no_grad():
            for b in tqdm(vl, desc=f"  ep{ep} val", leave=False):
                b = {k: x.to(DEVICE) for k, x in b.items()}
                with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                    v += model(**b).loss.item()
        tr_l, va_l = tot / len(tl), v / len(vl)
        print(f"  epoch {ep}: train={tr_l:.4f}  val={va_l:.4f}")
        if va_l < best:
            best = va_l
            (CKPT / "best_model").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(CKPT / "best_model")
            proc.save_pretrained(CKPT / "best_model")
            print(f"    saved (val {va_l:.4f})")
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate():
    print("\n=== EVALUATING (real vs blank image) ===")
    d = CKPT / "best_model"
    if not d.exists():
        raise SystemExit(f"no checkpoint at {d}")
    proc = BlipProcessor.from_pretrained(str(d))
    model = BlipForConditionalGeneration.from_pretrained(
        str(d), torch_dtype=DTYPE).to(DEVICE).eval()

    test = load_split("test")
    refs = [r["description"].strip() for r in test]
    prompts = [build_metadata_prompt(r) for r in test]

    def gen(img, p):
        inp = proc(images=img, text=p, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
            o = model.generate(**inp, max_new_tokens=200, num_beams=4,
                               early_stopping=True, no_repeat_ngram_size=3)
        full = proc.decode(o[0], skip_special_tokens=True).strip()
        cont, _, _ = strip_prefix(full, p)
        return (cont if cont.strip() else "."), full

    real, blank, rows = [], [], []
    for rec, p, ref in tqdm(list(zip(test, prompts, refs)), desc="  infer"):
        a, a_full = gen(load_image(rec), p)
        b, _ = gen(BLANK, p)
        real.append(a); blank.append(b)
        rows.append({"item_id": rec["item_id"], "reference": ref,
                     "real": a, "blank": b, "real_full": a_full})

    with open(GENS, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mr = compute_all_metrics(real, refs)
    mb = compute_all_metrics(blank, refs)
    mr["Visual Attribute Density"] = round(sum(density(x, VISUAL) for x in real) / len(real), 2)
    mb["Visual Attribute Density"] = round(sum(density(x, VISUAL) for x in blank) / len(blank), 2)
    mr["mean_words"] = round(sum(len(x.split()) for x in real) / len(real), 1)
    mb["mean_words"] = round(sum(len(x.split()) for x in blank) / len(blank), 1)

    drop = round(100 * (mr["ROUGE-L"] - mb["ROUGE-L"]) / mr["ROUGE-L"], 2) if mr["ROUGE-L"] else None
    res = {"real_image": mr, "blank_image": mb,
           "rougeL_drop_pct": drop, "n": len(test)}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"\n  {'metric':<26}{'real':>10}{'blank':>10}")
    for k in ("ROUGE-L", "BLEU-1", "METEOR", "CIDEr", "Visual Attribute Density", "mean_words"):
        print(f"  {k:<26}{mr[k]:>10}{mb[k]:>10}")
    print(f"\n  ROUGE-L lost by blanking the image: {drop}%   (n={len(test)})")
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--evaluate-only", action="store_true")
    ap.add_argument("--original-split", action="store_true",
                    help="ablation: use the leaky Milestone-2 split")
    ap.add_argument("--original-text", action="store_true",
                    help="ablation: use uncleaned descriptions")
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for checkpoint and result paths")
    ap.add_argument("--augmented", action="store_true",
                    help="train on LLM-repaired labels; test still scored on seller text")
    ap.add_argument("--v2", action="store_true",
                    help="train on the expanded 3,091-listing corpus")
    a = ap.parse_args()

    if a.v2:
        globals()["CLEAN_SPLITS"] = _V2 / "splits_clean"
        globals()["CLEAN_META"] = _V2 / "metadata" / "listings_final_textclean.jsonl"
        globals()["METADATA_FILE"] = _V2 / "metadata" / "listings_final.jsonl"
        globals()["IMAGES_DIR"] = _V2 / "images"
        if not a.tag:
            a.tag = "_v2"

    globals()["USE_ORIGINAL_SPLIT"] = a.original_split
    globals()["USE_ORIGINAL_TEXT"] = a.original_text
    globals()["USE_AUGMENTED"] = a.augmented
    if a.tag:
        CKPT = ROOT / "models" / "checkpoints" / f"dpd_clean_blip{a.tag}"
        OUT = RESULTS_DIR / f"novelty_dpd_clean_result{a.tag}.json"
        GENS = RESULTS_DIR / f"novelty_dpd_clean_generations{a.tag}.jsonl"
        globals()["CKPT"] = CKPT
        globals()["OUT"] = OUT
        globals()["GENS"] = GENS

    if not a.evaluate_only:
        train(a.epochs)
    evaluate()
