"""
BLIP Fine-tuning — runs locally (GPU) or on Google Colab.

Run:
    python -m models.blip.train_colab
    python -m models.blip.train_colab --subset-samples 200 --learning-rate 3e-5
    python -m models.blip.train_colab --epochs 3 --batch-size 8
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ── Fix CUDA memory fragmentation BEFORE importing torch ─────────────────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.shared.config import RESULTS_DIR, TRAINING_SWEEP


# ── Image augmentation (training split only) ──────────────────────────────────
# Applied to PIL images before the BLIP processor converts them to tensors.
# Val/test images skip this — they always use the processor's default transform.
TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10),
])


# ── Paths (derived from this file's location — works locally and on Colab) ───
_HERE          = Path(__file__).resolve().parent
_PROJECT_ROOT  = _HERE.parents[1]
DATA_ROOT      = _PROJECT_ROOT / "data" / "data" / "processed"
METADATA_FILE  = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_DIR     = DATA_ROOT / "images"
SPLITS_DIR     = DATA_ROOT / "splits"
CHECKPOINT_DIR = _PROJECT_ROOT / "models" / "checkpoints" / "blip"

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Salesforce/blip-image-captioning-base"
LEARNING_RATE  = 1e-5
BATCH_SIZE     = 4       # 4 fits comfortably on 17GB VRAM; lower to 2 if OOM
GRAD_ACCUM     = 2       # effective batch size = BATCH_SIZE * GRAD_ACCUM = 8
NUM_EPOCHS     = 5
WEIGHT_DECAY   = 0.01
MAX_SEQ_LENGTH = 128     # ONE unified sequence length (see explanation below)
WARMUP_STEPS   = 100
USE_FP16       = True

# ─────────────────────────────────────────────────────────────────────────────
# WHY ONE MAX_SEQ_LENGTH (fixes the ValueError batch_size mismatch):
#
# BlipForConditionalGeneration is an encoder-decoder model:
#   - IMAGE        → vision encoder
#   - TEXT         → text decoder  (input_ids)
#   - labels       → MUST be the SAME sequence as input_ids, just with
#                    padding tokens replaced by -100
#
# The original code had input_ids (MAX_TEXT_LENGTH=128) and labels
# (MAX_LABEL_LEN=200) at DIFFERENT lengths, causing:
#   ValueError: Expected input batch_size (1016) to match target batch_size (1592)
#   (1016 = 8 samples x 127 tokens, 1592 = 8 x 199 tokens)
#
# Fix: combine metadata + description into ONE string, tokenize once,
# use those same token ids for both input_ids AND labels.
# ─────────────────────────────────────────────────────────────────────────────


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_metadata_prompt(rec: dict) -> str:
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    for k, v in list((rec.get("attributes") or {}).items())[:3]:
        if k and v:
            parts.append(f"{k}: {v}")
    return ". ".join(parts)


class DarazBlipDataset(Dataset):
    def __init__(self, split: str, processor: BlipProcessor,
                 max_samples=None, use_augmented=False):
        split_ids = set((SPLITS_DIR / f"{split}.txt").read_text().strip().splitlines())
        self.processor      = processor
        self.use_augmented  = use_augmented
        self.split          = split
        self.apply_augment  = (split == "train")
        self.records: list[dict] = []

        # Use augmented file if requested and it exists, else fall back to original
        # Augmented descriptions only exist for the train split
        source_file = METADATA_FILE
        aug_file    = METADATA_FILE.parent / "listings_augmented.jsonl"
        if use_augmented and split == "train" and aug_file.exists():
            source_file = aug_file
            print(f"  BLIP [{split}] using augmented descriptions")

        with open(source_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("item_id") not in split_ids:
                    continue
                desc_key = "description_augmented" if (use_augmented and split == "train") else "description"
                if not rec.get("images") or not rec.get(desc_key, "").strip():
                    continue
                self.records.append(rec)
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"  BLIP [{split}] {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load image
        image = None
        for rel in rec["images"]:
            p = IMAGES_DIR.parent / rel
            if p.exists():
                try:
                    image = Image.open(p).convert("RGB")
                    break
                except Exception:
                    pass
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        if self.apply_augment:
            image = TRAIN_AUGMENT(image)

        metadata    = build_metadata_prompt(rec)
        desc_key    = "description_augmented" if (self.use_augmented and self.split == "train") else "description"
        description = rec.get(desc_key, rec.get("description", "")).strip()
        combined_text = f"{metadata}. {description}"

        encoding = self.processor(
            images=image,
            text=combined_text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values   = encoding["pixel_values"].squeeze(0)

        # Loss masking: only description tokens contribute to loss.
        # Tokenize metadata prefix (no special tokens) to find its length,
        # then mask [CLS] + metadata tokens + padding to -100.
        prefix_ids = self.processor.tokenizer(
            f"{metadata}. ",
            add_special_tokens=False,
        )["input_ids"]
        prefix_len = min(1 + len(prefix_ids), MAX_SEQ_LENGTH)  # +1 for [CLS]

        labels = input_ids.clone()
        labels[:prefix_len] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train(subset_samples=None, learning_rate=LEARNING_RATE,
          num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
          save_checkpoints=True, use_augmented=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  [!] WARNING: No GPU detected. Training will be very slow.")

    if subset_samples:
        print(f"  Subset mode: {subset_samples} samples")

    print(f"\n  Loading {MODEL_NAME}...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Gradient checkpointing: recomputes activations during backward pass
    # instead of storing them — saves ~30-40% VRAM at cost of ~20% slower training
    model.gradient_checkpointing_enable()
    model = model.to(device)

    train_ds = DarazBlipDataset("train", processor, max_samples=subset_samples, use_augmented=use_augmented)
    val_ds   = DarazBlipDataset("val",   processor,  max_samples=subset_samples // 5 if subset_samples else None, use_augmented=use_augmented)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * num_epochs
    scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    scaler      = torch.amp.GradScaler("cuda", enabled=USE_FP16)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    epoch_history: list[dict] = []

    for epoch in range(1, num_epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()   # reset once before loop for gradient accumulation

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=USE_FP16):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / GRAD_ACCUM

            scaler.scale(loss).backward()

            # Step optimizer every GRAD_ACCUM batches
            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.item()
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

            # Aggressively free GPU memory each step
            del batch, outputs, loss
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [val]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda", enabled=USE_FP16):
                    outputs = model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                val_loss += outputs.loss.item()
                del batch, outputs
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        print(f"  Epoch {epoch}/{num_epochs}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")
        epoch_history.append({"epoch": epoch,
                               "train_loss": round(avg_train_loss, 6),
                               "val_loss":   round(avg_val_loss,   6)})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if save_checkpoints:
                best_path = CHECKPOINT_DIR / "best_model"
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path)
                print(f"  [BEST] Saved -> {best_path}")

        if save_checkpoints:
            ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}"
            model.save_pretrained(ckpt_path)
            processor.save_pretrained(ckpt_path)
            print(f"  Saved -> {ckpt_path}")

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    if save_checkpoints:
        print(f"  Best checkpoint: {CHECKPOINT_DIR / 'best_model'}")
        hist_file = RESULTS_DIR / "training_history_blip.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        import json as _json
        hist_file.write_text(_json.dumps(epoch_history, indent=2), encoding="utf-8")
        print(f"  History saved  : {hist_file}")
    return best_val_loss


# ── Auto HP sweep + full training ─────────────────────────────────────────────

def sweep_and_train(batch_size=BATCH_SIZE, use_augmented=False):
    """Sweep learning rates on a small subset, save results, then full training."""
    sw = TRAINING_SWEEP["blip"]
    lr_values    = sw["learning_rate"]
    subset       = sw["subset_samples"]
    sweep_epochs = sw["num_epochs"]

    print(f"\n{'='*55}")
    print(f"  BLIP HP SWEEP — lr ∈ {lr_values}")
    print(f"  Subset: {subset} samples  |  Epochs: {sweep_epochs}")
    print(f"{'='*55}")

    sweep_results = []
    for lr in lr_values:
        print(f"\n  [Sweep] lr={lr:.1e}")
        val_loss = train(
            subset_samples=subset,
            learning_rate=lr,
            num_epochs=sweep_epochs,
            batch_size=batch_size,
            save_checkpoints=False,
            use_augmented=use_augmented,
        )
        sweep_results.append({"learning_rate": lr, "val_loss": round(val_loss, 6)})
        print(f"  [Sweep] lr={lr:.1e}  →  val_loss={val_loss:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sweep_file = RESULTS_DIR / "training_sweep_blip.json"
    with open(sweep_file, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"\n  Sweep results saved → {sweep_file}")
    print(f"\n  {'lr':>10}  {'val_loss':>10}")
    print(f"  {'-'*22}")
    for r in sweep_results:
        marker = "  ← best" if r == min(sweep_results, key=lambda x: x["val_loss"]) else ""
        print(f"  {r['learning_rate']:>10.1e}  {r['val_loss']:>10.4f}{marker}")

    best_lr = min(sweep_results, key=lambda x: x["val_loss"])["learning_rate"]
    print(f"\n  Best LR: {best_lr:.1e}  → starting full training")

    print(f"\n{'='*55}")
    print(f"  BLIP FULL TRAINING — lr={best_lr:.1e}, {NUM_EPOCHS} epochs")
    print(f"{'='*55}")
    train(learning_rate=best_lr, batch_size=batch_size, use_augmented=use_augmented)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-sweep",     action="store_true",
                        help="Sweep LRs on a subset first, then full training with best LR")
    parser.add_argument("--subset-samples", type=int, default=None,
                        help="Train on this many samples only (skips sweep)")
    parser.add_argument("--learning-rate",  type=float, default=LEARNING_RATE)
    parser.add_argument("--epochs",         type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--augmented",      action="store_true",
                        help="Train on Gemma-augmented descriptions instead of raw Daraz text")
    args = parser.parse_args()

    if args.auto_sweep:
        sweep_and_train(batch_size=args.batch_size, use_augmented=args.augmented)
    else:
        train(
            subset_samples=args.subset_samples,
            learning_rate=args.learning_rate,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            use_augmented=args.augmented,
        )
