"""
BLIP Fine-tuning — designed to run on Google Colab (free T4 GPU).

HOW TO USE:
  1. Zip and upload your dataset:
       - data/processed/metadata/listings_final.jsonl
       - data/processed/splits/ (train.txt, val.txt, test.txt)
       - data/processed/images/ (all product image folders)
     Upload the zip to Google Drive.

  2. Open this file as a Colab notebook (or copy its content there).

  3. Mount Google Drive in Colab:
       from google.colab import drive
       drive.mount('/content/drive')

  4. Unzip the dataset to /content/daraz_data/

  5. Run all cells. Training takes ~1-2 hours on Colab T4 for 5 epochs.

  6. Download the checkpoint from /content/checkpoints/blip/best_model/
     back to your local machine after training.

Required Colab packages (first cell):
  !pip install transformers accelerate pillow tqdm
"""

import json
import os
from pathlib import Path

# ── Fix CUDA memory fragmentation BEFORE importing torch ─────────────────────
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm


# ── Paths (edit these to match your Colab setup) ──────────────────────────────
DATA_ROOT      = Path("/content/daraz_data")
METADATA_FILE  = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_DIR     = DATA_ROOT / "images"
SPLITS_DIR     = DATA_ROOT / "splits"
CHECKPOINT_DIR = Path("/content/checkpoints/blip")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Salesforce/blip-image-captioning-base"
LEARNING_RATE  = 1e-5
BATCH_SIZE     = 2       # reduced from 8 — OOM fix for 15GB VRAM
GRAD_ACCUM     = 4       # effective batch size = BATCH_SIZE * GRAD_ACCUM = 8
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
    def __init__(self, split: str, processor: BlipProcessor, max_samples=None):
        split_ids = set((SPLITS_DIR / f"{split}.txt").read_text().strip().splitlines())
        self.processor = processor
        self.records: list[dict] = []

        with open(METADATA_FILE, encoding="utf-8") as f:
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
                if not rec.get("images") or not rec.get("description", "").strip():
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

        # Combine metadata + description into ONE text string
        # BLIP decoder sees the full combined text and learns to reproduce it
        metadata = build_metadata_prompt(rec)
        description = rec["description"].strip()
        combined_text = f"{metadata}. {description}"

        # Single processor call: handles image AND text together
        # pixel_values: (1, 3, 384, 384) — BLIP uses 384px internally
        # input_ids:    (1, MAX_SEQ_LENGTH) — the combined text
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

        # labels = same token ids, padding masked to -100 (ignored in loss)
        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  [!] WARNING: No GPU detected. Training will be very slow.")

    print(f"\n  Loading {MODEL_NAME}...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)

    # Gradient checkpointing: recomputes activations during backward pass
    # instead of storing them — saves ~30-40% VRAM at cost of ~20% slower training
    model.gradient_checkpointing_enable()
    model = model.to(device)

    train_ds = DarazBlipDataset("train", processor)
    val_ds   = DarazBlipDataset("val",   processor)

    # num_workers=0 avoids Colab shared-memory errors
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    scaler      = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()   # reset once before loop for gradient accumulation

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False)

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                # Divide loss by accumulation steps to keep gradient scale correct
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
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=USE_FP16):
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
        print(f"\n  Epoch {epoch}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

        # Save per-epoch checkpoint
        ckpt_path = CHECKPOINT_DIR / f"epoch_{epoch}"
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        print(f"  Saved -> {ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = CHECKPOINT_DIR / "best_model"
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            print(f"  [BEST] Saved -> {best_path}")

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Best checkpoint: {CHECKPOINT_DIR / 'best_model'}")


if __name__ == "__main__":
    train()
