"""
BLIP Fine-tuning — Kaggle (T4/P100), trained on AUGMENTED descriptions.

Differences vs train_colab.py
─────────────────────────────
1. Loads `description_augmented` field if present (NIM-augmented), else
   falls back to `description`. This is the whole point — train on
   visually-grounded text instead of metadata echoes.
2. Correct loss masking: loss is computed ONLY on description tokens.
   Metadata-prompt tokens, CLS/BOS, and padding are masked to -100.
   Fixes the bug where the model was wasting capacity learning to
   predict its own input.
3. Paths default to Kaggle layout (/kaggle/input, /kaggle/working).

How to use on Kaggle
────────────────────
  1. Upload daraz_dataset_kaggle.zip as a Kaggle Dataset.
     Suppose Kaggle assigns it the slug "your-username/daraz-vlm-augmented".
  2. Create a new Notebook. Settings → Accelerator: GPU T4 x1.
  3. Add Data → search for your dataset → Add.
  4. Inside the notebook, FIRST CELL:

       !pip install -q transformers==4.40.0 accelerate==0.30.0

  5. SECOND CELL: paste this entire file. Edit DATASET_SLUG below if needed.
  6. Run. Training takes ~45-60 min for 5 epochs on T4 with 1100 train items.
  7. Checkpoint saved to /kaggle/working/checkpoints/blip/best_model/.
     Click "Save Version" on Kaggle to persist /kaggle/working/.
"""

import json
import os
from pathlib import Path

# Fix CUDA memory fragmentation BEFORE importing torch
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


# ── Paths (Kaggle layout) ─────────────────────────────────────────────────────
# !! EDIT THIS to match your dataset's slug after you Add Data.
# It will look like /kaggle/input/<slug>/  — find it in the right sidebar.
DATASET_SLUG   = "daraz-vlm-augmented"

DATA_ROOT      = Path(f"/kaggle/input/{DATASET_SLUG}")
METADATA_FILE  = DATA_ROOT / "metadata" / "listings.jsonl"
IMAGES_DIR     = DATA_ROOT / "images"
SPLITS_DIR     = DATA_ROOT / "splits"
CHECKPOINT_DIR = Path("/kaggle/working/checkpoints/blip")


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "Salesforce/blip-image-captioning-base"
LEARNING_RATE  = 1e-5
BATCH_SIZE     = 2
GRAD_ACCUM     = 4               # effective batch size = 8
NUM_EPOCHS     = 5
WEIGHT_DECAY   = 0.01
MAX_SEQ_LENGTH = 128
WARMUP_STEPS   = 100
USE_FP16       = True

# Which text field to train on. We prefer the augmented one (rich, visually
# grounded) but fall back to the original description if augmentation is
# missing for a given record.
PRIMARY_FIELD   = "description_augmented"
FALLBACK_FIELD  = "description"


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


def pick_target_text(rec: dict) -> str:
    """Use augmented description when present and non-empty, else fall back."""
    aug = (rec.get(PRIMARY_FIELD) or "").strip()
    if aug:
        return aug
    return (rec.get(FALLBACK_FIELD) or "").strip()


# ── Dataset ───────────────────────────────────────────────────────────────────

class DarazBlipDataset(Dataset):
    def __init__(self, split: str, processor: BlipProcessor, max_samples=None):
        split_ids = set(
            (SPLITS_DIR / f"{split}.txt").read_text(encoding="utf-8").strip().splitlines()
        )
        self.processor = processor
        self.records: list[dict] = []
        n_aug = n_fallback = 0

        with open(METADATA_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("item_id") not in split_ids:
                    continue
                if not rec.get("images"):
                    continue
                target = pick_target_text(rec)
                if not target:
                    continue
                # Track which field we're using for sanity
                if rec.get(PRIMARY_FIELD):
                    n_aug += 1
                else:
                    n_fallback += 1
                self.records.append(rec)
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"  BLIP [{split}] {len(self.records)} samples  "
              f"(augmented={n_aug}, fallback={n_fallback})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load first available image (paths in zip are forward-slash-normalized)
        image = None
        for rel in rec.get("images", []):
            rel = rel.replace("\\", "/")
            p = DATA_ROOT / rel
            if p.exists():
                try:
                    image = Image.open(p).convert("RGB")
                    break
                except Exception:
                    pass
        if image is None:
            image = Image.new("RGB", (224, 224), (255, 255, 255))

        metadata    = build_metadata_prompt(rec)
        description = pick_target_text(rec)
        combined    = f"{metadata}. {description}"

        encoding = self.processor(
            images=image,
            text=combined,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        pixel_values   = encoding["pixel_values"].squeeze(0)

        # ── CORRECTED LOSS MASK ────────────────────────────────────────────────
        # Goal: compute loss ONLY on description tokens. The metadata prompt
        # is part of the input/context, not the prediction target. Without
        # this fix, BLIP learns to predict its own metadata input — wasted
        # capacity and the source of "metadata-echo" hallucinations at inference.
        #
        # We tokenize "metadata." alone (no special tokens) to count how many
        # token positions belong to metadata, then add 1 for the [CLS]/[BOS]
        # token that the full encoding prepends.
        meta_only_ids = self.processor.tokenizer(
            metadata + ".",
            add_special_tokens=False,
        ).input_ids
        n_meta_tokens = min(len(meta_only_ids) + 1, MAX_SEQ_LENGTH)

        labels = input_ids.clone()
        labels[:n_meta_tokens] = -100
        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Sanity check helper (call once before training) ─────────────────────────────
def sanity_check(ds: DarazBlipDataset, processor):
    """Print one example so we can confirm only description tokens are unmasked."""
    sample = ds[0]
    visible_label_ids = [t for t in sample["labels"].tolist() if t != -100]
    decoded_label = processor.tokenizer.decode(visible_label_ids, skip_special_tokens=True)
    print("\n  ── sanity check ──")
    print(f"  Number of unmasked label tokens: {len(visible_label_ids)}")
    print(f"  These tokens decode to:\n    {decoded_label[:400]}")
    print("  (should look like the description, NOT the metadata prefix)\n")


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n  Loading {MODEL_NAME}...")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model     = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.gradient_checkpointing_enable()
    model = model.to(device)

    train_ds = DarazBlipDataset("train", processor)
    val_ds   = DarazBlipDataset("val",   processor)

    sanity_check(train_ds, processor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

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
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False)

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM
            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += outputs.loss.item()
            pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

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
                    outputs = model(**batch)
                val_loss += outputs.loss.item()
                del batch, outputs
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\n  Epoch {epoch}: train={avg_train_loss:.4f}  val={avg_val_loss:.4f}")

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
