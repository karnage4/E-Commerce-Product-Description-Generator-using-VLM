"""
CLIP + GPT-2 Fine-tuning — Self-contained Colab cell.

IMPORTANT: This file merges model.py + train_colab.py into ONE cell so you
can paste everything directly into Colab without uploading separate files.

HOW TO USE IN COLAB:
  1. The dataset must already be at /content/daraz_data/ (from BLIP step)
  2. Paste the ENTIRE content of this file into a new Colab cell
  3. Run it — training will start automatically
  4. Download checkpoint when done (Cell after this one)

Architecture:
  Image  ->  CLIP ViT encoder  ->  CLS embedding (768-dim)
                                        |
                               Linear projection layer
                                        |
                          Visual prefix (10 tokens x 768-dim)
                                        |
  Metadata text -> GPT-2 tokenizer -> prompt embeddings
                                        |
             [visual_prefix | prompt_embeds | description_embeds]
                                        |
                                 GPT-2 LM head
                                        |
                              Generated description

Key difference from BLIP:
  - CLIP-GPT2 uses SEPARATE encoding for metadata (prompt) and description (labels)
    because GPT-2 is a CAUSAL LM (decoder-only), not encoder-decoder like BLIP.
  - Loss is ONLY computed on description tokens (metadata prefix is masked to -100)
  - This is architecturally correct and will NOT cause the batch_size mismatch error
"""

import json
import os
from pathlib import Path

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    CLIPVisionModel,
    CLIPProcessor,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = Path("/content/daraz_data")
METADATA_FILE  = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_DIR     = DATA_ROOT / "images"
SPLITS_DIR     = DATA_ROOT / "splits"
CHECKPOINT_DIR = Path("/content/checkpoints/clip_gpt2")

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_MODEL     = "openai/clip-vit-base-patch32"
GPT2_MODEL     = "gpt2"
PREFIX_LENGTH  = 10        # number of visual prefix tokens fed to GPT-2
LEARNING_RATE  = 2e-5
CLIP_LR_MULT   = 0.1      # CLIP encoder LR multiplier when unfrozen
BATCH_SIZE     = 2         # keep low for 15GB VRAM; effective batch = 2 * GRAD_ACCUM
GRAD_ACCUM     = 4         # effective batch size = 8
NUM_EPOCHS     = 5
UNFREEZE_EPOCH = 3         # unfreeze CLIP encoder from this epoch onward
WEIGHT_DECAY   = 0.01
MAX_TEXT_LEN   = 64        # metadata prompt tokens
MAX_LABEL_LEN  = 128       # description tokens (reduced for VRAM)
WARMUP_STEPS   = 100
USE_FP16       = True


# ── CLIP image transform (224x224, CLIP normalisation) ────────────────────────
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION (from model.py)
# ══════════════════════════════════════════════════════════════════════════════

class ClipGPT2Model(nn.Module):
    """
    Vision-language model: CLIP visual encoder + GPT-2 language model.

    The CLIP image encoder is frozen for the first UNFREEZE_EPOCH epochs,
    then unfrozen with a 10x smaller learning rate for joint fine-tuning.
    """

    def __init__(
        self,
        clip_model_name: str = CLIP_MODEL,
        gpt2_model_name: str = GPT2_MODEL,
        prefix_length:   int = PREFIX_LENGTH,
        freeze_clip:    bool = True,
    ):
        super().__init__()
        self.prefix_length = prefix_length

        # CLIP vision encoder — ViT-B/32 (hidden_size=768)
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_embed_dim = self.clip.config.hidden_size

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # GPT-2 language model (n_embd=768 for gpt2-base)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2_embed_dim = self.gpt2.config.n_embd

        # Projection: CLIP CLS token -> prefix_length GPT-2 embedding vectors
        self.visual_projection = nn.Sequential(
            nn.Linear(self.clip_embed_dim, self.gpt2_embed_dim * prefix_length),
            nn.Tanh(),
        )

    def get_visual_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, 224, 224) -> visual_prefix: (B, P, 768)"""
        clip_out      = self.clip(pixel_values=pixel_values)
        cls_embedding = clip_out.pooler_output                            # (B, 768)
        prefix_flat   = self.visual_projection(cls_embedding)             # (B, P*768)
        return prefix_flat.view(-1, self.prefix_length, self.gpt2_embed_dim)

    def forward(
        self,
        pixel_values:   torch.Tensor,   # (B, 3, 224, 224)
        input_ids:      torch.Tensor,   # (B, T_text)  — metadata prompt tokens
        attention_mask: torch.Tensor,   # (B, T_text)
        labels:         torch.Tensor,   # (B, T_label) — description tokens, padding=-100
    ) -> torch.Tensor:
        """
        Returns scalar training loss.

        GPT-2 sees: [visual_prefix (P)] + [metadata_prompt (T_text)] + [description (T_label)]
        Loss is computed ONLY on description tokens (prefix and prompt are masked).

        Note: Unlike BLIP, input_ids (T_text) and labels (T_label) CAN have different
        lengths here — this is correct for a causal LM because they represent different
        parts of the sequence (prompt vs target), not the same sequence at different offsets.
        """
        B      = pixel_values.size(0)
        device = pixel_values.device
        P      = self.prefix_length
        T_text = input_ids.size(1)

        # 1. Encode image to visual prefix
        visual_prefix = self.get_visual_prefix(pixel_values)     # (B, P, 768)

        # 2. Embed metadata prompt and description via GPT-2's embedding table
        wte           = self.gpt2.transformer.wte
        prompt_embeds = wte(input_ids)                            # (B, T_text, 768)
        label_embeds  = wte(labels.clamp(min=0))                  # (B, T_label, 768)

        # 3. Concatenate: [prefix | prompt | description]
        combined_embeds = torch.cat([visual_prefix, prompt_embeds, label_embeds], dim=1)

        # 4. Build labels: -100 for prefix+prompt positions, real ids for description
        prefix_ignore = torch.full((B, P + T_text), -100, dtype=torch.long, device=device)
        full_labels   = torch.cat([prefix_ignore, labels], dim=1)

        # 5. Build attention mask: 1 for prefix+prompt, real mask for description
        prefix_mask = torch.ones(B, P, device=device)
        label_mask  = (labels != -100).long()
        full_attn   = torch.cat([prefix_mask, attention_mask, label_mask], dim=1)

        outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values:         torch.Tensor,
        input_ids:            torch.Tensor,
        attention_mask:       torch.Tensor,
        max_new_tokens:       int = 150,
        num_beams:            int = 4,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        """Beam-search generation at inference time."""
        visual_prefix = self.get_visual_prefix(pixel_values)
        prompt_embeds = self.gpt2.transformer.wte(input_ids)
        combined      = torch.cat([visual_prefix, prompt_embeds], dim=1)

        prefix_mask = torch.ones(input_ids.size(0), self.prefix_length, device=visual_prefix.device)
        full_attn   = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.gpt2.generate(
            inputs_embeds=combined,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            pad_token_id=self.gpt2.config.eos_token_id,
        )

    def unfreeze_clip(self):
        for p in self.clip.parameters():
            p.requires_grad = True
        print("  CLIP encoder unfrozen")

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":           f"{total:,}",
            "trainable":       f"{trainable:,}",
            "pct_trainable":   f"{100 * trainable / total:.2f}%",
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

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


class DarazCLIPGPT2Dataset(Dataset):
    def __init__(self, split: str, tokenizer: GPT2Tokenizer, max_samples=None):
        split_ids = set((SPLITS_DIR / f"{split}.txt").read_text().strip().splitlines())
        self.tokenizer = tokenizer
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

        print(f"  CLIP-GPT2 [{split}] {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Load image and apply CLIP transform
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
        pixel_values = CLIP_TRANSFORM(image)   # (3, 224, 224)

        # Tokenize metadata prompt (encoder-side, used as visual context prefix)
        prompt_enc = self.tokenizer(
            build_metadata_prompt(rec),
            max_length=MAX_TEXT_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize description (decoder target)
        label_enc = self.tokenizer(
            rec["description"].strip(),
            max_length=MAX_LABEL_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = label_enc["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100   # mask padding

        return {
            "pixel_values":   pixel_values,
            "input_ids":      prompt_enc["input_ids"].squeeze(0),
            "attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "labels":         labels,
        }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:  {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # GPT-2 tokenizer — set pad token to EOS (GPT-2 has none by default)
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    # Build model
    model = ClipGPT2Model(freeze_clip=True).to(device)
    params = model.count_parameters()
    print(f"  Parameters: {params['total']} total, "
          f"{params['trainable']} trainable ({params['pct_trainable']})")

    # Datasets and loaders
    train_ds = DarazCLIPGPT2Dataset("train", tokenizer)
    val_ds   = DarazCLIPGPT2Dataset("val",   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Optimizer: register BOTH param groups from the start ────────────────
    # This prevents the scheduler zip error.
    # If we used add_param_group() later, the scheduler's base_lrs list would
    # be out of sync with optimizer.param_groups and raise:
    #   ValueError: zip() argument 2 is shorter than argument 1
    #
    # Fix: two groups from day 1 — CLIP starts with lr=0 (effectively frozen).
    # At UNFREEZE_EPOCH we just change lr; no add_param_group needed.
    clip_params  = list(model.clip.parameters())
    other_params = [p for p in model.parameters() if not any(p is cp for cp in clip_params)]

    optimizer = torch.optim.AdamW(
        [
            {"params": clip_params,  "lr": 0.0},           # group 0 — CLIP (frozen: lr=0)
            {"params": other_params, "lr": LEARNING_RATE},  # group 1 — projection + GPT-2
        ],
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler   = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)
    scaler      = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):

        # Unfreeze CLIP encoder from UNFREEZE_EPOCH onwards
        if epoch == UNFREEZE_EPOCH:
            model.unfreeze_clip()
            # Just update the lr on the pre-registered CLIP param group (group 0)
            # and sync the scheduler's base_lrs so future scheduler.step() calls
            # use the right target lr for CLIP.
            clip_lr = LEARNING_RATE * CLIP_LR_MULT
            optimizer.param_groups[0]["lr"]    = clip_lr
            scheduler.base_lrs[0]              = clip_lr   # keep scheduler in sync
            print(f"  [Epoch {epoch}] CLIP unfrozen — lr set to {clip_lr:.2e}")

        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch}/{NUM_EPOCHS} [train]", leave=False)

        for step, batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                loss = model(**batch) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            actual_loss = loss.item() * GRAD_ACCUM
            train_loss += actual_loss
            pbar.set_postfix(loss=f"{actual_loss:.4f}")

            del batch, loss
            torch.cuda.empty_cache()

        avg_train = train_loss / len(train_loader)

        # ── Validate ────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [val]", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=USE_FP16):
                    loss = model(**batch)
                val_loss += loss.item()
                del batch, loss
                torch.cuda.empty_cache()

        avg_val = val_loss / len(val_loader)
        print(f"\n  Epoch {epoch}: train={avg_train:.4f}  val={avg_val:.4f}")

        # Save per-epoch checkpoint
        ckpt = CHECKPOINT_DIR / f"epoch_{epoch}"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt / "model.pt")
        tokenizer.save_pretrained(str(ckpt))
        print(f"  Saved -> {ckpt}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best = CHECKPOINT_DIR / "best_model"
            best.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best / "model.pt")
            tokenizer.save_pretrained(str(best))
            print(f"  [BEST] Saved -> {best}")

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoint: {CHECKPOINT_DIR / 'best_model'}")


# ── Run ───────────────────────────────────────────────────────────────────────
train()
