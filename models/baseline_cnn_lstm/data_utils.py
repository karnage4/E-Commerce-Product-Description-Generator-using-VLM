"""
Shared helpers for the Show-and-Tell baseline:
  - Caching frozen MobileNetV2 features for every record in every split.
  - Encoding a record's (metadata, description) into LSTM input/target ids.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    build_metadata_prompt,
)
from models.baseline_cnn_lstm.model import ImageFeatureExtractor
from models.baseline_cnn_lstm.vocab import Vocab, tokenize


SPLIT_FILES = {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT}

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "models" / "checkpoints" / "baseline_cnn_lstm"
FEATURES_FILE  = CHECKPOINT_DIR / "mobilenetv2_features.pt"
VOCAB_FILE     = CHECKPOINT_DIR / "vocab.json"
MODEL_FILE     = CHECKPOINT_DIR / "best.pt"


# ── Record loading ────────────────────────────────────────────────────────────

def load_records(split: str) -> list[dict]:
    ids = set(Path(SPLIT_FILES[split]).read_text(encoding="utf-8").split())
    out: list[dict] = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("item_id") not in ids:
                continue
            if not rec.get("images"):
                continue
            if not rec.get("description", "").strip():
                continue
            out.append(rec)
    return out


def _resolve_first_image(rec: dict) -> Path | None:
    for rel in rec["images"]:
        p = IMAGES_DIR.parent / rel
        if p.exists():
            return p
    return None


# ── Feature caching ───────────────────────────────────────────────────────────

@torch.no_grad()
def build_or_load_features(force: bool = False, batch_size: int = 16) -> dict[str, torch.Tensor]:
    """Cache MobileNetV2 global-pool features for every (item_id) we will use."""
    if FEATURES_FILE.exists() and not force:
        cache = torch.load(FEATURES_FILE, map_location="cpu", weights_only=True)
        print(f"  Loaded cached features for {len(cache)} items.")
        return cache

    print("  Computing MobileNetV2 features (one-time)...")
    extractor = ImageFeatureExtractor()
    cache: dict[str, torch.Tensor] = {}
    placeholder = Image.new("RGB", (224, 224), color=(255, 255, 255))

    todo: list[tuple[str, Path | None]] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        for rec in load_records(split):
            iid = rec["item_id"]
            if iid in seen:
                continue
            seen.add(iid)
            todo.append((iid, _resolve_first_image(rec)))

    for i in range(0, len(todo), batch_size):
        chunk = todo[i : i + batch_size]
        imgs = []
        for _, p in chunk:
            try:
                img = Image.open(p).convert("RGB") if p else placeholder
            except Exception:
                img = placeholder
            imgs.append(extractor.preprocess(img))
        batch = torch.stack(imgs)
        feats = extractor(batch).cpu()
        for (iid, _), f in zip(chunk, feats):
            cache[iid] = f
        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(chunk)}/{len(todo)}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, FEATURES_FILE)
    print(f"  Saved {len(cache)} features -> {FEATURES_FILE}")
    return cache


# ── Token-sequence assembly ───────────────────────────────────────────────────

@dataclass
class EncodedSample:
    input_ids: list[int]   # what the LSTM sees at step t
    target_ids: list[int]  # what the LSTM should predict at step t (-100 = ignore)


def encode_sample(
    record: dict,
    vocab: Vocab,
    max_meta_tokens: int = 30,
    max_desc_tokens: int = 80,
) -> EncodedSample:
    meta_tok = tokenize(build_metadata_prompt(record))[:max_meta_tokens]
    desc_tok = tokenize(record["description"])[:max_desc_tokens]

    meta_ids = vocab.encode(meta_tok)
    desc_ids = vocab.encode(desc_tok)

    bos, sep, eos, unk = vocab.bos_id, vocab.sep_id, vocab.eos_id, vocab.unk_id

    # Full sequence: [BOS] meta [SEP] desc [EOS]
    seq = [bos] + meta_ids + [sep] + desc_ids + [eos]
    # LSTM input is the sequence without the final token;
    # target is shifted by one.
    input_ids = seq[:-1]
    full_tgt  = seq[1:]
    # Mask loss on positions whose target is metadata or [SEP].
    # Targets to keep: anything after the [SEP] token in the input.
    sep_pos_in_input = 1 + len(meta_ids)   # index of [SEP] in input_ids
    target_ids = [-100] * len(full_tgt)
    for i in range(sep_pos_in_input, len(full_tgt)):
        tok = full_tgt[i]
        # Don't train the model to emit <unk> -- that just teaches it to give up.
        target_ids[i] = -100 if tok == unk else tok
    return EncodedSample(input_ids=input_ids, target_ids=target_ids)


def encode_prefix(
    record: dict,
    vocab: Vocab,
    max_meta_tokens: int = 30,
) -> list[int]:
    """Prefix used at inference: [BOS] meta [SEP]."""
    meta_tok = tokenize(build_metadata_prompt(record))[:max_meta_tokens]
    return [vocab.bos_id] + vocab.encode(meta_tok) + [vocab.sep_id]


# ── Torch Dataset wrapping cached features + encoded text ─────────────────────

class FeatureCaptionDataset(Dataset):
    def __init__(
        self,
        split: str,
        vocab: Vocab,
        features: dict[str, torch.Tensor],
        max_meta_tokens: int = 30,
        max_desc_tokens: int = 80,
    ):
        self.records = load_records(split)
        self.vocab = vocab
        self.features = features
        self.max_meta = max_meta_tokens
        self.max_desc = max_desc_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        sample = encode_sample(rec, self.vocab, self.max_meta, self.max_desc)
        feat = self.features[rec["item_id"]]
        return {
            "feat":      feat,
            "input_ids": torch.tensor(sample.input_ids, dtype=torch.long),
            "target_ids": torch.tensor(sample.target_ids, dtype=torch.long),
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    feats = torch.stack([b["feat"] for b in batch])
    max_t = max(b["input_ids"].numel() for b in batch)

    inp  = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    tgt  = torch.full((len(batch), max_t), -100,  dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["input_ids"].numel()
        inp[i, :t] = b["input_ids"]
        tgt[i, :t] = b["target_ids"]
    return {"feat": feats, "input_ids": inp, "target_ids": tgt}


def training_corpus_for_vocab() -> Iterable[str]:
    for rec in load_records("train"):
        yield build_metadata_prompt(rec)
        yield rec["description"]
