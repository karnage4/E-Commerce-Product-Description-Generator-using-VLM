"""
Train the Show-and-Tell baseline locally (CPU-friendly).

Steps:
  1. Cache MobileNetV2 features for every image (one-time, persisted to disk).
  2. Build a word-level vocab from the training split.
  3. Train an LSTM decoder conditioned on (image feature, metadata prefix).

Usage:
    python -m models.baseline_cnn_lstm.train
    python -m models.baseline_cnn_lstm.train --epochs 3 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline_cnn_lstm.data_utils import (
    CHECKPOINT_DIR, MODEL_FILE, VOCAB_FILE,
    FeatureCaptionDataset, build_or_load_features, collate,
    training_corpus_for_vocab,
)
from models.baseline_cnn_lstm.model import ImageFeatureExtractor, ShowAndTellModel
from models.baseline_cnn_lstm.vocab import Vocab


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-meta-tokens", type=int, default=30)
    p.add_argument("--max-desc-tokens", type=int, default=80)
    p.add_argument("--rebuild-features", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def run_one_epoch(model, loader, opt, criterion, device, train: bool):
    model.train(train)
    total_loss, total_tok = 0.0, 0
    for batch in loader:
        feat = batch["feat"].to(device)
        inp  = batch["input_ids"].to(device)
        tgt  = batch["target_ids"].to(device)

        logits = model(feat, inp)                       # (B, T, V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        n_tok = int((tgt != -100).sum().item())

        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        total_loss += float(loss.item()) * max(n_tok, 1)
        total_tok  += max(n_tok, 1)
    return total_loss / total_tok


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Image features (cached) ────────────────────────────────────────────
    features = build_or_load_features(force=args.rebuild_features)

    # 2. Vocab ────────────────────────────────────────────────────────────────
    if VOCAB_FILE.exists():
        vocab = Vocab.load(VOCAB_FILE)
        print(f"  Loaded vocab ({len(vocab)} tokens) <- {VOCAB_FILE.name}")
    else:
        print("  Building vocab from training corpus...")
        vocab = Vocab.build(list(training_corpus_for_vocab()), min_freq=args.min_freq)
        vocab.save(VOCAB_FILE)
        print(f"  Saved vocab ({len(vocab)} tokens) -> {VOCAB_FILE.name}")

    # 3. Datasets / loaders ───────────────────────────────────────────────────
    train_ds = FeatureCaptionDataset("train", vocab, features,
                                     args.max_meta_tokens, args.max_desc_tokens)
    val_ds   = FeatureCaptionDataset("val",   vocab, features,
                                     args.max_meta_tokens, args.max_desc_tokens)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)}")

    collate_fn = partial(collate, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn)

    # 4. Model ────────────────────────────────────────────────────────────────
    model = ShowAndTellModel(
        vocab_size=len(vocab),
        pad_id=vocab.pad_id,
        feature_dim=ImageFeatureExtractor.FEATURE_DIM,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    history = []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_one_epoch(model, train_loader, opt, criterion, device, train=True)
        val_loss   = run_one_epoch(model, val_loader,   opt, criterion, device, train=False)
        sched.step()
        dt = time.time() - t0
        ppl = math.exp(min(val_loss, 20))
        print(f"  [{epoch:2d}/{args.epochs}] train={train_loss:.4f}  val={val_loss:.4f}  "
              f"ppl={ppl:.2f}  lr={opt.param_groups[0]['lr']:.2e}  {dt:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                         "val_ppl": ppl, "time_s": round(dt, 1)})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "pad_id": vocab.pad_id,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "feature_dim": ImageFeatureExtractor.FEATURE_DIM,
                "max_meta_tokens": args.max_meta_tokens,
                "max_desc_tokens": args.max_desc_tokens,
            }, MODEL_FILE)
            print(f"      [best] val improved -- saved {MODEL_FILE.name}")

    (CHECKPOINT_DIR / "history.json").write_text(json.dumps(history, indent=2))
    print(f"\n  Done. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
