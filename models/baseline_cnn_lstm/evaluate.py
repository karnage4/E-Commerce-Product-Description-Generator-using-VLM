"""
Evaluate the Show-and-Tell baseline on the test split.

Loads cached features + the best checkpoint, greedy-decodes a description
for every test record, and computes BLEU-1/4, ROUGE-L, METEOR, CIDEr.

Usage:
    python -m models.baseline_cnn_lstm.evaluate
    python -m models.baseline_cnn_lstm.evaluate --max-samples 30
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.nn.utils.rnn import pad_sequence

from models.baseline_cnn_lstm.data_utils import (
    MODEL_FILE, VOCAB_FILE,
    build_or_load_features, encode_prefix, load_records,
)
from models.baseline_cnn_lstm.model import ShowAndTellModel
from models.baseline_cnn_lstm.vocab import Vocab
from models.shared.config import RESULTS_DIR
from models.shared.metrics import compute_all_metrics, save_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No checkpoint at {MODEL_FILE} — run train.py first.")
    if not VOCAB_FILE.exists():
        raise FileNotFoundError(f"No vocab at {VOCAB_FILE} — run train.py first.")

    vocab    = Vocab.load(VOCAB_FILE)
    features = build_or_load_features()
    ckpt     = torch.load(MODEL_FILE, map_location=device, weights_only=True)

    model = ShowAndTellModel(
        vocab_size=ckpt["vocab_size"],
        pad_id=ckpt["pad_id"],
        feature_dim=ckpt["feature_dim"],
        embed_dim=ckpt["embed_dim"],
        hidden_dim=ckpt["hidden_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    records = load_records(args.split)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    print(f"  Evaluating {len(records)} {args.split} samples...")

    hyps: list[str] = []
    refs: list[str] = []
    per_item: list[dict] = []

    pad_id = vocab.pad_id
    for i in range(0, len(records), args.batch_size):
        chunk = records[i : i + args.batch_size]
        prefixes = [
            torch.tensor(
                encode_prefix(r, vocab, max_meta_tokens=ckpt["max_meta_tokens"]),
                dtype=torch.long,
            )
            for r in chunk
        ]
        prefix_ids = pad_sequence(prefixes, batch_first=True, padding_value=pad_id).to(device)
        feats = torch.stack([features[r["item_id"]] for r in chunk]).to(device)

        gen_ids = model.generate(
            feats, prefix_ids,
            eos_id=vocab.eos_id, pad_id=pad_id,
            max_new_tokens=args.max_new_tokens,
            banned_ids=[vocab.unk_id, vocab.pad_id, vocab.bos_id, vocab.sep_id],
        )
        for rec, ids in zip(chunk, gen_ids):
            text = vocab.decode(ids).strip() or "<empty>"
            hyps.append(text)
            refs.append(rec["description"])
            per_item.append({
                "item_id":   rec["item_id"],
                "category":  rec.get("category", ""),
                "generated": text,
                "reference": rec["description"],
            })
        print(f"    {min(i + args.batch_size, len(records))}/{len(records)}")

    metrics = compute_all_metrics(hyps, refs)
    print("\n  CNN+LSTM baseline metrics on", args.split)
    for k, v in metrics.items():
        print(f"    {k:8s}  {v:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, str(RESULTS_DIR / "baseline_cnn_lstm_metrics.json"))
    (RESULTS_DIR / "baseline_cnn_lstm_predictions.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in per_item),
        encoding="utf-8",
    )
    print(f"  Saved predictions -> {RESULTS_DIR / 'baseline_cnn_lstm_predictions.jsonl'}")


if __name__ == "__main__":
    main()
