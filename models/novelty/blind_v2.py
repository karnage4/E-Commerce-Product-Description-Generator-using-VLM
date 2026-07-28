"""
Experiment N26 — Blind baseline on the v2 corpus.

The DPD blind-vs-vision comparison (N11) ran on v1: 873 training samples and a
135-item leaky test split. v2 provides 2,472 leakage-controlled training
samples and a 309-item test set — enough to put meaningful intervals on the
DPD side of the paper's central table.

This reuses blind_baseline.py's trainer unchanged, pointed at the v2 splits and
the markup-cleaned metadata, with the epoch budget matched to the v2 vision
model (12, best-val checkpointing). The paired vision model is
checkpoints/dpd_clean_blip_v2b (same splits, same text, same budget).

Run:
    python -m models.novelty.blind_v2 --train --evaluate --epochs 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.novelty import blind_baseline as B
from models.shared.config import RESULTS_DIR

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "data" / "processed"


def repoint():
    meta = V2 / "metadata" / "listings_final_textclean.jsonl"
    if not meta.exists():
        meta = V2 / "metadata" / "listings_final.jsonl"
    B.METADATA_FILE = meta
    splits = V2 / "splits_clean"
    B.BlindTextDataset._SPLITS = {
        "train": splits / "train.txt",
        "val": splits / "val.txt",
        "test": splits / "test.txt",
    }
    B.CKPT_DIR = ROOT / "models" / "checkpoints" / "blind_gpt2_v2"
    B.RESULTS_FILE = RESULTS_DIR / "novelty_blind_v2_results.jsonl"
    B.METRICS_FILE = RESULTS_DIR / "novelty_blind_v2_metrics.json"
    print(f"  metadata: {meta.name}")
    print(f"  splits  : {splits}")
    print(f"  ckpt    : {B.CKPT_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    repoint()
    if a.train:
        B.train(a.epochs)
    if a.evaluate:
        B.evaluate()
