"""
Experiment N14 — The corpus comparison, with the training pipeline held constant.

N13 compared image-sensitivity on ABO against image-sensitivity on Daraz, and
concluded that the difference is caused by the catalogue rather than the corpus
size. That conclusion has one remaining hole: the two BLIP models were not
trained by the same code. The ABO model came from models/novelty/abo_head2head.py;
the Daraz model came from the original Milestone-2 pipeline, with different batch
size, different augmentation, a different epoch schedule and a different
truncation convention.

A reviewer will say the difference could be the training setup. They would be
right to.

This script removes that objection by training Daraz through the *identical*
code path as ABO — same dataset class, same loss masking, same optimiser, same
schedule, same prompt and description budgets, same decoding, same prefix
stripping at evaluation — by reusing abo_head2head's functions and swapping only
the data source.

The original Daraz checkpoints are untouched and remain the Milestone-2 baseline.

Run:
    python -m models.novelty.matched_pipeline --epochs 5
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.novelty import abo_head2head as H
from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    RESULTS_DIR, build_metadata_prompt,
)

_SPLIT_FILES = {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT}

_CLEAN_DIR = TRAIN_SPLIT.parent.parent / "splits_clean"
_CLEAN_FILES = {n: _CLEAN_DIR / f"{n}.txt" for n in ("train", "val", "test")}


USE_CLEAN = False


def dpd_load_split(name):
    """Daraz records, reshaped into the field names abo_head2head expects."""
    files = _CLEAN_FILES if USE_CLEAN else _SPLIT_FILES
    ids = set(Path(files[name]).read_text(encoding="utf-8").split())
    out = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") not in ids:
                continue
            if not r.get("images") or not (r.get("description") or "").strip():
                continue
            r["abo_metadata_prompt"] = build_metadata_prompt(r)
            out.append(r)
    if name == "train" and H.TRAIN_N:
        import random
        random.Random(H.SEED).shuffle(out)
        out = out[:H.TRAIN_N]
    return out


def dpd_load_image(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return Image.new("RGB", (224, 224), (255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--eval-n", type=int, default=135)
    ap.add_argument("--counterfactual-n", type=int, default=135)
    ap.add_argument("--skip-blind", action="store_true")
    ap.add_argument("--clean-split", action="store_true",
                    help="use data/data/processed/splits_clean (no item crosses train/test)")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="reuse existing checkpoints; skip training (crash recovery)")
    args = ap.parse_args()

    global USE_CLEAN
    USE_CLEAN = args.clean_split

    print("\n" + "=" * 78)
    print("N14 — DARAZ THROUGH THE IDENTICAL ABO PIPELINE")
    print("=" * 78)
    print("  Same code, same hyperparameters, same budgets, same decoding as N12/N13.")
    print("  Only the corpus differs.\n")

    # Swap the data source; everything downstream is abo_head2head's own code.
    H.load_split = dpd_load_split
    H.load_image = dpd_load_image
    suffix = args.tag or ("_clean" if USE_CLEAN else "")
    H.CKPT = H.CKPT.parent / f"dpd_matched{suffix}"
    H.OUT = RESULTS_DIR / f"novelty_matched_pipeline_dpd{suffix}.json"
    print(f"  splits: {'splits_clean (leakage-free)' if USE_CLEAN else 'original'}")
    print(f"  epochs: {args.epochs}\n")

    if not args.evaluate_only:
        if not args.skip_blind:
            H.train_blind(args.epochs)
        H.train_vision(args.epochs)
    H.evaluate(args.eval_n, args.counterfactual_n)


if __name__ == "__main__":
    main()
