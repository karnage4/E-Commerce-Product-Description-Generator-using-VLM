"""
Shared PyTorch Dataset class for all model experiments.

Works with data/processed/metadata/listings_final.jsonl and split .txt files.
Each record yields: (PIL.Image, metadata_prompt_str, ground_truth_description)
"""

import json
from pathlib import Path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    build_metadata_prompt
)


class DarazProductDataset(Dataset):
    """
    Loads Daraz product records for vision-language model training.

    Args:
        split:       "train", "val", or "test"
        transform:   Optional torchvision transform applied to PIL images
        max_samples: Cap the dataset size (useful for quick debug runs)
    """

    _SPLIT_FILES = {
        "train": TRAIN_SPLIT,
        "val":   VAL_SPLIT,
        "test":  TEST_SPLIT,
    }

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        assert split in self._SPLIT_FILES, f"split must be one of {list(self._SPLIT_FILES)}"
        self.transform   = transform
        self.images_root = IMAGES_DIR          # data/processed/images/

        # Load item_ids for this split
        split_ids = set(
            Path(self._SPLIT_FILES[split]).read_text(encoding="utf-8").strip().splitlines()
        )

        # Load all metadata records and filter to this split
        self.records: list[dict] = []
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
                if not rec.get("description", "").strip():
                    continue

                self.records.append(rec)
                if max_samples and len(self.records) >= max_samples:
                    break

        print(f"  [{split}] Loaded {len(self.records)} samples")

    # ──────────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # ── Load first available image ────────────────────────────────────────
        image = self._load_image(rec)

        # ── Build text prompt from metadata ───────────────────────────────────
        prompt = build_metadata_prompt(rec)

        # ── Ground-truth description ──────────────────────────────────────────
        description = rec["description"].strip()

        if self.transform is not None:
            image = self.transform(image)

        return image, prompt, description

    # ──────────────────────────────────────────────────────────────────────────

    def _load_image(self, rec: dict) -> Image.Image:
        """Load the first image for a product record (RGB)."""
        # rec["images"] is a list of relative paths like "images/{item_id}/0.jpg"
        for rel_path in rec["images"]:
            # IMAGES_DIR is data/processed/images; rel_path starts with "images/..."
            # So we need to go up one level to data/processed/
            abs_path = IMAGES_DIR.parent / rel_path
            if abs_path.exists():
                try:
                    return Image.open(abs_path).convert("RGB")
                except Exception:
                    continue
        # Fallback: 224x224 white image if all loads fail
        return Image.new("RGB", (224, 224), color=(255, 255, 255))

    # ──────────────────────────────────────────────────────────────────────────

    def get_record(self, idx: int) -> dict:
        """Return the raw metadata record for inspection / qualitative eval."""
        return self.records[idx]


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = DarazProductDataset(split="train", max_samples=5)
    for i in range(len(ds)):
        img, prompt, desc = ds[i]
        print(f"\n[{i}] Image size: {img.size}")
        print(f"    Prompt: {prompt[:120]}...")
        print(f"    Description: {desc[:120]}...")
