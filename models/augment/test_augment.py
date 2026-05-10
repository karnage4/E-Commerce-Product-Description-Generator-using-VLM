"""
Sanity-check the augmentation pipeline on a tiny sample.

Pulls 5 random training records, runs them through NIM, prints the
original description and the new augmented description side-by-side
so you can eyeball quality before kicking off the full run.

Run:
  python -m models.augment.test_augment

If outputs look weak, edit the AUGMENTATION_PROMPT in
augment_descriptions.py and re-run this until the prompt is dialed in.
THEN run the full augmentation.
"""

import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Reuse the heavy lifting from the main script
from models.augment.augment_descriptions import (
    BASE_URL, MODEL, METADATA_FILE, TRAIN_SPLIT,
    augment_one, load_records_to_augment,
)


SAMPLE_SIZE = 5
RANDOM_SEED = 42   # deterministic samples so prompt iterations are comparable


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.exit("[!] NVIDIA_API_KEY not set in .env")

    client = OpenAI(base_url=BASE_URL, api_key=api_key)

    records = load_records_to_augment()
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(records, min(SAMPLE_SIZE, len(records)))

    print(f"Sampling {len(sample)} of {len(records)} train records "
          f"(seed={RANDOM_SEED})\n")

    for i, rec in enumerate(sample, 1):
        print("=" * 78)
        print(f"[{i}/{len(sample)}] item_id={rec.get('item_id')}")
        print(f"  category : {rec.get('category')}")
        print(f"  brand    : {rec.get('brand')}")
        print(f"  name     : {rec.get('item_name')}")
        print(f"\n  ORIGINAL DESCRIPTION:\n  {rec.get('description', '')[:300]}")

        augmented = augment_one(client, rec)
        if augmented is None:
            print("\n  [!] FAILED to augment this record")
            continue

        print(f"\n  AUGMENTED DESCRIPTION:\n  {augmented}")
        print()


if __name__ == "__main__":
    main()
