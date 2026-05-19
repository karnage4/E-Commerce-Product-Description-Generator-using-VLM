"""
Main entry point for the Daraz dataset pipeline.

Usage:
  python run.py                          # run all steps
  python run.py --steps scrape           # only scrape
  python run.py --steps clean            # clean raw data → listings_clean.jsonl
  python run.py --steps dedup            # fuzzy dedup  (uses existing raw data)
  python run.py --steps build            # download images + write splits
  python run.py --steps scrape clean     # scrape then clean
  python run.py --categories smartphones home-appliances
"""

import argparse
import json
from pathlib import Path

import config
from scraper.daraz_scraper import scrape_category
from pipeline.cleaner import run_cleaning
from dedup.deduplicator import deduplicate
from organizer.dataset_builder import build_dataset


DEDUP_CACHE = Path(config.RAW_DIR) / "deduped.jsonl"


def load_raw_products() -> list[dict]:
    """Load all checkpointed raw products from data/raw/*.jsonl."""
    products = []
    for path in Path(config.RAW_DIR).glob("*.jsonl"):
        if path.name == "deduped.jsonl":
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        products.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return products


def step_scrape(categories: dict[str, str]) -> None:
    print("=" * 60)
    print("STEP 1 — Scraping Daraz")
    print("=" * 60)
    for slug, name in categories.items():
        print(f"\n[+] Category: {name} ({slug})")
        scrape_category(slug, name)


def step_dedup() -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 2 — Deduplication")
    print("=" * 60)

    if DEDUP_CACHE.exists():
        print(f"  [=] Loading cached dedup result from {DEDUP_CACHE}")
        products = []
        with open(DEDUP_CACHE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    products.append(json.loads(line))
        print(f"  Loaded {len(products)} deduplicated products")
        return products

    raw = load_raw_products()
    if not raw:
        print("  [!] No raw products found. Run the scrape step first.")
        return []

    deduped = deduplicate(raw)

    DEDUP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(DEDUP_CACHE, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Cached to {DEDUP_CACHE}")

    return deduped


def step_clean() -> list[dict]:
    print("\n" + "=" * 60)
    print("STEP 2 — Cleaning")
    print("=" * 60)
    return run_cleaning()


def step_build(products: list[dict] = None) -> None:
    print("\n" + "=" * 60)
    print("STEP 3 — Building Dataset (downloading images + splits)")
    print("=" * 60)
    build_dataset(products if products else None)
    print("\nDone! Dataset written to:", config.PROCESSED_DIR)


def main():
    parser = argparse.ArgumentParser(description="Daraz dataset pipeline")
    parser.add_argument(
        "--steps", nargs="+",
        choices=["scrape", "clean", "dedup", "build"],
        default=["scrape", "clean", "dedup", "build"],
        help="Which pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--categories", nargs="+",
        help="Category slugs to scrape (default: all from config)",
    )
    # Force re-run dedup even if cache exists
    parser.add_argument(
        "--rededup", action="store_true",
        help="Ignore dedup cache and re-run deduplication",
    )
    args = parser.parse_args()

    if args.rededup and DEDUP_CACHE.exists():
        DEDUP_CACHE.unlink()
        print("[!] Cleared dedup cache — will re-run deduplication")

    # Resolve category filter
    if args.categories:
        categories = {
            slug: config.CATEGORIES[slug]
            for slug in args.categories
            if slug in config.CATEGORIES
        }
        unknown = set(args.categories) - set(config.CATEGORIES)
        if unknown:
            print(f"[!] Unknown categories (ignored): {unknown}")
    else:
        categories = config.CATEGORIES

    # Run steps
    if "scrape" in args.steps:
        step_scrape(categories)

    if "clean" in args.steps:
        step_clean()

    if "build" in args.steps:
        step_build([])


if __name__ == "__main__":
    main()
