"""
Deduplication pipeline for Daraz product listings.

The same physical product is frequently listed by multiple sellers with
different item_ids. We detect duplicates through two passes:

  Pass 1 — Title similarity
    Normalize titles (lowercase, strip punctuation, collapse whitespace),
    then use RapidFuzz token_sort_ratio to group listings whose titles
    are above TITLE_SIMILARITY_THRESHOLD.

  Pass 2 — Perceptual image hash
    For pairs that survived pass 1 (high title similarity), download the
    thumbnail and compute a phash. If the Hamming distance is below
    IMAGE_HASH_THRESHOLD, confirm as duplicate.

Within each duplicate group we keep the single listing with the most
reviews (or highest rating, depending on config.DEDUP_KEEP_STRATEGY).
"""

import io
import re
import time
import random
from collections import defaultdict

import requests
from PIL import Image
import imagehash
from rapidfuzz import fuzz
from tqdm import tqdm

import config


# ── Title normalisation ───────────────────────────────────────────────────────

_PUNCT_RE  = re.compile(r"[^\w\s]")
_SPACE_RE  = re.compile(r"\s+")
_UNITS_RE  = re.compile(
    r"\b(\d+)\s*(pcs?|pack|pieces?|set|kg|g|ml|l|cm|mm|inch|inches|\")\b",
    re.IGNORECASE,
)

def _normalize_title(title: str) -> str:
    """
    Produce a canonical form of a product title for comparison.
    - Lowercase
    - Collapse unit suffixes (e.g. '500 ML' → '500ml')
    - Remove punctuation
    - Collapse whitespace
    """
    t = title.lower()
    t = _UNITS_RE.sub(lambda m: m.group(1) + m.group(2).lower(), t)
    t = _PUNCT_RE.sub(" ", t)
    t = _SPACE_RE.sub(" ", t).strip()
    return t


# ── Image hash comparison ─────────────────────────────────────────────────────

_IMG_HEADERS = {
    "User-Agent": config.HEADERS["User-Agent"],
}

def _phash_from_url(url: str) -> imagehash.ImageHash | None:
    """Download image and return its perceptual hash, or None on failure."""
    if not url:
        return None
    try:
        time.sleep(random.uniform(0.3, 0.8))
        resp = requests.get(url, headers=_IMG_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return imagehash.phash(img)
    except Exception:
        return None


# ── Core deduplication ────────────────────────────────────────────────────────

def deduplicate(products: list[dict]) -> list[dict]:
    """
    Takes the full list of scraped products (all categories combined),
    returns a deduplicated list.
    """
    print(f"\n[Dedup] Input: {len(products)} listings")

    # ── Pass 1: title-similarity clustering ──────────────────────────────────
    normalized = [_normalize_title(p["name"]) for p in products]
    groups: list[set[int]] = []   # each set is a group of indices
    assigned = [False] * len(products)

    for i in tqdm(range(len(products)), desc="  Title clustering", unit="item"):
        if assigned[i]:
            continue
        group = {i}
        for j in range(i + 1, len(products)):
            if assigned[j]:
                continue
            score = fuzz.token_sort_ratio(normalized[i], normalized[j])
            if score >= config.TITLE_SIMILARITY_THRESHOLD:
                group.add(j)
                assigned[j] = True
        assigned[i] = True
        groups.append(group)

    # Groups with more than one member are potential duplicates
    dup_groups     = [g for g in groups if len(g) > 1]
    unique_indices = [list(g)[0] for g in groups if len(g) == 1]

    print(f"  Title pass: {len(dup_groups)} duplicate groups, "
          f"{len(unique_indices)} singletons")

    # ── Pass 2: confirm with perceptual hash ──────────────────────────────────
    confirmed_keepers = []

    for group in tqdm(dup_groups, desc="  Image hash verification", unit="group"):
        members = list(group)

        # Build hash map for each member (only download if group size > 1)
        hashes: dict[int, imagehash.ImageHash | None] = {}
        for idx in members:
            url = products[idx].get("image_url", "")
            hashes[idx] = _phash_from_url(url)

        # Cluster by image hash distance
        img_groups: list[set[int]] = []
        img_assigned = set()
        for i in members:
            if i in img_assigned:
                continue
            img_group = {i}
            for j in members:
                if j == i or j in img_assigned:
                    continue
                hi, hj = hashes[i], hashes[j]
                if hi is not None and hj is not None:
                    if (hi - hj) <= config.IMAGE_HASH_THRESHOLD:
                        img_group.add(j)
                        img_assigned.add(j)
                # If either hash failed, fall back to title similarity alone
                # (already grouped — keep together)
            img_assigned.add(i)
            img_groups.append(img_group)

        for img_group in img_groups:
            keeper = _pick_best(products, list(img_group))
            confirmed_keepers.append(keeper)

    # ── Collect final deduplicated list ──────────────────────────────────────
    kept_from_singletons = [products[i] for i in unique_indices]
    final = kept_from_singletons + confirmed_keepers

    print(f"  After dedup: {len(final)} unique products "
          f"(removed {len(products) - len(final)} duplicates)")
    return final


def _pick_best(products: list[dict], indices: list[int]) -> dict:
    """From a duplicate group, return the 'best' listing."""
    candidates = [products[i] for i in indices]

    if config.DEDUP_KEEP_STRATEGY == "most_reviews":
        return max(candidates, key=lambda p: p.get("num_reviews", 0))
    elif config.DEDUP_KEEP_STRATEGY == "highest_rating":
        return max(candidates, key=lambda p: p.get("rating", 0.0))
    else:
        return candidates[0]
