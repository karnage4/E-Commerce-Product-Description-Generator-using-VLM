"""
Central configuration for the Daraz dataset scraper.
"""

# ── Scraping targets ─────────────────────────────────────────────────────────
# Logical category keys → friendly names (must match keys in CATEGORY_SLUGS in scraper)
CATEGORIES = {
    "consumer-electronics": "Electronics",
    "smartphones":          "Smartphones",
    "tablets":              "Tablets",
    "womens-fashion":       "Women's Fashion",
    "mens-fashion":         "Men's Fashion",
    "home-appliances":      "Home Appliances",
    "beauty-health":        "Beauty & Health",
    "travel-bags":          "Bags & Travel",
    "toys":                 "Toys",
}

# Maximum listings to collect per category (before dedup)
MAX_LISTINGS_PER_CATEGORY = 500

# ── Rate limiting ────────────────────────────────────────────────────────────
REQUEST_DELAY_MIN = 1.5   # seconds
REQUEST_DELAY_MAX = 3.5   # seconds
MAX_RETRIES       = 3
RETRY_BACKOFF     = 5     # seconds added per retry

# ── Deduplication thresholds ─────────────────────────────────────────────────
# Fuzzy title similarity (0-100). Listings above this are considered duplicates.
TITLE_SIMILARITY_THRESHOLD = 82

# Perceptual hash Hamming distance. Images this close are considered identical.
IMAGE_HASH_THRESHOLD = 8

# Among duplicate groups, keep the listing with the most reviews.
DEDUP_KEEP_STRATEGY = "most_reviews"   # or "highest_rating"

# ── Dataset split ratios ──────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR            = "data"
RAW_DIR             = f"{DATA_DIR}/raw"
PROCESSED_DIR       = f"{DATA_DIR}/processed"
IMAGES_DIR          = f"{PROCESSED_DIR}/images"
METADATA_DIR        = f"{PROCESSED_DIR}/metadata"
SPLITS_DIR          = f"{PROCESSED_DIR}/splits"

# ── HTTP headers ─────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": "https://www.daraz.pk/",
    "X-Requested-With": "XMLHttpRequest",
}
