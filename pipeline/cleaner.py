"""
Cleaning pipeline for raw Daraz product listings.

Steps (in order):
  1. Unescape HTML entities in descriptions  (e.g. &amp; → &)
  2. Strip leftover HTML tags
  3. Filter descriptions under MIN_WORDS words
  4. Drop emoji-heavy descriptions (> MAX_EMOJIS emojis)
  5. Drop spam descriptions (excessive ALL-CAPS or repeated punctuation)
  6. Exact-name deduplication  (keep listing with most reviews)
  7. Subcategory labelling     (rule-based keyword match on product name)

Output:
  data/processed/metadata/listings_clean.jsonl  — training-ready records
  data/processed/metadata/clean_stats.json      — cleaning report
"""

import html
import json
import re
from collections import defaultdict
from pathlib import Path

import config

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_WORDS   = 20     # drop descriptions shorter than this
MAX_EMOJIS  = 5      # drop descriptions with more emojis than this
MAX_CAPS_RATIO = 0.5 # drop if >50% of alpha chars are uppercase

# ── Regex helpers ─────────────────────────────────────────────────────────────
_EMOJI_RE     = re.compile(
    "[\U00010000-\U0010FFFF\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]",
    flags=re.UNICODE,
)
_HTML_TAG_RE  = re.compile(r"<[^>]+>")
_REPEAT_RE    = re.compile(r"(.)\1{4,}")          # 5+ repeated chars
_WHITESPACE   = re.compile(r"\s+")


# ── Step 1-2: description normalisation ──────────────────────────────────────

def _clean_description(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ── Step 3-5: quality filters ─────────────────────────────────────────────────

def _is_good_description(text: str) -> tuple[bool, str]:
    """Returns (keep, reason_if_dropped)."""
    words = text.split()

    if len(words) < MIN_WORDS:
        return False, f"short ({len(words)} words)"

    emoji_count = len(_EMOJI_RE.findall(text))
    if emoji_count > MAX_EMOJIS:
        return False, f"emoji-heavy ({emoji_count} emojis)"

    if _REPEAT_RE.search(text):
        return False, "repeated-char spam"

    alpha = [c for c in text if c.isalpha()]
    if alpha:
        caps_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if caps_ratio > MAX_CAPS_RATIO and len(alpha) > 30:
            return False, f"excessive caps ({caps_ratio:.0%})"

    return True, ""


# ── Step 6: exact-name deduplication ─────────────────────────────────────────

def _exact_dedup(products: list[dict]) -> list[dict]:
    """
    Groups products by normalised name. Within each group keeps the
    listing with the most reviews (best quality signal).
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in products:
        key = re.sub(r"\s+", " ", p.get("name", "").lower().strip())
        groups[key].append(p)

    result = []
    for group in groups.values():
        best = max(group, key=lambda p: p.get("num_reviews", 0))
        result.append(best)
    return result


# ── Step 7: subcategory labelling ─────────────────────────────────────────────

# keyword sets are checked in order; first match wins
_SUBCATEGORY_RULES: dict[str, list[tuple[str, list[str]]]] = {
    "consumer-electronics": [
        ("Laptop & PC",       ["laptop", "notebook", "desktop", "pc ", "computer"]),
        ("TV & Display",      ["tv", "television", "monitor", "led tv", "oled", "qled"]),
        ("Audio",             ["speaker", "headphone", "earphone", "earbuds", "soundbar", "headset", "airpods"]),
        ("Camera",            ["camera", "dslr", "mirrorless", "lens", "tripod", "drone"]),
        ("Networking",        ["router", "wifi", "modem", "switch", "access point", "extender"]),
        ("Power & Battery",   ["power bank", "charger", "ups", "inverter", "battery"]),
        ("Smart Home",        ["smart", "alexa", "google home", "automation"]),
        ("Accessories",       ["cable", "adapter", "hub", "keyboard", "mouse", "cover", "case"]),
    ],
    "smartphones": [
        ("Samsung",           ["samsung", "galaxy"]),
        ("Apple",             ["iphone", "apple"]),
        ("Xiaomi",            ["xiaomi", "redmi", "poco"]),
        ("Oppo",              ["oppo", "reno", "find x"]),
        ("Vivo",              ["vivo", "v series", "y series"]),
        ("Realme",            ["realme"]),
        ("Infinix",           ["infinix"]),
        ("Tecno",             ["tecno"]),
        ("Other Brand",       []),  # catch-all
    ],
    "tablets": [
        ("iPad",              ["ipad", "apple tablet"]),
        ("Samsung Tab",       ["samsung tab", "galaxy tab"]),
        ("Android Tablet",    ["tablet", "tab "]),
        ("Drawing Tablet",    ["drawing", "wacom", "graphics tablet"]),
        ("E-Reader",          ["kindle", "kobo", "e-reader", "ebook"]),
    ],
    "womens-fashion": [
        ("Tops & Shirts",     ["shirt", "top", "blouse", "tunic", "kurta"]),
        ("Dresses",           ["dress", "frock", "maxi", "gown", "abaya"]),
        ("Bottoms",           ["trouser", "pant", "jeans", "legging", "skirt", "shorts"]),
        ("Outerwear",         ["jacket", "coat", "hoodie", "sweater", "cardigan"]),
        ("Traditional",       ["shalwar", "kameez", "dupatta", "suit", "lawn"]),
        ("Footwear",          ["shoes", "sandals", "heels", "slippers", "boots", "sneakers"]),
        ("Accessories",       ["scarf", "bag", "belt", "jewellery", "jewelry", "watch"]),
    ],
    "mens-fashion": [
        ("Shirts",            ["shirt", "polo", "t-shirt", "tee"]),
        ("Bottoms",           ["trouser", "pant", "jeans", "shorts", "chinos"]),
        ("Outerwear",         ["jacket", "coat", "hoodie", "sweater", "blazer"]),
        ("Traditional",       ["shalwar", "kameez", "kurta", "waistcoat"]),
        ("Footwear",          ["shoes", "sandals", "slippers", "boots", "sneakers"]),
        ("Accessories",       ["watch", "belt", "wallet", "bag", "sunglasses"]),
    ],
    "home-appliances": [
        ("Refrigerator",      ["refrigerator", "fridge"]),
        ("Washing Machine",   ["washing machine", "washer", "dryer"]),
        ("Air Conditioner",   ["air conditioner", "ac ", " ac,", "inverter ac", "split ac"]),
        ("Microwave & Oven",  ["microwave", "oven", "toaster"]),
        ("Water Dispenser",   ["water dispenser", "water cooler", "purifier"]),
        ("Vacuum & Cleaning", ["vacuum", "cleaner", "mop", "sweeper"]),
        ("Iron & Garment",    ["iron", "steamer", "garment"]),
        ("Kitchen Appliance", ["blender", "juicer", "mixer", "food processor", "kettle", "rice cooker"]),
        ("Fan",               ["fan", "pedestal fan", "ceiling fan", "table fan"]),
    ],
    "beauty-health": [
        ("Skincare",          ["moisturizer", "serum", "sunscreen", "face wash", "toner", "cream", "skincare"]),
        ("Haircare",          ["shampoo", "conditioner", "hair oil", "hair mask", "hair serum"]),
        ("Makeup",            ["lipstick", "foundation", "mascara", "eyeliner", "blush", "makeup", "eyeshadow"]),
        ("Fragrance",         ["perfume", "fragrance", "deodorant", "body mist", "attar"]),
        ("Health Device",     ["blood pressure", "glucose", "thermometer", "oximeter", "weighing scale"]),
        ("Supplements",       ["vitamin", "supplement", "protein", "collagen", "omega"]),
        ("Personal Care",     ["razor", "trimmer", "toothbrush", "nail", "wax"]),
    ],
    "travel-bags": [
        ("Backpack",          ["backpack", "rucksack", "school bag"]),
        ("Handbag",           ["handbag", "purse", "clutch", "tote"]),
        ("Trolley & Luggage", ["trolley", "luggage", "suitcase", "travel bag"]),
        ("Laptop Bag",        ["laptop bag", "laptop sleeve", "laptop backpack"]),
        ("Crossbody",         ["crossbody", "shoulder bag", "sling bag"]),
        ("Wallet",            ["wallet", "cardholder", "card holder"]),
        ("Men's Bag",         ["men", "messenger", "briefcase"]),
    ],
    "toys": [
        ("Action & Figures",  ["action figure", "figurine", "superhero", "robot toy"]),
        ("Building & LEGO",   ["lego", "building block", "construction set"]),
        ("Educational",       ["educational", "learning", "puzzle", "alphabet", "number toy"]),
        ("Remote Control",    ["remote control", "rc car", "rc truck", "drone toy"]),
        ("Stuffed & Plush",   ["stuffed", "plush", "teddy", "doll"]),
        ("Board Game",        ["board game", "card game", "chess", "monopoly", "jenga"]),
        ("Outdoor Play",      ["bicycle", "scooter", "swing", "slide", "trampoline"]),
    ],
}


def _assign_subcategory(product: dict) -> str:
    category  = product.get("category", "")
    name_lower = product.get("name", "").lower()
    rules = _SUBCATEGORY_RULES.get(category, [])

    for subcategory, keywords in rules:
        if not keywords:
            return subcategory  # catch-all
        if any(kw in name_lower for kw in keywords):
            return subcategory

    return "Other"


# ── Main clean pipeline ───────────────────────────────────────────────────────

def clean(products: list[dict]) -> tuple[list[dict], dict]:
    """
    Run all cleaning steps.
    Returns (cleaned_products, stats_dict).
    """
    stats: dict = {
        "input": len(products),
        "dropped_short_desc":   0,
        "dropped_emoji_heavy":  0,
        "dropped_spam":         0,
        "dropped_exact_dupes":  0,
        "output": 0,
    }

    cleaned = []
    for p in products:
        desc = _clean_description(p.get("description", ""))
        keep, reason = _is_good_description(desc)

        if not keep:
            if "short"   in reason: stats["dropped_short_desc"]  += 1
            elif "emoji" in reason: stats["dropped_emoji_heavy"] += 1
            else:                   stats["dropped_spam"]         += 1
            continue

        cleaned.append({**p, "description": desc})

    before_dedup = len(cleaned)
    cleaned = _exact_dedup(cleaned)
    stats["dropped_exact_dupes"] = before_dedup - len(cleaned)

    for p in cleaned:
        p["subcategory"] = _assign_subcategory(p)

    stats["output"] = len(cleaned)
    stats["retention_pct"] = round(100 * stats["output"] / stats["input"], 1)
    return cleaned, stats


# ── Entry point ───────────────────────────────────────────────────────────────

def run_cleaning() -> list[dict]:
    """Load all raw JSONL, clean, write listings_clean.jsonl + stats."""
    raw_products: list[dict] = []
    for path in sorted(Path(config.RAW_DIR).glob("*.jsonl")):
        if path.name == "deduped.jsonl":
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        raw_products.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    print(f"  Loaded {len(raw_products)} raw products")
    cleaned, stats = clean(raw_products)

    # Write outputs
    Path(config.METADATA_DIR).mkdir(parents=True, exist_ok=True)
    out_path   = Path(config.METADATA_DIR) / "listings_clean.jsonl"
    stats_path = Path(config.METADATA_DIR) / "clean_stats.json"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in cleaned:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Print report
    print(f"\n  {'Input products:':<35} {stats['input']}")
    print(f"  {'Dropped (short description):':<35} {stats['dropped_short_desc']}")
    print(f"  {'Dropped (emoji-heavy):':<35} {stats['dropped_emoji_heavy']}")
    print(f"  {'Dropped (spam/caps):':<35} {stats['dropped_spam']}")
    print(f"  {'Dropped (exact duplicates):':<35} {stats['dropped_exact_dupes']}")
    print(f"  {'Output (clean products):':<35} {stats['output']}  ({stats['retention_pct']}% retained)")
    print(f"\n  Written to: {out_path}")

    # Subcategory breakdown
    sub_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in cleaned:
        sub_counts[p["category"]][p["subcategory"]] += 1

    print("\n  Subcategory breakdown:")
    for cat, subs in sorted(sub_counts.items()):
        print(f"    {cat}:")
        for sub, n in sorted(subs.items(), key=lambda x: -x[1]):
            print(f"      {sub:<30} {n}")

    return cleaned
