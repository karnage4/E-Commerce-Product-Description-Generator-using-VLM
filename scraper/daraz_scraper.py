"""
Daraz category scraper.

Strategy
--------
1. Launch a headless Chromium browser with playwright-stealth to bypass
   Daraz's bot fingerprinting.
2. Solve the slider CAPTCHA automatically (Alibaba/Daraz x5sec challenge).
3. Intercept the internal AJAX response (`?ajax=true`) that fires when a
   category page loads — this contains all product data as clean JSON.
4. After the CAPTCHA is solved once, export browser cookies to a
   `requests.Session` for fast product detail page fetching.

The AJAX endpoint:
  GET /catalog/?q={category_name}&from=hp_categories&src=all_channel
                &ajax=true&page={n}
returns JSON with 40 products/page under mods.listItems.
Descriptions are already embedded, so detail pages are only fetched for
multi-angle images and the specifications table.

Checkpoints raw listings to data/raw/{category_key}.jsonl so the run
can be resumed without re-scraping.
"""

import json
import random
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page
from playwright_stealth import Stealth
from tqdm import tqdm

import config

BASE_URL = "https://www.daraz.pk"

# Logical category key → search query used in /catalog/?q=
CATEGORY_QUERIES: dict[str, str] = {
    "consumer-electronics": "Electronics",
    "smartphones":          "Smartphones",
    "tablets":              "Tablets",
    "womens-fashion":       "Women Fashion",
    "mens-fashion":         "Men Fashion",
    "home-appliances":      "Home Appliances",
    "beauty-health":        "Beauty Health",
    "travel-bags":          "Bags Travel",
    "toys":                 "Toys",
}


# ── Slider CAPTCHA solver ─────────────────────────────────────────────────────

def _solve_slider(page: Page) -> bool:
    """
    Detects and solves Daraz's 'Please slide to verify' CAPTCHA by
    simulating a human-like drag across the slider track.
    Returns True if a slider was found and dragged.
    """
    for sel in [".btn_slide", "[class*='nc_btn']", "[class*='sliderbtn']",
                "button[class*='slide']", ".nc-lang-cnt .btn_slide"]:
        slider = page.query_selector(sel)
        if slider is None:
            continue
        box = slider.bounding_box()
        if not box:
            continue

        sx = box["x"] + box["width"] / 2
        sy = box["y"] + box["height"] / 2
        drag_distance = 285  # px — enough to reach track end

        page.mouse.move(sx, sy)
        page.mouse.down()
        time.sleep(0.08)

        steps = 35
        for i in range(steps + 1):
            t = i / steps
            # Ease-in then ease-out (mimics human acceleration curve)
            ease = 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t
            page.mouse.move(
                sx + drag_distance * ease,
                sy + random.uniform(-0.8, 0.8),
            )
            time.sleep(random.uniform(0.008, 0.022))

        time.sleep(0.12)
        page.mouse.up()
        return True

    return False


# ── AJAX interception ─────────────────────────────────────────────────────────

def _fetch_listing_page(page: Page, category_query: str, pg: int) -> list[dict]:
    """
    Navigate to the category page. The browser fires an internal AJAX call
    which we intercept to get clean product JSON (40 items/page).
    If a slider CAPTCHA appears, solve it automatically and retry.
    """
    url = (
        f"{BASE_URL}/catalog/"
        f"?q={requests.utils.quote(category_query)}"
        f"&from=hp_categories&src=all_channel&ajax=true&page={pg}"
    )
    captured: list[dict] = []

    def _on_response(response):
        ct = response.headers.get("content-type", "")
        if "json" not in ct or response.status != 200:
            return
        try:
            data = response.json()
            text = json.dumps(data)
            if "listItems" not in text:
                return
            m = re.search(
                r'"listItems"\s*:\s*(\[.+?\])\s*,\s*"(?:breadcrumb|filter)',
                text, re.DOTALL,
            )
            if m:
                captured.extend(json.loads(m.group(1)))
        except Exception:
            pass

    page.on("response", _on_response)
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_timeout(1500)

    # Solve slider CAPTCHA if present, then wait for AJAX to retry
    if page.query_selector(".btn_slide") or page.query_selector("[class*='nc_btn']"):
        _solve_slider(page)
        page.wait_for_timeout(2500)

    page.remove_listener("response", _on_response)
    return captured


# ── Listing collection ────────────────────────────────────────────────────────

def _parse_listing_item(raw: dict, category_key: str) -> dict | None:
    item_id = str(raw.get("itemId") or raw.get("nid", "")).strip()
    if not item_id:
        return None

    item_url = raw.get("itemUrl", "")
    if item_url.startswith("//"):
        item_url = "https:" + item_url
    elif item_url.startswith("/"):
        item_url = BASE_URL + item_url

    desc_raw = raw.get("description", "")
    description = (
        " ".join(str(x) for x in desc_raw).strip()
        if isinstance(desc_raw, list)
        else str(desc_raw).strip()
    )

    return {
        "item_id":     item_id,
        "seller_id":   str(raw.get("sellerId", "")),
        "name":        raw.get("name", "").strip(),
        "url":         item_url,
        "image_url":   raw.get("image", ""),
        "price_pkr":   _parse_price(raw.get("price") or raw.get("priceShow", "0")),
        "rating":      float(raw.get("ratingScore") or 0),
        "num_reviews": _parse_int(raw.get("review", 0)),
        "seller_name": raw.get("sellerName", "").strip(),
        "brand_name":  raw.get("brandName", "").strip(),
        "description": description,
        "category":    category_key,
    }


def _scrape_listings(page: Page, category_key: str,
                     category_query: str, max_listings: int) -> list[dict]:
    results: list[dict] = []
    seen_ids: set[str] = set()
    pg = 1
    empty_streak = 0

    pbar = tqdm(total=max_listings, desc=f"  Listing [{category_key}]", unit="item")

    while len(results) < max_listings:
        raw_items = _fetch_listing_page(page, category_query, pg)
        if not raw_items:
            empty_streak += 1
            if empty_streak >= 2:
                break
            pg += 1
            continue
        empty_streak = 0

        added = 0
        for raw in raw_items:
            item = _parse_listing_item(raw, category_key)
            if item and item["item_id"] not in seen_ids:
                seen_ids.add(item["item_id"])
                results.append(item)
                pbar.update(1)
                added += 1
                if len(results) >= max_listings:
                    break

        if added == 0:
            break
        pg += 1
        time.sleep(random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX))

    pbar.close()
    return results


# ── Product detail pages ──────────────────────────────────────────────────────

def _make_requests_session(page: Page) -> requests.Session:
    """Copy cookies from Playwright into a requests.Session."""
    s = requests.Session()
    s.headers.update(config.HEADERS)
    for c in page.context.cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain", ""))
    return s


def scrape_product_detail(session: requests.Session, listing: dict) -> dict:
    url = listing.get("url", "")
    if not url:
        return {**listing, "all_images": [listing["image_url"]], "attributes": {}}

    for attempt in range(config.MAX_RETRIES):
        try:
            time.sleep(random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX))
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                break
        except requests.RequestException:
            time.sleep(config.RETRY_BACKOFF * (attempt + 1))
    else:
        return {**listing, "all_images": [listing["image_url"]], "attributes": {}}

    soup = BeautifulSoup(resp.text, "lxml")
    html_desc  = _extract_description(soup)
    all_images = _extract_images(soup) or [listing["image_url"]]
    attributes = _extract_attributes(soup)

    return {
        **listing,
        "description": listing["description"] or html_desc,
        "all_images":  all_images,
        "attributes":  attributes,
    }


def _extract_description(soup: BeautifulSoup) -> str:
    desc_div = (
        soup.find("div", {"class": re.compile(r"module-description|pdp-product-desc", re.I)})
        or soup.find("div", {"id": re.compile(r"description", re.I)})
    )
    if desc_div:
        return desc_div.get_text(separator=" ", strip=True)
    meta = soup.find("meta", {"property": "og:description"})
    return meta.get("content", "").strip() if meta else ""


def _extract_images(soup: BeautifulSoup) -> list[str]:
    for script in soup.find_all("script"):
        text = script.string or ""
        m = re.search(r'"images"\s*:\s*(\[.*?\])', text, re.DOTALL)
        if m:
            try:
                urls = []
                for entry in json.loads(m.group(1)):
                    url = (entry.get("url") or entry.get("image") or "") \
                        if isinstance(entry, dict) else str(entry)
                    if url.startswith("http"):
                        urls.append(url)
                if urls:
                    return urls
            except (json.JSONDecodeError, TypeError):
                pass

    gallery = soup.find("div", {"class": re.compile(r"gallery|pdp-block__main-img", re.I)})
    if gallery:
        return [u for u in (i.get("src") or i.get("data-src", "") for i in gallery.find_all("img")) if u.startswith("http")]
    return []


def _extract_attributes(soup: BeautifulSoup) -> dict[str, str]:
    attrs: dict[str, str] = {}
    spec = soup.find(["ul", "div"], {"class": re.compile(r"specification|attr|product-prop", re.I)})
    if spec:
        for li in spec.find_all("li"):
            parts = li.get_text(separator="\n", strip=True).split("\n")
            if len(parts) >= 2:
                attrs[parts[0].strip()] = parts[1].strip()
        if attrs:
            return attrs
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["th", "td"])
            if len(cells) == 2:
                k = cells[0].get_text(strip=True)
                v = cells[1].get_text(strip=True)
                if k:
                    attrs[k] = v
    return attrs


# ── Full pipeline for one category ───────────────────────────────────────────

def scrape_category(category_key: str, friendly_name: str) -> list[dict]:
    """
    Scrapes listings + detail pages for one category.
    Checkpoints to data/raw/{category_key}.jsonl.
    """
    checkpoint_path = Path(config.RAW_DIR) / f"{category_key}.jsonl"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_path.exists():
        existing = _load_jsonl(checkpoint_path)
        if len(existing) >= config.MAX_LISTINGS_PER_CATEGORY:
            print(f"  [=] {friendly_name}: {len(existing)} items in checkpoint — skipping")
            return existing
        already_ids = {p["item_id"] for p in existing}
        print(f"  [~] {friendly_name}: resuming from {len(existing)} items")
    else:
        existing = []
        already_ids: set[str] = set()

    category_query = CATEGORY_QUERIES.get(category_key, friendly_name)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=config.HEADERS["User-Agent"],
            locale="en-US",
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        Stealth().apply_stealth_sync(page)

        # Warm up with homepage visit
        page.goto(BASE_URL + "/", wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(1500)

        listings = _scrape_listings(
            page, category_key, category_query, config.MAX_LISTINGS_PER_CATEGORY
        )
        new_listings = [l for l in listings if l["item_id"] not in already_ids]

        req_session = _make_requests_session(page)
        browser.close()

    enriched = list(existing)
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        for listing in tqdm(new_listings, desc=f"  Details [{friendly_name}]", unit="item"):
            product = scrape_product_detail(req_session, listing)
            enriched.append(product)
            f.write(json.dumps(product, ensure_ascii=False) + "\n")

    return enriched


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_price(raw) -> float:
    cleaned = re.sub(r"[^\d.]", "", str(raw))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def _parse_int(raw) -> int:
    cleaned = re.sub(r"[^\d]", "", str(raw))
    return int(cleaned) if cleaned else 0


def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items
