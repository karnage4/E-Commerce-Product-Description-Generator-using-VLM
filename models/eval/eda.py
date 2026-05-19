"""
Exploratory Data Analysis — generates figures for the Daraz dataset.

Output: models/results/eda/*.png

Usage:
    python -m models.eval.eda
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.shared.config import METADATA_FILE, RESULTS_DIR

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
SPLITS_DIR    = _PROJECT_ROOT / "data" / "data" / "processed" / "splits"
EDA_DIR       = RESULTS_DIR / "eda"

# ── Dark-theme style ──────────────────────────────────────────────────────────
DARK_BG = "#0d0d1a"
PANEL   = "#1a1a2e"
ACCENT  = "#00d4ff"
GOLD    = "#ffd700"
WARN    = "#ff7043"
TEXT    = "#eee"
SPINE   = "#555"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   PANEL,
    "axes.edgecolor":   SPINE,
    "axes.labelcolor":  TEXT,
    "xtick.color":      TEXT,
    "ytick.color":      TEXT,
    "text.color":       TEXT,
})


def _save(fig: plt.Figure, name: str) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    out = EDA_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {out}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_records() -> list[dict]:
    records = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def load_splits() -> dict[str, set[str]]:
    splits: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}.txt"
        if p.exists():
            splits[split] = set(p.read_text(encoding="utf-8").strip().splitlines())
    return splits


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_category_distribution(records: list[dict]) -> None:
    counts = Counter(r.get("category", "unknown") for r in records)
    cats   = sorted(counts, key=counts.get, reverse=True)
    vals   = [counts[c] for c in cats]
    labels = [c.replace("-", "\n") for c in cats]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, vals, color=ACCENT, edgecolor=SPINE)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title("Category Distribution", fontsize=14)
    ax.set_ylabel("Products")
    ax.set_xlabel("Category")
    plt.tight_layout()
    _save(fig, "category_distribution.png")


def fig_split_sizes(records: list[dict], splits: dict[str, set[str]]) -> None:
    ids_by_split = {s: ids for s, ids in splits.items()}
    sizes = {s: sum(1 for r in records if r.get("item_id") in ids)
             for s, ids in ids_by_split.items()}

    fig, ax = plt.subplots(figsize=(5, 4))
    colors  = [ACCENT, GOLD, WARN]
    bars    = ax.bar(list(sizes.keys()), list(sizes.values()),
                     color=colors[:len(sizes)], edgecolor=SPINE)
    for bar, v in zip(bars, sizes.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(v), ha="center", va="bottom", fontsize=11)
    ax.set_title("Train / Val / Test Split Sizes", fontsize=13)
    ax.set_ylabel("Products")
    plt.tight_layout()
    _save(fig, "split_sizes.png")


def fig_desc_word_count(records: list[dict]) -> None:
    lengths = [len(r.get("description", "").split())
               for r in records if r.get("description")]
    median  = float(np.median(lengths))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=40, color=ACCENT, edgecolor=SPINE, alpha=0.85)
    ax.axvline(median, color=GOLD, linestyle="--", linewidth=1.8,
               label=f"median = {median:.0f} words")
    ax.set_title("Description Word Count Distribution", fontsize=13)
    ax.set_xlabel("Words per description")
    ax.set_ylabel("Products")
    ax.legend(facecolor="#0f3460", labelcolor=TEXT, fontsize=10)
    plt.tight_layout()
    _save(fig, "desc_word_count.png")


def fig_images_per_product(records: list[dict]) -> None:
    counts  = Counter(len(r.get("images", [])) for r in records)
    keys    = sorted(counts)
    vals    = [counts[k] for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(k) for k in keys], vals, color=GOLD, edgecolor=SPINE)
    ax.set_title("Images per Product", fontsize=13)
    ax.set_xlabel("Image count")
    ax.set_ylabel("Products")
    plt.tight_layout()
    _save(fig, "images_per_product.png")


def fig_desc_length_by_category(records: list[dict]) -> None:
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in records:
        desc = r.get("description", "")
        if desc:
            by_cat[r.get("category", "unknown")].append(len(desc.split()))

    cats   = sorted(by_cat, key=lambda c: np.median(by_cat[c]), reverse=True)
    data   = [by_cat[c] for c in cats]
    labels = [c.replace("-", "\n") for c in cats]

    fig, ax = plt.subplots(figsize=(12, 5))
    bps = ax.boxplot(data, patch_artist=True, labels=labels,
                     medianprops={"color": GOLD, "linewidth": 2})
    for patch in bps["boxes"]:
        patch.set_facecolor(ACCENT)
        patch.set_alpha(0.5)
    for whisker in bps["whiskers"]:
        whisker.set_color(SPINE)
    for cap in bps["caps"]:
        cap.set_color(SPINE)
    ax.set_title("Description Word Count by Category", fontsize=13)
    ax.set_ylabel("Words")
    plt.tight_layout()
    _save(fig, "desc_length_by_category.png")


def fig_category_split_breakdown(records: list[dict], splits: dict[str, set[str]]) -> None:
    cats   = sorted({r.get("category", "unknown") for r in records})
    split_order = ["train", "val", "test"]
    colors_map  = {"train": ACCENT, "val": GOLD, "test": WARN}

    # count per category per split
    data: dict[str, list[int]] = {s: [] for s in split_order}
    for cat in cats:
        cat_ids = {r["item_id"] for r in records if r.get("category") == cat}
        for s in split_order:
            data[s].append(len(cat_ids & splits.get(s, set())))

    x      = np.arange(len(cats))
    width  = 0.25
    labels = [c.replace("-", "\n") for c in cats]

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, s in enumerate(split_order):
        bars = ax.bar(x + i * width, data[s], width,
                      label=s, color=colors_map[s], edgecolor=SPINE, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Category × Split Distribution", fontsize=13)
    ax.set_ylabel("Products")
    ax.legend(facecolor="#0f3460", labelcolor=TEXT, fontsize=10)
    plt.tight_layout()
    _save(fig, "category_split_breakdown.png")


# ── Summary printout ──────────────────────────────────────────────────────────

def print_summary(records: list[dict], splits: dict[str, set[str]]) -> None:
    print(f"\n  {'='*50}")
    print(f"  Dataset summary")
    print(f"  {'='*50}")
    print(f"  Total products : {len(records)}")
    for split, ids in splits.items():
        n = sum(1 for r in records if r.get("item_id") in ids)
        print(f"  {split:<6} split  : {n}")

    cats = Counter(r.get("category", "unknown") for r in records)
    print(f"\n  Categories ({len(cats)}):")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat:<32} {n:>4}")

    lengths = [len(r.get("description", "").split()) for r in records if r.get("description")]
    if lengths:
        print(f"\n  Description word counts:")
        print(f"    min={min(lengths)}  "
              f"median={int(np.median(lengths))}  "
              f"mean={np.mean(lengths):.1f}  "
              f"max={max(lengths)}")

    imgs = [len(r.get("images", [])) for r in records]
    print(f"\n  Images per product:")
    print(f"    min={min(imgs)}  median={int(np.median(imgs))}  "
          f"mean={np.mean(imgs):.1f}  max={max(imgs)}")
    print(f"  {'='*50}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("  Loading dataset...")
    records = load_records()
    splits  = load_splits()
    print(f"  {len(records)} records loaded")

    print_summary(records, splits)

    print("  Generating figures...")
    fig_category_distribution(records)
    fig_split_sizes(records, splits)
    fig_desc_word_count(records)
    fig_images_per_product(records)
    fig_desc_length_by_category(records)
    fig_category_split_breakdown(records, splits)

    print(f"\n  All EDA figures saved to {EDA_DIR}")


if __name__ == "__main__":
    main()
