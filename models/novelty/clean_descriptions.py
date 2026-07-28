"""
Experiment N18 — Strip markup residue from the reference descriptions.

Milestone 2 flagged this and never acted on it: many raw Daraz descriptions
carry a literal CSS fragment, `<pre>{ white-space: pre-wrap; }`, pasted in by
the seller's editor. The tokeniser treats it as ordinary words, it appears at
the start of dozens of records, and both the CNN+LSTM baseline and BLIP learned
to emit it.

It also actively distorts the evaluation. In the blank-image counterfactual on
the leakage-free split, the image-blind model opened with "pre white space pre
wrap" and *out-scored* the model that could see the product — because the
reference started the same way. A model is being rewarded for reproducing
editor noise.

This pass removes markup residue from the description field only. Metadata,
images and item ids are untouched, and the output is written to a new file so
listings_final.jsonl stays exactly as it was.

Run:
    python -m models.novelty.clean_descriptions
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import METADATA_FILE, RESULTS_DIR

OUT_FILE = METADATA_FILE.parent / "listings_final_textclean.jsonl"
REPORT = RESULTS_DIR / "novelty_text_clean.json"

# Ordered: CSS blocks first, then the bare declarations they leave behind.
PATTERNS = [
    (re.compile(r"<\s*pre\s*>", re.I), " "),
    (re.compile(r"\{[^{}]{0,120}\}"), " "),                    # { white-space: pre-wrap; }
    (re.compile(r"\b(?:white[-\s]*space|font[-\s]*size|line[-\s]*height|"
                r"text[-\s]*align|margin|padding|border|display)\s*:\s*[^;.\n]{0,40};?", re.I), " "),
    (re.compile(r"<[a-z/][^>]{0,60}>", re.I), " "),            # any stray html tag
    (re.compile(r"&(?:nbsp|amp|lt|gt|quot|#\d{2,5});", re.I), " "),
    (re.compile(r"\bpre[-\s]*wrap\b", re.I), " "),
    (re.compile(r"^\s*(?:pre|span|div|style)\b[:\s]*", re.I), " "),
]

WS = re.compile(r"\s+")


def clean(text: str) -> str:
    out = text
    for pat, rep in PATTERNS:
        out = pat.sub(rep, out)
    out = WS.sub(" ", out).strip()
    # Drop a leading orphan punctuation run left by the substitutions.
    out = re.sub(r"^[^\w(]+", "", out).strip()
    return out


def main():
    print("\n" + "=" * 78)
    print("N18 — CLEANING MARKUP RESIDUE FROM REFERENCES")
    print("=" * 78)

    changed = 0
    total = 0
    words_before = words_after = 0
    examples = []

    with open(METADATA_FILE, encoding="utf-8") as f, \
         open(OUT_FILE, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = r.get("description") or ""
            if d.strip():
                total += 1
                c = clean(d)
                words_before += len(d.split())
                words_after += len(c.split())
                if c != d.strip():
                    changed += 1
                    if len(examples) < 3 and len(d) - len(c) > 20:
                        examples.append((d[:130], c[:130]))
                r["description"] = c
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  descriptions processed : {total}")
    print(f"  descriptions modified  : {changed}  ({100*changed/max(total,1):.1f}%)")
    print(f"  mean words before/after: {words_before/max(total,1):.1f} -> {words_after/max(total,1):.1f}")

    for before, after in examples:
        print("\n  BEFORE: " + before)
        print("  AFTER : " + after)

    REPORT.write_text(json.dumps(
        {"total": total, "changed": changed,
         "mean_words_before": round(words_before / max(total, 1), 1),
         "mean_words_after": round(words_after / max(total, 1), 1),
         "output": str(OUT_FILE)}, indent=2), encoding="utf-8")
    print(f"\n  Saved → {OUT_FILE}")
    print("  (listings_final.jsonl is unchanged)")


if __name__ == "__main__":
    main()
