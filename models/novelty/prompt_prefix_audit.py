"""
Experiment N9 — Prompt-Prefix Leakage in the Scored Output.

Additive experiment, CPU only. Reads existing result files; changes nothing.

Finding
-------
BLIP is invoked as

    ids = model.generate(**processor(images=img, text=prompt, ...))
    text = processor.decode(ids[0], skip_special_tokens=True)

For BlipForConditionalGeneration the metadata prompt is the *decoder prefix*, so
the returned id sequence contains the prompt followed by the continuation, and
`decode` returns both. The saved "generated description" therefore begins with a
verbatim copy of the metadata prompt, and that copy is scored against the
reference.

CLIP-GPT2 is invoked through `inputs_embeds`, whose `generate` returns only the
newly sampled tokens. Its saved output contains no prompt prefix.

The two models are therefore not being compared on the same quantity: BLIP is
scored on (prompt + description), CLIP-GPT2 on (description). Because ~20% of
the average reference is itself recoverable from the metadata (see N1), the
prefix is not neutral padding — it is text that scores.

This script measures the prefix and re-scores BLIP on the continuation alone.

Run:
    python -m models.novelty.prompt_prefix_audit
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)
from models.shared.metrics import compute_all_metrics

OUT_JSON = RESULTS_DIR / "novelty_prompt_prefix_audit.json"


def norm_words(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip().split()


def load_records():
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
    recs = {}
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids:
                recs[r["item_id"]] = r
    return recs


def load_system(path, key):
    out = {}
    if not Path(path).exists():
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            if d.get(key):
                out[str(d["item_id"])] = d[key]
    return out


def strip_prefix(gen: str, prompt: str) -> tuple[str, int, int]:
    """Remove the longest verbatim word-level prefix of `prompt` from `gen`.

    Returns (continuation, n_prefix_words, n_total_words).
    """
    pw, gw = norm_words(prompt), norm_words(gen)
    k = 0
    while k < min(len(pw), len(gw)) and pw[k] == gw[k]:
        k += 1
    return " ".join(gw[k:]), k, len(gw)


def main():
    print("\n" + "=" * 78)
    print("N9 — PROMPT-PREFIX LEAKAGE IN THE SCORED OUTPUT")
    print("=" * 78)

    recs = load_records()
    blip = load_system(RESULTS_DIR / "blip_results.jsonl", "generated")
    clip = load_system(RESULTS_DIR / "clip_gpt2_results.jsonl", "generated")

    report = {}

    for name, gens in (("BLIP fine-tuned", blip), ("CLIP-GPT2 fine-tuned", clip)):
        if not gens:
            continue
        ids = [i for i in gens if i in recs and recs[i].get("description", "").strip()]
        refs = [recs[i]["description"].strip() for i in ids]
        raw = [gens[i] for i in ids]
        prompts = [build_metadata_prompt(recs[i]) for i in ids]

        stripped, kfrac, nprefix = [], [], 0
        for g, p in zip(raw, prompts):
            cont, k, tot = strip_prefix(g, p)
            stripped.append(cont)
            kfrac.append(k / max(tot, 1))
            if k >= 5:
                nprefix += 1

        as_is = compute_all_metrics(raw, refs)
        # guard against empty continuations
        cont = [c if c.strip() else "." for c in stripped]
        corrected = compute_all_metrics(cont, refs)

        entry = {
            "n": len(ids),
            "items_starting_with_prompt_prefix_pct": round(100 * nprefix / len(ids), 1),
            "mean_pct_of_output_that_is_prompt_prefix": round(100 * sum(kfrac) / len(kfrac), 1),
            "mean_words_as_scored": round(sum(len(g.split()) for g in raw) / len(raw), 1),
            "mean_words_after_stripping": round(sum(len(c.split()) for c in stripped) / len(stripped), 1),
            "as_scored_in_milestone2": as_is,
            "prompt_prefix_removed": corrected,
        }
        report[name] = entry

        print(f"\n  [{name}]  n={entry['n']}")
        print(f"    outputs beginning with a verbatim prompt prefix : "
              f"{entry['items_starting_with_prompt_prefix_pct']}%")
        print(f"    mean share of the scored text that IS the prompt: "
              f"{entry['mean_pct_of_output_that_is_prompt_prefix']}%")
        print(f"    mean length as scored / after stripping         : "
              f"{entry['mean_words_as_scored']} / {entry['mean_words_after_stripping']} words")
        print(f"    {'metric':<10}{'as scored':>12}{'prefix removed':>17}{'delta':>10}")
        for m in ("BLEU-1", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"):
            d = corrected[m] - as_is[m]
            print(f"    {m:<10}{as_is[m]:>12}{corrected[m]:>17}{d:>+10.2f}")

    if "BLIP fine-tuned" in report and "CLIP-GPT2 fine-tuned" in report:
        b = report["BLIP fine-tuned"]
        c = report["CLIP-GPT2 fine-tuned"]
        print("\n  [Like-for-like comparison: continuation only]")
        print(f"    {'metric':<10}{'BLIP':>10}{'CLIP-GPT2':>12}{'winner':>14}")
        flips = {}
        for m in ("BLEU-1", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"):
            bv = b["prompt_prefix_removed"][m]
            cv = c["prompt_prefix_removed"][m]
            old_w = ("BLIP" if b["as_scored_in_milestone2"][m] > c["as_scored_in_milestone2"][m]
                     else "CLIP-GPT2")
            new_w = "BLIP" if bv > cv else "CLIP-GPT2"
            flips[m] = {"old_winner": old_w, "new_winner": new_w, "flipped": old_w != new_w}
            mark = "  <- FLIPPED" if old_w != new_w else ""
            print(f"    {m:<10}{bv:>10}{cv:>12}{new_w:>14}{mark}")
        report["ranking_changes"] = flips

    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
