"""
Day-3 human study builder.

Produces a blinded rating sheet so the three of you can score generated
descriptions for visual grounding without knowing which system produced which
text. The resulting human scores replace the Gemma judge as ground truth for
validating Visual Attribute Density — which is the difference between
"we validated our metric with another LLM" and a claim a reviewer will accept.

Design notes that matter for the paper:
  - System identity is hidden and the display order is shuffled per item, so a
    rater cannot learn "the third one is always BLIP".
  - The product photograph is shown. Raters score how well the text describes
    *that image*, which is exactly the construct VAD claims to measure.
  - The reference description is deliberately NOT shown. We are measuring visual
    grounding, not similarity to a seller's marketing copy.
  - Three raters allow inter-annotator agreement (report Krippendorff's alpha or
    mean pairwise Spearman).

Outputs:
  models/results/human_study/rating_sheet.html   open in a browser, score, export
  models/results/human_study/key.json            system identities (do not open
                                                 until scoring is finished)

Run:
    python -m models.novelty.human_study --n 60
    python -m models.novelty.human_study --score ratings_shaheer.csv ratings_karam.csv
"""

import argparse
import base64
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import RESULTS_DIR, IMAGES_DIR, build_metadata_prompt
from models.novelty.visual_lexicon import load_records, density, VISUAL
from models.novelty.prompt_prefix_audit import strip_prefix

OUT_DIR = RESULTS_DIR / "human_study"
SEED = 7


def load_system(fname, key):
    out = {}
    p = RESULTS_DIR / fname
    if not p.exists():
        return out
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            d = json.loads(line)
            if d.get(key):
                out[str(d["item_id"])] = d[key]
    return out


def img_data_uri(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
            except Exception:
                continue
    return ""


def build(n):
    recs = load_records()
    systems = {
        "blind":     (load_system("novelty_blind_baseline_results.jsonl", "generated"), False),
        "blip":      (load_system("blip_results.jsonl", "generated"), True),
        "clip_gpt2": (load_system("clip_gpt2_results.jsonl", "generated"), False),
        "blip_stage1": (load_system("two_stage_results_blip.jsonl", "description_stage1"), False),
    }
    systems = {k: v for k, v in systems.items() if v[0]}
    ids = sorted(set.intersection(*[set(g) for g, _ in systems.values()]) & set(recs))
    rng = random.Random(SEED)
    rng.shuffle(ids)
    ids = ids[:n]
    print(f"  {len(ids)} items x {len(systems)} systems = {len(ids)*len(systems)} ratings per person")

    key, cards = {}, []
    for item_id in ids:
        rec = recs[item_id]
        entries = []
        for sysname, (gens, needs_strip) in systems.items():
            text = gens[item_id]
            if needs_strip:
                text, _, _ = strip_prefix(text, build_metadata_prompt(rec))
            text = text.strip() or "(empty)"
            entries.append((sysname, text))
        rng.shuffle(entries)
        uri = img_data_uri(rec)
        for slot, (sysname, text) in enumerate(entries):
            rid = f"{item_id}_{slot}"
            key[rid] = {"item_id": item_id, "system": sysname,
                        "vad": round(density(text, VISUAL), 2)}
        cards.append({"item_id": item_id, "image": uri,
                      "category": rec.get("category", ""),
                      "entries": [(f"{item_id}_{s}", t) for s, (_, t) in enumerate(entries)]})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    write_sheet(cards)
    print(f"\n  Sheet  → {OUT_DIR / 'rating_sheet.html'}")
    print(f"  Key    → {OUT_DIR / 'key.json'}  (do not open until scoring is done)")
    print("\n  Each rater: open the sheet, score every block, click Export, save as")
    print("  ratings_<name>.csv, then run:")
    print("    python -m models.novelty.human_study --score models/results/human_study/ratings_*.csv")


SHEET_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0;
background:#f4f6f8;color:#1a2230;line-height:1.5}
.wrap{max-width:900px;margin:0 auto;padding:24px 20px 80px}
h1{font-size:24px;margin:0 0 6px}
.intro{background:#fff;border:1px solid #d5dbe1;border-radius:8px;padding:18px 22px;margin-bottom:24px}
.intro ul{margin:8px 0 0;padding-left:20px}
.card{background:#fff;border:1px solid #d5dbe1;border-radius:8px;padding:20px;margin-bottom:18px}
.hd{display:flex;gap:18px;align-items:flex-start;margin-bottom:14px}
.hd img{width:190px;height:190px;object-fit:contain;background:#eef1f4;border-radius:6px;flex:none}
.meta{font-size:12px;color:#5d6875;text-transform:uppercase;letter-spacing:.08em}
.entry{border-top:1px solid #e6eaee;padding:14px 0}
.txt{font-size:14px;margin-bottom:10px;white-space:pre-wrap}
.scale{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.scale label{font-size:13px;border:1px solid #c8d0d8;border-radius:5px;padding:5px 11px;cursor:pointer;background:#fafbfc}
.scale input{margin-right:5px}
.scale label:has(input:checked){background:#0c6d76;color:#fff;border-color:#0c6d76}
.bar{position:fixed;bottom:0;left:0;right:0;background:#0f1620;color:#fff;padding:12px 20px;
display:flex;gap:16px;align-items:center;justify-content:center;font-size:14px}
button{background:#0c6d76;color:#fff;border:0;border-radius:6px;padding:9px 20px;font-size:14px;cursor:pointer}
@media (prefers-color-scheme:dark){body{background:#0c1015;color:#e8eef4}
.card,.intro{background:#161d26;border-color:#26303b}.hd img{background:#0c1015}
.scale label{background:#1c242d;border-color:#2f3b47;color:#e8eef4}.entry{border-color:#26303b}}
"""

SHEET_JS = """
function pct(){const t=document.querySelectorAll('.entry').length;
const d=document.querySelectorAll('.entry input:checked').length;
document.getElementById('prog').textContent=d+' / '+t+' scored';}
document.addEventListener('change',pct);
function exportCSV(){
 let rows=[['rating_id','visual_grounding']];
 document.querySelectorAll('.entry').forEach(e=>{
  const id=e.dataset.rid;const c=e.querySelector('input:checked');
  rows.push([id,c?c.value:'']);});
 const csv=rows.map(r=>r.join(',')).join('\\n');
 const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));
 a.download='ratings.csv';a.click();}
window.addEventListener('load',pct);
"""


def write_sheet(cards):
    h = ['<!doctype html><html><head><meta charset="utf-8">',
         '<meta name="viewport" content="width=device-width,initial-scale=1">',
         '<title>Visual grounding rating sheet</title>',
         f"<style>{SHEET_CSS}</style></head><body><div class='wrap'>",
         "<h1>Visual grounding rating sheet</h1>",
         "<div class='intro'><strong>Question for every block of text:</strong> how much of "
         "this text describes what you can actually see in the photograph?<ul>"
         "<li><b>1</b> — nothing visual; it restates specs, brand or marketing copy</li>"
         "<li><b>2</b> — one vague visual word (\"nice design\")</li>"
         "<li><b>3</b> — some correct visual detail (a colour, a shape)</li>"
         "<li><b>4</b> — several correct visual details</li>"
         "<li><b>5</b> — richly and accurately describes the object's appearance</li></ul>"
         "<p>Score <em>visual grounding only</em>. Ignore fluency, grammar and repetition. "
         "If a visual claim is wrong (says black, item is red), score it low. "
         "Do not discuss with the other raters until everyone has finished.</p></div>"]

    for c in cards:
        h.append("<div class='card'><div class='hd'>")
        if c["image"]:
            h.append(f"<img src='{c['image']}' alt='product'>")
        h.append(f"<div><div class='meta'>{c['category']} &middot; item {c['item_id']}</div></div></div>")
        for rid, text in c["entries"]:
            h.append(f"<div class='entry' data-rid='{rid}'><div class='txt'>{esc(text)}</div><div class='scale'>")
            for v in range(1, 6):
                h.append(f"<label><input type='radio' name='{rid}' value='{v}'>{v}</label>")
            h.append("</div></div>")
        h.append("</div>")

    h.append("</div><div class='bar'><span id='prog'></span>"
             "<button onclick='exportCSV()'>Export CSV</button></div>")
    h.append(f"<script>{SHEET_JS}</script></body></html>")
    (OUT_DIR / "rating_sheet.html").write_text("\n".join(h), encoding="utf-8")


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def score(files):
    import csv as csvmod
    from scipy.stats import spearmanr

    key = json.loads((OUT_DIR / "key.json").read_text(encoding="utf-8"))
    raters = {}
    for fp in files:
        name = Path(fp).stem
        rows = {}
        with open(fp, encoding="utf-8") as f:
            for r in csvmod.DictReader(f):
                if r.get("visual_grounding"):
                    rows[r["rating_id"]] = float(r["visual_grounding"])
        raters[name] = rows
        print(f"  {name}: {len(rows)} ratings")

    common = sorted(set.intersection(*[set(v) for v in raters.values()]))
    print(f"\n  {len(common)} items rated by all {len(raters)} raters")

    print("\n  [Inter-annotator agreement, pairwise Spearman]")
    names = list(raters)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = [raters[names[i]][k] for k in common]
            b = [raters[names[j]][k] for k in common]
            rho, p = spearmanr(a, b)
            print(f"    {names[i]} vs {names[j]}: rho = {rho:+.3f} (p={p:.3g})")

    mean = {k: sum(raters[n][k] for n in names) / len(names) for k in common}
    vad = [key[k]["vad"] for k in common]
    human = [mean[k] for k in common]
    rho, p = spearmanr(vad, human)
    print(f"\n  *** Visual Attribute Density vs HUMAN grounding: rho = {rho:+.3f} (p={p:.3g}) ***")

    print("\n  [Mean human grounding score by system]")
    bysys = {}
    for k in common:
        bysys.setdefault(key[k]["system"], []).append(mean[k])
    for s, v in sorted(bysys.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        print(f"    {s:<14} {sum(v)/len(v):.2f}  (n={len(v)})")

    out = {"n_items": len(common), "raters": names,
           "vad_vs_human_spearman": round(float(rho), 3), "p_value": float(p),
           "mean_by_system": {s: round(sum(v) / len(v), 2) for s, v in bysys.items()}}
    (OUT_DIR / "human_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_DIR / 'human_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--score", nargs="+")
    a = ap.parse_args()
    if a.score:
        score(a.score)
    else:
        build(a.n)
