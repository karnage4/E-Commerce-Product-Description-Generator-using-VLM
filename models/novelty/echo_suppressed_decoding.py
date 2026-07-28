"""
Experiment N4 — Echo-Suppressed Decoding (ESD).

Additive experiment: a decoding-time intervention. No retraining, no weight
changes, no modification to any existing script. The original checkpoints are
loaded read-only and the baseline (lambda = 0) reproduces the current pipeline
exactly.

Motivation
----------
N1 showed that BLIP spends ~57% of its emitted content tokens re-stating its own
metadata prompt, and that it recovers 100% of the metadata-derivable reference
tokens but only ~4% of the visual/novel ones. N3 showed that with the full
metadata prompt attached, blanking the image barely changes the output.

Both point at the same failure: the metadata prompt is a cheaper way to score
well than the photograph is, so the decoder leans on it.

ESD attacks this at decode time. During generation we subtract a fixed penalty
lambda from the logits of every *content* token that appears in the metadata
prompt. The model keeps the prompt as conditioning but is discouraged from
re-emitting it verbatim, so probability mass shifts toward whatever else it
knows — which, for a vision-language model, is the image.

This costs nothing: no extra parameters, no extra forward passes, no retraining.

Reported per lambda:
    ROUGE-L                 quality against the (metadata-contaminated) reference
    Residual ROUGE-L        quality against the reference with metadata removed
    Visual Residual Recall  % of the reference's non-metadata tokens recovered
    Echo Rate               % of emitted content tokens copied from the prompt
    VSS                     100 - ROUGE-L(gen_real, gen_blank); image dependence

Run:
    python -m models.novelty.echo_suppressed_decoding
    python -m models.novelty.echo_suppressed_decoding --max-samples 40 --lambdas 0 2 4
"""

import argparse
import json
import re
import sys
from pathlib import Path

import nltk
import torch
from PIL import Image
from nltk.stem.porter import PorterStemmer
from rouge_score import rouge_scorer as rouge_lib
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)
from models.shared.metrics import compute_all_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
OUT_JSON = RESULTS_DIR / "novelty_echo_suppressed_decoding.json"
OUT_JSONL = RESULTS_DIR / "novelty_esd_generations.jsonl"
BLANK = Image.new("RGB", (224, 224), (128, 128, 128))

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

STOP = set(stopwords.words("english"))
STEM = PorterStemmer()
WORD_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text):
    return [STEM.stem(t) for t in WORD_RE.findall(text.lower())
            if t not in STOP and len(t) > 1]


# ── The intervention ──────────────────────────────────────────────────────────

class EchoSuppressionProcessor(LogitsProcessor):
    """Subtract `lam` from the logits of content tokens present in the prompt.

    Only tokens that are alphabetic, longer than two characters and not English
    stopwords are penalised, so grammatical glue ("the", "a", "with") is left
    alone and fluency is preserved.
    """

    def __init__(self, penalised_ids: torch.Tensor, lam: float):
        self.penalised_ids = penalised_ids
        self.lam = lam

    def __call__(self, input_ids, scores):
        if self.lam == 0 or self.penalised_ids.numel() == 0:
            return scores
        scores[:, self.penalised_ids] -= self.lam
        return scores


def build_penalised_ids(tokenizer, prompt: str) -> torch.Tensor:
    """Token ids of the prompt's content words (plus their leading-space forms)."""
    ids = set()
    for word in WORD_RE.findall(prompt.lower()):
        if word in STOP or len(word) <= 2 or word.isdigit():
            continue
        for variant in (word, " " + word, word.capitalize(), " " + word.capitalize()):
            enc = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if len(enc) == 1:
                ids.add(enc[0])
            elif enc:
                ids.add(enc[0])          # penalise the leading sub-word piece
    return torch.tensor(sorted(ids), dtype=torch.long, device=DEVICE)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_test_records(max_samples):
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
    recs = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids and r.get("images") and r.get("description", "").strip():
                recs.append(r)
    return recs[:max_samples]


def load_image(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return Image.new("RGB", (224, 224), (255, 255, 255))


def strip_prompt_tokens(text, prompt_set):
    return " ".join(t for t in WORD_RE.findall(text.lower())
                    if STEM.stem(t) not in prompt_set)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=135)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--vss-samples", type=int, default=60,
                    help="how many items also get a blank-image pass for VSS")
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("N4 — ECHO-SUPPRESSED DECODING")
    print("=" * 78)

    ckpt = RESULTS_DIR.parent / "checkpoints" / "blip" / "best_model"
    print(f"  Loading BLIP from {ckpt}  (device={DEVICE})")
    proc = BlipProcessor.from_pretrained(str(ckpt))
    model = BlipForConditionalGeneration.from_pretrained(
        str(ckpt), torch_dtype=DTYPE).to(DEVICE).eval()
    tok = proc.tokenizer

    records = load_test_records(args.max_samples)
    images = [load_image(r) for r in records]
    prompts = [build_metadata_prompt(r) for r in records]
    refs = [r["description"].strip() for r in records]
    print(f"  {len(records)} test items | lambdas = {args.lambdas}")

    scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    penalised = [build_penalised_ids(tok, p) for p in prompts]

    @torch.no_grad()
    def gen(image, prompt, pen_ids, lam):
        inputs = proc(images=image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        lp = LogitsProcessorList([EchoSuppressionProcessor(pen_ids, lam)]) if lam else None
        with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
            ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, num_beams=4,
                early_stopping=True, no_repeat_ngram_size=3, logits_processor=lp)
        return proc.decode(ids[0], skip_special_tokens=True).strip()

    results = {}
    log = open(OUT_JSONL, "w", encoding="utf-8")

    for lam in args.lambdas:
        gens, gens_blank = [], []
        for i, rec in enumerate(tqdm(records, desc=f"  lambda={lam}", unit="item")):
            out = gen(images[i], prompts[i], penalised[i], lam)
            gens.append(out)
            if i < args.vss_samples:
                gens_blank.append(gen(BLANK, prompts[i], penalised[i], lam))
            log.write(json.dumps({"lambda": lam, "item_id": rec["item_id"],
                                  "generated": out}, ensure_ascii=False) + "\n")
        log.flush()

        # standard metrics
        cell = compute_all_metrics(gens, refs)

        # decomposed metrics
        vr_num = vr_den = echo_num = echo_den = 0
        res = []
        for g, ref, p in zip(gens, refs, prompts):
            p_set = set(content_tokens(p))
            g_set = set(content_tokens(g))
            r_toks = content_tokens(ref)
            v_set = {t for t in r_toks if t not in p_set}
            vr_num += len(g_set & v_set); vr_den += len(v_set)
            g_all = content_tokens(g)
            echo_num += sum(1 for t in g_all if t in p_set); echo_den += len(g_all)
            r_res = strip_prompt_tokens(ref, p_set)
            if r_res.strip():
                res.append(scorer.score(r_res, strip_prompt_tokens(g, p_set))["rougeL"].fmeasure)

        # Visual Attribute Density — the reference-free metric validated in N8.
        # This is the metric ESD should actually be judged on: N8 shows the
        # references contain too little visual content for a reference-based
        # metric to register a change in visual grounding.
        from models.novelty.visual_lexicon import density, VISUAL, COMMERCIAL
        cell["Visual Attribute Density"] = round(
            sum(density(g, VISUAL) for g in gens) / len(gens), 2)
        cell["Commercial Density"] = round(
            sum(density(g, COMMERCIAL) for g in gens) / len(gens), 2)

        cell["Residual ROUGE-L"] = round(100 * sum(res) / len(res), 2)
        cell["Visual Residual Recall"] = round(100 * vr_num / max(vr_den, 1), 2)
        cell["Echo Rate"] = round(100 * echo_num / max(echo_den, 1), 2)

        sims = [scorer.score(a, b)["rougeL"].fmeasure
                for a, b in zip(gens[:len(gens_blank)], gens_blank)]
        cell["VSS"] = round(100 - 100 * sum(sims) / len(sims), 2) if sims else None
        cell["mean_gen_words"] = round(sum(len(g.split()) for g in gens) / len(gens), 1)

        results[str(lam)] = cell
        print(f"\n  [lambda={lam}]")
        for k, v in cell.items():
            print(f"    {k:<26} {v}")
        OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    log.close()

    print("\n" + "-" * 78)
    print(f"  {'lambda':<8}{'ROUGE-L':>10}{'Resid-RL':>10}{'VisResRec':>11}{'EchoRate':>10}{'VSS':>8}{'CIDEr':>8}")
    print("-" * 78)
    for lam, c in results.items():
        print(f"  {lam:<8}{c['ROUGE-L']:>10}{c['Residual ROUGE-L']:>10}"
              f"{c['Visual Residual Recall']:>11}{c['Echo Rate']:>10}"
              f"{c['VSS'] if c['VSS'] is not None else '-':>8}{c['CIDEr']:>8}")
    print("-" * 78)
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
