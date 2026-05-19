# VLM Description Generator — Improvement Plan

**Audience:** project team
**Status:** proposal for review before implementation
**Author note:** this is a verification of Claude's suggestions against our actual code, plus a proper design for the Gemini-augmentation idea.

---

## TL;DR

| Suggestion | Verdict | Priority |
|---|---|---|
| (1a) Price-fix regex | Correct, useful band-aid | P3 — do it, takes 5 min |
| (1b) Repetition penalties | Correct | P1 — do this on next eval run |
| (1c) Strip metadata echo | Symptom-treatment only | Skip — Phase 2 fixes root cause |
| (2) BLIP loss-masking | **Real bug** in BLIP training. Claude's snippet has bugs too. | P1 — fix both, retrain |
| (2) CLIP-GPT2 loss-masking | **Already done correctly in our code.** Don't change. | Skip |
| (3) Multi-view averaging | Modest gain, can hurt if views differ a lot | P3 — try at inference only first |
| (4) BLIP-2 + LoRA | Big architectural upgrade, feasible on T4 | P2 — do *after* Phase 2 |
| (Teammate) Gemini-augmented features | Strongest idea — addresses the root cause | **P1 — make this the centerpiece** |

The single highest-leverage move is the teammate's Gemini idea, properly framed as **caption augmentation**. The model isn't learning visual features because the *targets* (Daraz descriptions) barely mention them. Fix the targets before fixing the architecture.

---

## What I verified in our code

I read `models/blip/train_colab.py`, `models/clip_gpt2/train_colab.py`, `models/blip/evaluate_colab.py`, and `models/api_baseline/gemini_baseline.py`. Specifics:

**BLIP training has the loss-masking bug.** In `DarazBlipDataset.__getitem__` (lines 142–165 of `train_colab.py`), the metadata and description are concatenated into a single string, tokenized once, and `labels = input_ids.clone()` with only padding masked to -100. The model is trained to predict the metadata tokens from themselves — wasted capacity, and a real reason for "metadata echoing" at inference. Claude is right that this is the biggest single training fix for BLIP.

**CLIP-GPT2 training does NOT have this bug.** In `ClipGPT2Model.forward` (lines 138–185 of `clip_gpt2/train_colab.py`), the prompt portion is already masked with `prefix_ignore = torch.full((B, P + T_text), -100, ...)`. Loss is correctly computed only on description tokens. **Do not apply Claude's "Improvement 2" to CLIP-GPT2** — it would be a no-op at best, a regression at worst.

**Both models only use the first image.** Confirmed at `train_colab.py` line 130–138 (BLIP) and line 279–289 (CLIP-GPT2): a `for rel in rec["images"]` loop that breaks on the first successful load.

**Gemini integration already exists.** `models/api_baseline/gemini_baseline.py` already calls Gemini 1.5 Flash on the test set as a baseline. We can reuse that plumbing for caption augmentation on the training set.

---

## Phase 1 — Quick wins (1 day, no retraining)

These are safe, cheap changes you can run on the *existing* checkpoints.

### 1.1 Generation hyperparameters

In `models/blip/evaluate_colab.py` lines 150–154 and `models/clip_gpt2/model.py` line 195:

```python
output_ids = model.generate(
    **inputs,
    max_new_tokens=150,
    num_beams=4,
    no_repeat_ngram_size=4,      # was 3
    repetition_penalty=1.3,       # new
    length_penalty=1.1,           # new — gentler than 1.2 to avoid empty rambling
    early_stopping=True,
)
```

Expected gain: cuts visible repetition loops, may move BLEU/ROUGE 1–3 points. Free.

### 1.2 Price hallucination patch

Add to `evaluate_colab.py` after generation:

```python
import re
def fix_price_hallucination(text: str, rec: dict) -> str:
    if not rec.get("price_pkr"):
        return text
    correct = f"PKR {rec['price_pkr']:.0f}"
    return re.sub(r'(?:PKR|Rs\.?|₨)\s*[\d,]+', correct, text, flags=re.IGNORECASE)
```

This is a band-aid. It will be replaced by Phase 2.

### 1.3 Skip the metadata-echo strip

The "strip metadata echo" suggestion is treating a symptom of the BLIP loss bug. Once we apply 2.1 below and retrain, the echo largely disappears. Don't add fragile string-matching post-processing — fix the cause.

---

## Phase 2 — Fix the real bugs and retrain BLIP (~2 hours training on T4)

### 2.1 Correct BLIP loss masking — corrected code

Claude's suggested fix has two bugs: (a) it doesn't account for the CLS/BOS token the BLIP tokenizer prepends, so `labels[:n_meta] = -100` is off by one; (b) it doesn't mask padding tokens *within* the description portion.

Replacement for `DarazBlipDataset.__getitem__` (the text-handling part):

```python
metadata    = build_metadata_prompt(rec)
description = rec["description"].strip()
combined    = f"{metadata}. {description}"

# Tokenize the FULL combined sequence once (this is what the decoder sees)
encoding = self.processor(
    images=image,
    text=combined,
    padding="max_length",
    truncation=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors="pt",
)
input_ids      = encoding["input_ids"].squeeze(0)
attention_mask = encoding["attention_mask"].squeeze(0)
pixel_values   = encoding["pixel_values"].squeeze(0)

# Find where metadata ends in the tokenized sequence.
# Tokenize "metadata." alone WITHOUT special tokens to count its tokens,
# then add 1 for the CLS/BOS that the full encoding starts with.
meta_only_ids = self.processor.tokenizer(
    metadata + ".",
    add_special_tokens=False,
).input_ids
n_meta_tokens = len(meta_only_ids) + 1   # +1 for [CLS]/[BOS]
n_meta_tokens = min(n_meta_tokens, MAX_SEQ_LENGTH)

# Labels: ignore metadata positions AND padding
labels = input_ids.clone()
labels[:n_meta_tokens] = -100
labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

return {
    "pixel_values":   pixel_values,
    "input_ids":      input_ids,
    "attention_mask": attention_mask,
    "labels":         labels,
}
```

This forces BLIP to compute loss only on description tokens, conditioned on image + metadata — which matches what we do at inference.

**Sanity check before full training run:** print `processor.tokenizer.decode(input_ids[n_meta_tokens:])` for one sample — it should start with the description, not mid-sentence in metadata.

### 2.2 Do NOT touch CLIP-GPT2 training loss

Already correctly masked. Skip Claude's Improvement 2 for this model entirely.

### 2.3 Re-run evaluation with Phase 1 generation params + Phase 2 BLIP retrain

Expected combined gain on BLIP: target +5–10 BLEU-4, fewer hallucinated metadata echoes. CLIP-GPT2 only benefits from Phase 1 (generation params).

---

## Phase 3 — The teammate's idea, properly designed (3–5 days)

This is the heart of the project. The diagnosis is correct: **our models default to regurgitating metadata because the Daraz descriptions in our training set rarely mention visual features**, so the image signal has nothing to teach the decoder. Both models are doing the rational thing given their data.

There are two ways to attack this. We should pick one (or do A first, B later).

### Option A — Caption augmentation via Gemini (recommended start)

Use Gemini 1.5 Flash to **rewrite our training descriptions** so they describe what's actually in the image, then fine-tune BLIP/CLIP-GPT2 on the augmented descriptions. This is the same technique BLIP-2 used to bootstrap from noisy web data (CapFilt) and the same idea behind ShareGPT4V.

**Pipeline:**

1. **Pick the augmentation prompt.** For each training record, send Gemini: `image + metadata + original description` and ask it to rewrite the description in 60–120 words. The prompt should explicitly ask for: visible product features (color, material texture, design elements, ports, attachments, branding visible on the product), retained factual metadata (brand, price, dimensions if given), and the same Daraz-style tone. Critical guard: **"Do not invent specifications that are not present in the image or metadata."**
2. **Run on training split only.** ~5,000 train items × 1 call ≈ free tier finishes in ~6 hrs at 15 req/min, or ~$1–2 on paid tier (Flash is cheap). Cache results to JSONL keyed by `item_id` so we never re-pay.
3. **Build a new metadata file** `listings_augmented.jsonl` with `description_augmented` field.
4. **Train both models on `description_augmented` instead of `description`.** Reuse the existing Phase 2 loss-masking logic.
5. **Evaluate against the original Daraz `description` test set** (don't touch test). If the augmented-trained model scores worse on BLEU but humans like it better — that's the expected trade-off and matches the Stanford paper's qualitative-vs-quantitative gap. Plan to do small-scale human eval (50 samples, 3 raters).

**Why this works:** the model sees lots of (image → rich-feature-description) pairs, so the visual encoder gradient finally has a reason to learn MagSafe-cutout-shaped features instead of just learning "ignore image, copy metadata."

**Risk to call out:** Gemini will hallucinate too, and we'll be training on those hallucinations. Mitigations: (i) keep the original description as a constraint in the prompt ("preserve all factual claims from this original"); (ii) spot-check 100 augmented samples before training on all 5,000; (iii) include a category-aware filter that rejects augmented descriptions whose length grows >5x or shrinks below 30 words.

### Option B — Auxiliary visual-feature head (more research-y)

Use Gemini to extract **structured feature lists per category** (e.g., for phone cases: `[has_magsafe, material, has_camera_cutout, color, finish]`), then add a multi-label classification head on top of the CLIP/BLIP vision encoder. Total loss = `λ * description_loss + (1-λ) * feature_loss`. The classifier forces the encoder to learn category-relevant visual features even if the description doesn't mention them.

This is more work than Option A and harder to scope (need per-category schemas), but it's a more interpretable and publishable result — closer to "we taught it to extract MagSafe-ness." If the team wants a more novel contribution rather than a known technique, this is the move.

**Recommendation:** start with A. If time permits after evaluation, layer B on top.

---

## Phase 4 — Architecture upgrade: BLIP-2 + LoRA (~3 hours training)

**Do this after Phase 3, not before.** Better targets matter more than a better architecture; if we upgrade to BLIP-2 first we'll just have a stronger model overfitting to the same metadata-regurgitation pattern.

Stanford's paper actually tried BLIP-2 and abandoned it because of compute. With 8-bit + LoRA on a T4, it becomes feasible. Claude's snippet is largely correct, but a few notes:

- **VRAM is tight.** OPT-2.7B in 8-bit ≈ 3 GB; activations + Q-Former + LoRA grads bring it to ~10–11 GB. Use `BATCH_SIZE=1` with `GRAD_ACCUM=8`. Don't try `BATCH_SIZE=2`.
- **Prompt format is different.** BLIP-2 expects an OPT-style prompt: `"Question: describe this product. Context: {metadata}. Answer:"`. Don't reuse our BLIP prompt template verbatim.
- **Gradient checkpointing must be enabled** (`model.gradient_checkpointing_enable()`) before `prepare_model_for_kbit_training`.
- **LoRA targets `q_proj, v_proj` is a reasonable default.** Stanford reported rank=256, but on our smaller dataset (~5k) rank=16 is plenty and trains faster.

---

## Phase 5 — Multi-view averaging (defer)

Try at **inference only** first, on top of the Phase 2/3 model, before adding it to training. Reason: averaging CLIP CLS vectors across multiple product views (front, back, packaging, lifestyle) often blurs features. If averaging doesn't beat first-image-only at inference, don't bother training with it. If it does beat, then add it to training.

---

## Recommended sequence

| Phase | What | Time | Owner | Gate |
|---|---|---|---|---|
| 1 | Generation hyperparams + price regex | 1 day | anyone | Just rerun eval, no retrain |
| 2 | Fix BLIP loss masking, retrain BLIP | 1 day code + 2 hr GPU | whoever owns BLIP | Sanity-check decoded labels first |
| 3A | Gemini caption augmentation | 3–5 days | one person on Gemini, one on training | Spot-check 100 augmented samples before full retrain |
| 4 | BLIP-2 + LoRA on augmented data | 1 day code + 3 hr GPU | whoever owns BLIP | Only after Phase 3 numbers are in |
| 5 | Multi-view at inference | half day | anyone | Skip if no gain |
| 3B | Auxiliary feature head (optional) | 1 week | research-leaning person | Only if time allows after Phase 4 |

---

## Decisions the team needs to make

1. **Gemini budget.** Free tier (15/min, ~6 hrs to augment 5k items, no cost) vs paid tier (~$1–2, runs in 30 min). Recommendation: free tier, run overnight.
2. **Which model do we showcase?** If the deliverable is "best numbers", BLIP-2 + augmented data is the answer. If it's "novel contribution", Option B (auxiliary feature head) on BLIP is more interesting.
3. **Human evaluation setup.** With augmentation, BLEU may *drop* while quality goes up. Do we have time for a 50-sample, 3-rater human eval?
4. **Test set policy.** We must not let Gemini touch test descriptions. Confirm Phase 3.1 only runs over `train.txt` IDs.
5. **Keep CLIP-GPT2 in the comparison?** Stanford reported it as the worst model; with augmented data it might still be the worst. Argument for keeping it: clean ablation. Argument for dropping it: focus.

---

## What to push back on from Claude's original suggestions

- "Improvement 2 is the single biggest improvement without changing architecture." → True for BLIP, false for CLIP-GPT2 (already done). Don't apply both.
- The strip-metadata-echo function → unnecessary once 2.1 is in. Don't add fragile post-processing.
- Multi-view averaging as a top-3 priority → it's a nice-to-have, not a difference-maker. Defer.
- Claude did not flag the `n_meta` off-by-one (CLS/BOS) or the in-description padding mask. Use the corrected snippet in 2.1.
