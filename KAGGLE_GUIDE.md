# Same-Day Execution Guide

End-to-end: from raw data → augmented descriptions → Kaggle BLIP retrain → evaluation.

Estimated total wall time: **~3 hours** (≈45 min augmenting locally + ≈10 min upload + ≈60 min training on Kaggle T4 + ≈15 min evaluation + buffer).

---

## Step 0 — One-time setup (5 minutes)

```bash
# In the project root
cp .env.example .env
# Open .env and paste your NIM key:
#   NVIDIA_API_KEY=nvapi-...

pip install -r requirements.txt
```

Get a NIM key at <https://build.nvidia.com/> if you don't have one.

---

## Step 1 — Sanity-check the augmentation prompt (5 minutes)

Don't blow ~$0.50 (or 75 minutes) on 1,100 calls until you've eyeballed 5.

```bash
python -m models.augment.test_augment
```

Read the 5 outputs. Look for:

- Does it describe what's actually in the image (color, material, shape, ports, branding)?
- Does it avoid inventing specs not in metadata?
- Does it skip price/ratings/shipping?
- Does it sound like a Daraz description (not "I can see in this image…")?

If quality looks weak, edit `AUGMENTATION_PROMPT` in `models/augment/augment_descriptions.py` and re-run the test until it's good. The test uses a fixed seed (42), so the same 5 records come back each time — easy to compare prompt iterations.

**Gate:** don't proceed to Step 2 until Step 1 outputs look right.

---

## Step 2 — Run full augmentation (~45 minutes)

```bash
python -m models.augment.augment_descriptions
```

Output: `data/processed/metadata/listings_augmented.jsonl` with 1,100 rows.

Resumable: if the script crashes or you Ctrl-C it, re-running picks up where it left off.

**Sanity check after it finishes:**

```bash
wc -l data/processed/metadata/listings_augmented.jsonl
# should be ~1100

# Spot-check 3 random records
shuf -n 3 data/processed/metadata/listings_augmented.jsonl | \
  python3 -c 'import json,sys; [print("\nID:", j["item_id"], "\nORIG:", j["description_original"][:200], "\nAUG:", j["description_augmented"][:400]) for j in (json.loads(l) for l in sys.stdin)]'
```

If you see >5% obvious failures (e.g., "I cannot help with that"), bump `RATE_LIMIT_DELAY` in the script to 1.0 and rerun (the failures will resume from where it left off).

---

## Step 3 — Bundle dataset for Kaggle (~5 minutes)

```bash
python -m models.prepare_kaggle_zip
```

Produces `daraz_dataset_kaggle.zip` (~340 MB) at the project root.

Inside the zip:
```
metadata/listings.jsonl        ← augmented + original descriptions, normalized paths
splits/{train,val,test}.txt
images/<item_id>/*.jpg
```

---

## Step 4 — Upload to Kaggle as a Dataset (~10 minutes)

1. Go to <https://www.kaggle.com/datasets> → **New Dataset**.
2. Drag in `daraz_dataset_kaggle.zip`. Title it e.g. `daraz-vlm-augmented`.
3. Wait for upload + processing. Note the slug Kaggle assigns (visible in the URL: `kaggle.com/datasets/<username>/<slug>`).

---

## Step 5 — Train BLIP on Kaggle (~60 minutes)

1. Click **New Notebook** from your dataset page (this auto-attaches the dataset).
2. **Settings → Accelerator: GPU T4 x1**. (P100 also fine.) Internet **on**.
3. **Cell 1** — install:
   ```python
   !pip install -q transformers==4.40.0 accelerate==0.30.0
   ```
4. **Cell 2** — open `models/blip/train_kaggle.py` from this repo, copy its **entire contents**, paste into the cell.
5. Edit the line at the top of cell 2:
   ```python
   DATASET_SLUG = "daraz-vlm-augmented"
   ```
   so it matches whatever slug Kaggle assigned in Step 4.
6. **Run All**. Watch the sanity-check output that prints right after the dataset loads:

   ```
   ── sanity check ──
   Number of unmasked label tokens: 87
   These tokens decode to:
     This sleek wireless earbuds case comes in matte black with...
   ```

   If those tokens decode to the **description** (not the metadata prefix), the loss-mask fix is working correctly. If they look like "Product: ... Brand: ...", stop and check `n_meta_tokens` calculation.

7. Training runs ~45–60 min. Best checkpoint saved to `/kaggle/working/checkpoints/blip/best_model/`.
8. Click **Save Version** → **Save & Run All (commit)** when training finishes, so `/kaggle/working/` persists. (Without committing, you lose the checkpoint when the session ends.)

---

## Step 6 — Evaluate on Kaggle (~15 minutes)

Same notebook, new cell (or new notebook based on the same dataset, with the trained checkpoint dataset added):

1. Cell:
   ```python
   !pip install -q nltk rouge-score
   import nltk; nltk.download("punkt"); nltk.download("wordnet"); nltk.download("omw-1.4")
   ```
2. Paste the contents of `models/blip/evaluate_kaggle.py` into a new cell. Set `DATASET_SLUG` to the same value as in training.
3. Run. You'll get a printed metrics table, plus `/kaggle/working/results/blip_results.jsonl` and `blip_metrics.json`.

---

## What success looks like

The **headline number** to compare is BLEU-4 / ROUGE-L on the test set, but the real signal is qualitative. Open `blip_results.jsonl`, pull 20 random rows, and compare each `generated` vs the `reference`. You should see:

- The output describes visible features (color, material, design) instead of just repeating the product name.
- No more "Product: <name>. Brand: ..." metadata-echo lead-ins (loss-masking fix).
- No looping repetitions like "soft fabric soft fabric soft" (Phase 1 generation params).
- Prices in the output match metadata prices (price regex).

Compared to the original Stanford-style numbers, **expect BLEU-4 on the original test descriptions to roughly match or modestly improve, but qualitative read to be substantially better.** The original Daraz test descriptions are also metadata-fluff, so a model that learned to write rich descriptions will technically "miss" some test n-grams while producing better text.

If BLEU drops noticeably while quality looks better, that's the expected augmentation tradeoff — call it out in the writeup and back it with a small human-eval (say, 30 paired samples, 3 raters).

---

## Troubleshooting

**`401 Unauthorized` on NIM** — wrong key, or you didn't put it in `.env` at the project root. Check `cat .env`.

**`429 Too Many Requests`** — bump `RATE_LIMIT_DELAY` in `augment_descriptions.py` to 1.0 or 2.0 and rerun (it resumes).

**Image too large for NIM** — already handled (resize to 512px, JPEG quality 80, fallback to 60). If still failing, drop `MAX_IMAGE_DIM` to 384.

**Kaggle dataset slug mismatch** — `train_kaggle.py` will fail to find `/kaggle/input/<slug>/...`. Check the right sidebar in the notebook; the slug is the folder name under `/kaggle/input/`.

**Training OOM on T4** — already configured for batch=2, grad-accum=4, gradient checkpointing on. If it still OOMs, set `MAX_SEQ_LENGTH = 96`.

**Sanity-check unmasked tokens look like metadata, not description** — there's an off-by-one in `n_meta_tokens` for some weird metadata strings. Print `meta_only_ids` and the first 10 tokens of `input_ids` and adjust the `+1`.

---

## What's next (after today's run)

In rough priority order:
1. Retrain CLIP-GPT2 on the same augmented dataset (its loss-mask is already correct, so it benefits purely from better targets).
2. BLIP-2 + LoRA on the augmented dataset (architectural upgrade — bigger Q-Former, stronger visual grounding).
3. Multi-view image averaging at inference time (modest gain, free experiment).
4. Small human-eval (30 samples, 3 raters) on three model variants to back the qualitative claims.

See `improvement_plan.md` for the full plan.
