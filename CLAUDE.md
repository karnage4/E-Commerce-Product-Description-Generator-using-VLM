# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project generates product descriptions from images and metadata for Daraz.pk (Pakistan's largest e-commerce platform) using Vision-Language Models. It replicates the approach from the Stanford CS231N paper on automated product description generation, adapted to a novel Pakistani e-commerce dataset.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Data Pipeline (run locally)
```bash
# Full pipeline
python run.py

# Individual phases
python run.py --steps scrape
python run.py --steps clean
python run.py --steps dedup
python run.py --steps build

# Specific categories
python run.py --steps scrape --categories smartphones tablets

# Force re-deduplication
python run.py --rededup
```

### Model Training (Google Colab / Kaggle — GPU required)
```bash
# Prepare dataset zip for upload
python -m models.prepare_colab_zip

# Upload daraz_dataset_colab.zip to Google Drive, then in Colab run:
# models/blip/train_colab.py
# models/clip_gpt2/train_colab.py
```

### Evaluation (local CPU)
```bash
python -m models.blip.evaluate
python -m models.clip_gpt2.evaluate
python -m models.compare_results

# Gemini baseline (needs GEMINI_API_KEY)
python -m models.api_baseline.gemini_baseline
```

### Hyperparameter Sweep (inference-only, no retraining)
```bash
# One-at-a-time sweep — ~5 min/config on CPU, 50 samples
python -m models.hparam_sweep --model blip
python -m models.hparam_sweep --model clip_gpt2

# Quick smoke-test (30 samples)
python -m models.hparam_sweep --model blip --max-samples 30

# Full 27-combination grid (long — use on Colab)
python -m models.hparam_sweep --model blip --full-grid
```

### Evaluation & Error Analysis (C2/C3)
```bash
# C2: per-category ROUGE-L breakdown (confusion matrix substitute)
python -m models.eval.category_breakdown

# C3: TP/FP/FN/Hard cases — generates qualitative_report.html
python -m models.eval.qualitative_sampler              # default: blip
python -m models.eval.qualitative_sampler --model clip_gpt2
python -m models.eval.qualitative_sampler --model two_stage

# C3: Grad-CAM / cross-attention maps (3 samples, BLIP only)
python -m models.eval.attention_viz
python -m models.eval.attention_viz --num-samples 3 --checkpoint <path>
```

### Two-Stage Pipeline (VLM → Gemma refiner)
```bash
# Requires OPENROUTER_API_KEY in .env
python -m models.two_stage_pipeline --model blip
python -m models.two_stage_pipeline --model clip_gpt2
python -m models.two_stage_pipeline --model blip --max-samples 20   # quick test
```

### Caption Augmentation
```bash
# Requires NVIDIA_API_KEY in environment
python -m models.augment.augment_descriptions
```

## Architecture

### Pipeline Phases

The project has a 6-phase pipeline coordinated by `run.py`:

1. **Scrape** (`scraper/daraz_scraper.py`) — Playwright + stealth scraping with automated slider CAPTCHA solving and AJAX interception. Output: `data/raw/{category}.jsonl`
2. **Clean** (`pipeline/cleaner.py`) — HTML unescaping, quality filtering (min 20-word descriptions, max 5 emojis, <50% caps ratio, spam detection), and rule-based subcategory labeling. Output: `data/processed/metadata/listings_clean.jsonl`
3. **Dedup** (`dedup/deduplicator.py`) — Two-pass: RapidFuzz title similarity (threshold 82) then perceptual image hashing pHash (Hamming threshold 8). Keeps highest-review listing per group.
4. **Build** (`organizer/dataset_builder.py`) — Downloads images into `data/processed/images/{item_id}/`, creates 80/10/10 train/val/test splits in `data/processed/splits/`.
5. **Train** — Run on Colab/Kaggle T4 GPU. Checkpoints saved to `models/checkpoints/{blip,clip_gpt2}/best_model/`.
6. **Evaluate** — Local CPU evaluation. Metrics: BLEU-1/4, ROUGE-L, METEOR, CIDEr (computed in `models/shared/metrics.py`).

### Model Architectures

**BLIP** (`models/blip/`): `Salesforce/blip-image-captioning-base` fine-tuned end-to-end. Input is metadata prompt + description concatenated. Loss masking should restrict loss to description tokens only (labels for metadata tokens set to -100).

**CLIP + GPT-2** (`models/clip_gpt2/`): Custom architecture in `model.py`. CLIP vision encoder (frozen for first 2 epochs) → linear projection → 10-token visual prefix prepended to GPT-2 input. Loss masking is already correctly implemented — only description tokens contribute to loss.

**Gemini 1.5 Flash** (`models/api_baseline/`): Zero-shot baseline via free API. 15 req/min, 1M tokens/day limit.

### Shared Utilities (`models/shared/`)

- `config.py` — Hyperparameters, model paths, and the `build_metadata_prompt()` function used consistently across all models
- `dataset.py` — `DarazProductDataset` PyTorch class with train/val/test split support
- `metrics.py` — All evaluation metrics

### Caption Augmentation (`models/augment/`)

Uses NVIDIA NIM (Llama 3.2 11B Vision-Instruct) to rewrite Daraz descriptions to emphasize visual features rather than echoing metadata. Output goes to `data/processed/metadata/listings_augmented.jsonl` with a `description_augmented` field. Resumable — skips already-processed items. To use augmented descriptions in training, switch the dataset's description field from `description` to `description_augmented`.

## Environment Variables

| Variable | Purpose | Source |
|---|---|---|
| `GEMINI_API_KEY` | Gemini 1.5 Flash baseline | aistudio.google.com (free) |
| `NVIDIA_API_KEY` | NVIDIA NIM caption augmentation | build.nvidia.com (free tier) |
| `OPENROUTER_API_KEY` | Stage 2 Gemma refiner | openrouter.ai |

## Configuration

- **Root `config.py`** — Categories (9 total), scraping rate limits, dedup thresholds, split ratios, data paths
- **`models/shared/config.py`** — Model hyperparameters and inference templates; the `build_metadata_prompt()` function is the single source of truth for how metadata is formatted as a prompt

## Data Layout

```
data/
  raw/              # Raw scraped JSONL per category (gitignored)
  processed/
    metadata/       # listings_final.jsonl, listings_augmented.jsonl
    images/         # {item_id}/{0.jpg, 1.jpg, ...}
    splits/         # train.txt, val.txt, test.txt
```

The `data/` directory is gitignored; regenerate via `python run.py`.

## Known Issues / Improvement Areas

See `improvement_plan.md` for the full roadmap. Key known issues:

- **BLIP loss masking**: The training loop needs to set metadata token labels to -100 so loss is only computed on description tokens — CLIP-GPT2 already does this correctly.
- **Single image**: Both models currently use only the first successfully-loaded image. Multi-view averaging at inference is a planned improvement.
- **Metadata echoing**: Raw Daraz descriptions tend to repeat product metadata rather than describe visual features. The augmentation pipeline in `models/augment/` addresses this at the data level.
- **Two-stage pipeline**: `models/two_stage_pipeline.py` runs Stage 1 (VLM with `build_stage1_prompt` — category only) then Stage 2 (OpenRouter Gemma refines with full metadata). Both descriptions saved per item to `models/results/two_stage_results.jsonl`. For best Stage 1 quality, retrain VLMs using `build_stage1_prompt()` instead of `build_metadata_prompt()` so training matches inference-time input.
- **OpenRouter model**: Stage 2 uses `google/gemma-4-31b-it` via OpenRouter.
