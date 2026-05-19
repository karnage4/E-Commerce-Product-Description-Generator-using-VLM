# Daraz Product Description Generator
### Automated E-commerce Description Generation via Vision-Language Model Fine-tuning

> A replication of the Stanford *"Automated Product Description Generation for E-commerce via Vision-Language Model Fine-tuning"* paper, applied to a novel **Pakistani Daraz e-commerce dataset**.

---

## Project Overview

End-to-end pipeline to automatically generate product descriptions from product images and metadata, trained on data scraped from [Daraz.pk](https://www.daraz.pk) — Pakistan's largest e-commerce platform.

### Models Implemented
| Model | Type | Stage |
|---|---|---|
| **BLIP** (`blip-image-captioning-base`) | Encoder-Decoder VLM | Stage 1 — visual description |
| **CLIP + GPT-2** | Vision prefix + Causal LM | Stage 1 — visual description |
| **Fine-tuned GPT-2** (extracted from CLIP-GPT2) | Causal LM | Stage 2 — metadata refiner |
| **Gemma 4 31B** (OpenRouter) | LLM judge + data augmentation | Stage 1 scorer + description rewriter |
| **Gemini 1.5 Flash** | API zero-shot baseline | Comparison baseline |

---

## Results

### NLP Metrics (test set, 135 samples)

| Model | BLEU-1 | BLEU-4 | ROUGE-L | METEOR |
|---|---|---|---|---|
| BLIP Fine-tuned | 9.23 | 0.56 | **12.41** | 12.33 |
| CLIP-GPT2 Fine-tuned | **11.63** | **1.08** | 10.23 | 10.53 |
| Two-Stage (BLIP + GPT-2) | 9.91 | 0.54 | 8.42 | 8.37 |

> CIDEr requires `pycocoevalcap` — skipped in current runs.

### Per-Category ROUGE-L Breakdown

| Category | n | BLIP | CLIP-GPT2 | Two-Stage |
|---|---|---|---|---|
| womens-fashion | 35 | **14.90** | 10.64 | 10.44 |
| tablets | 38 | **13.09** | 11.28 | 8.57 |
| consumer-electronics | 27 | **12.64** | 12.08 | 9.81 |
| home-appliances | 1 | 11.27 | 9.09 | **11.83** |
| smartphones | 34 | **8.94** | 7.20 | 4.53 |

### Inference Hyperparameter Sweep (CLIP-GPT2)

Best config: `num_beams=6, no_repeat_ngram_size=3, max_new_tokens=150` (ROUGE-L: **7.24**)

| num_beams | ngram | max_tok | ROUGE-L |
|---|---|---|---|
| 6 | 3 | 150 | **7.24** ← best |
| 4 | 2 | 150 | 6.91 |
| 4 | 4 | 150 | 6.82 |
| 2 | 3 | 150 | 6.79 |
| 4 | 3 | 150 | 6.67 |
| 4 | 3 | 100 | 6.58 |
| 4 | 3 | 200 | 6.58 |

### Training LR Sweep

**BLIP** (200-sample subset, 3 epochs)

| Learning Rate | Val Loss |
|---|---|
| 5e-6 | 7.069 |
| **1e-5** | **7.012** ← best |
| 3e-5 | 7.111 |

**CLIP-GPT2** (200-sample subset, 3 epochs)

| Learning Rate | Val Loss |
|---|---|
| **1e-5** | **4.151** ← best |
| 2e-5 | 4.164 |
| 5e-5 | 4.280 |

---

## Improvements Made

### 1. Gemma 4 Description Augmentation (`models/augment/augment_descriptions.py`)
Raw Daraz descriptions echo metadata rather than describe visual features. We rewrote all ~1,100 training descriptions using **Gemma 4 31B** (multimodal, via OpenRouter), with category-specific visual feature prompts:

- **Smartphones**: bezel thickness, camera arrangement, fingerprint sensor location, finish
- **Fashion**: fabric texture, neckline style, print/pattern, embroidery, silhouette
- **Tablets**: slim profile, stand/case, speaker grilles, port placement
- **Travel bags**: material, wheel type, pocket layout, zipper color
- **Toys**: dominant colors, character design, size impression, moving parts
- *(+ 4 other categories)*

Training on augmented descriptions forces the model to reproduce visually-grounded text, giving it a real learning signal from the image.

### 2. BLIP Loss Masking Fix (`models/blip/train_colab.py`)
Previously, loss was computed on **both** metadata and description tokens. The model spent most of its loss budget learning to reproduce metadata it already had access to. Fixed by tokenizing the metadata prefix separately and masking those positions to `-100`, so only description tokens contribute to loss — matching what CLIP-GPT2 already did correctly.

### 3. Automated Training HP Sweep (`--auto-sweep` flag)
Both training scripts now accept `--auto-sweep`: runs 3 LR candidates (5e-6, 1e-5, 3e-5) on a 200-sample subset for 3 epochs, saves results to `models/results/training_sweep_{model}.json`, picks the best LR, then launches full training automatically.

### 4. Vision Encoder Attention Rollout (`models/eval/attention_viz.py`)
Replaced text-decoder cross-attention hooks (unreliable when model echoes metadata) with **attention rollout** through BLIP's ViT vision encoder. Accumulates CLS→patch attention through all layers with residual connections, giving spatially meaningful heatmaps independent of the text decoder's behaviour.

### 5. Parallel Description Augmentation
Augmentation uses `ThreadPoolExecutor` with 5 parallel workers, reducing augmentation time from ~20 min to ~4 min for the full training set.

### 6. Local Training Support
Both training scripts were updated from hardcoded Colab paths (`/content/daraz_data`) to paths derived from `__file__`, with `BATCH_SIZE` increased from 2→4 and `num_workers` from 0→4 for the RTX 5060 Ti (17GB VRAM).

### 7. Image Augmentation (`models/blip/train_colab.py`, `models/clip_gpt2/train_colab.py`)
Added stochastic image augmentation on the training split only: `RandomHorizontalFlip(p=0.5)`, `ColorJitter(brightness/contrast/saturation=0.2, hue=0.05)`, `RandomRotation(10°)`. Val/test images always use the deterministic transform. BLIP applies augmentation as a PIL transform before the BLIP processor; CLIP-GPT2 bakes it into a `TRAIN_CLIP_TRANSFORM` pipeline.

### 8. Training Curves (`models/eval/training_curves.py`)
Epoch-by-epoch train and val loss are now saved to `models/results/training_history_{model}.json` at the end of each full training run. A separate reconstruction script (`models/eval/training_curves.py`) can also recover val loss curves from existing epoch checkpoints by running a forward pass on the val split. Plots saved to `models/results/training_curves.png`.

### 9. EDA Figures (`models/eval/eda.py`)
Six dataset figures saved to `models/results/eda/`: category distribution, train/val/test split sizes, description word-count histogram, images-per-product histogram, word-count boxplot by category, category×split breakdown.

### 10. Hard-Case Badges in Qualitative Report
Each card in the "Hard Cases" section of `qualitative_report.html` now carries an explicit purple **HARD CASE** badge, making them immediately distinguishable from TP/FP/FN examples.

---

## Pipeline Architecture

```
Phase 1: Web Scraping        scraper/daraz_scraper.py
Phase 2: Data Cleaning       pipeline/cleaner.py
Phase 3: Deduplication       dedup/deduplicator.py
Phase 4: Dataset Building    organizer/dataset_builder.py
Phase 5: Description Augment models/augment/augment_descriptions.py  ← NEW
Phase 6: Model Fine-tuning   models/blip/, models/clip_gpt2/
Phase 7: Two-Stage Inference models/two_stage_pipeline.py
Phase 8: Evaluation          models/eval/
```

**Two-stage inference:**
1. **Stage 1** — VLM sees `image + category only` → generates visual description (forced to use the image)
2. **Stage 2** — Fine-tuned GPT-2 sees `stage1 description + full metadata` → polished final description
3. **Judge** — Gemma 4 31B scores Stage 1 on visual grounding / fluency / relevance (1–5)

---

## Setup

### 1. Install uv (if not already installed)
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create the virtual environment
```powershell
uv venv --python 3.11 .venv
```

### 3. Activate it
```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install PyTorch (CUDA 12.8 — RTX 50 series / any CUDA 12.x GPU)
```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install all other dependencies
```powershell
uv pip install transformers accelerate openai python-dotenv Pillow tqdm nltk rouge-score matplotlib numpy requests beautifulsoup4 lxml rapidfuzz imagehash scipy
```

### 6. Configure API keys
Create `.env` at the project root:
```
OPENROUTER_API_KEY=sk-or-...     # Gemma 4 judge + augmentation
GEMINI_API_KEY=...               # Gemini baseline (optional)
```

---

## Running the Pipeline

### Step 1 — Augment training descriptions (Gemma 4, ~4 min)
```powershell
python -m models.augment.augment_descriptions
```
Resumable. Output: `data/data/processed/metadata/listings_augmented.jsonl`

### Step 2 — Train (auto LR sweep → full training)
```powershell
python -m models.blip.train_colab --auto-sweep --augmented      # ~90 min
python -m models.clip_gpt2.train_colab --auto-sweep --augmented # ~60 min
```

### Step 3 — Inference HP sweep
```powershell
python -m models.hparam_sweep --model blip
python -m models.hparam_sweep --model clip_gpt2
```

### Step 4 — Evaluate
```powershell
python -m models.blip.evaluate
python -m models.clip_gpt2.evaluate
python -m models.compare_results

python -m models.two_stage_pipeline --model blip
python -m models.two_stage_pipeline --model clip_gpt2
```

### Step 5 — Error analysis
```powershell
python -m models.eval.category_breakdown
python -m models.eval.qualitative_sampler --model blip
python -m models.eval.qualitative_sampler --model clip_gpt2
python -m models.eval.attention_viz --num-samples 6
```

### Step 6 — Training curves (from epoch checkpoints)
```powershell
python -m models.eval.training_curves          # reconstructs + saves training_curves.png
python -m models.eval.training_curves --plot-only  # re-plot if JSON already exists
```
Output: `models/results/training_curves.png`, `models/results/training_history_{blip,clip_gpt2}.json`

### Step 7 — EDA figures
```powershell
python -m models.eval.eda
```
Output: `models/results/eda/*.png` (6 figures: category distribution, split sizes, description length, images per product, category×split breakdown, boxplot by category)

### Optional — Gemini baseline
```powershell
python -m models.api_baseline.gemini_baseline
```

> **Note:** Delete `models/results/*.jsonl` before re-evaluating after retraining — the scripts are resumable and will skip already-processed items otherwise.

---

## Dataset

- **Source:** [Daraz.pk](https://www.daraz.pk) (Pakistani e-commerce)
- **Categories:** 9 (consumer-electronics, smartphones, tablets, womens-fashion, mens-fashion, home-appliances, beauty-health, travel-bags, toys)
- **Size:** ~1,370 products with multi-angle images
- **Splits:** 80% train / 10% val / 10% test

> Raw data and images are excluded from this repo. Run `python run.py` to regenerate.

---

## Hardware

| Task | Requirement | Time estimate |
|---|---|---|
| Scraping + cleaning | CPU, 8GB RAM | ~2 hrs for full dataset |
| Description augmentation | Internet (OpenRouter API) | ~4 min (5 parallel workers) |
| Fine-tuning | GPU, 8GB+ VRAM | ~60–90 min/model on RTX 5060 Ti |
| Inference / evaluation | GPU preferred, CPU works | ~2 min GPU vs ~30 min CPU |
| Two-stage pipeline | GPU + internet (OpenRouter) | ~5 min GPU for 135 samples |

---

## Key Technical Details

- **CAPTCHA-resilient scraper** — Playwright-stealth + automated Alibaba slider CAPTCHA solver
- **Two-pass deduplication** — RapidFuzz title similarity (threshold 82) + pHash image hashing (Hamming ≤ 8)
- **BLIP loss masking** — Metadata prefix tokens masked to `-100`; only description tokens contribute to loss
- **CLIP-GPT2 loss masking** — Visual prefix + metadata prompt masked; loss on description tokens only
- **Gemma 4 augmentation** — Category-specific visual prompts rewrite training descriptions to emphasize visible features (color, finish, shape, design details) rather than echoing metadata
- **Stage 1 minimal prompt** — VLM receives `Category: X` only at inference, forcing visual grounding
- **LLM judge** — Gemma 4 31B scores Stage 1 outputs on visual_grounding / fluency / relevance (JSON, 1–5 scale)
- **Attention rollout** — ViT vision encoder CLS→patch attention accumulated through all layers with residual connections for interpretable spatial heatmaps
- **Auto HP sweep** — `--auto-sweep` flag runs LR candidates on 200-sample subset before full training

---

## Reference

> *"Automated Product Description Generation for E-commerce via Vision-Language Model Fine-tuning"*  
> Stanford University CS231N — methodological blueprint for this work.
