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
| **Gemma 4 31B** (OpenRouter) | LLM judge | Stage 1 scorer (visual grounding, fluency, relevance) |
| **Gemini 1.5 Flash** | API zero-shot baseline | Comparison baseline |

---

## Pipeline Architecture

```
Phase 1: Web Scraping        scraper/daraz_scraper.py
Phase 2: Data Cleaning       pipeline/cleaner.py
Phase 3: Deduplication       dedup/deduplicator.py
Phase 4: Dataset Building    organizer/dataset_builder.py
Phase 5: Model Fine-tuning   models/blip/, models/clip_gpt2/
Phase 6: Two-Stage Inference models/two_stage_pipeline.py
Phase 7: Evaluation          models/eval/
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
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat
```

### 4. Install PyTorch (CUDA 12.8 — RTX 50 series / any CUDA 12.x GPU)
```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install all other dependencies
```powershell
uv pip install transformers accelerate openai python-dotenv Pillow tqdm nltk rouge-score matplotlib numpy requests beautifulsoup4 lxml rapidfuzz imagehash
```

### 6. Configure API keys
Create `.env` at the project root (already present, just fill in keys):
```
OPENROUTER_API_KEY=sk-or-...     # for Stage 2 Gemma judge
NVIDIA_API_KEY=nvapi-...         # for caption augmentation (optional)
GEMINI_API_KEY=...               # for Gemini baseline (optional)
```

---

## Running the Pipeline

### Step 1 — Build the dataset
```powershell
# Full pipeline (scrape → clean → dedup → build)
python run.py

# Or skip scraping if data/processed/ already exists
python run.py --steps build
```

### Step 2 — Sweep training hyperparameters (before full training)

In `models/shared/config.py`, temporarily set `subset_samples` and try different learning rates:

```python
# In BLIP_CONFIG (models/shared/config.py):
"subset_samples": 200,   # ~15 min/run instead of ~90 min
"num_epochs":     3,
"learning_rate":  5e-6,  # try: 5e-6, 1e-5, 3e-5 — pick best val ROUGE-L
```

```powershell
python -m models.blip.train_colab       # run once per LR value (~15 min each)
python -m models.clip_gpt2.train_colab  # same for CLIP-GPT2 (try prefix_length too)
```

Pick the best LR, reset `subset_samples = None`, then proceed to full training.

### Step 3 — Full training

```powershell
python -m models.blip.train_colab       # ~90 min on RTX 5060 Ti
python -m models.clip_gpt2.train_colab  # ~60 min on RTX 5060 Ti
```

**Google Colab (alternative):**
```powershell
python -m models.prepare_colab_zip      # creates daraz_dataset_colab.zip
# Upload to Google Drive, run train_colab.py cells, download best_model/ to:
#   models/blip/blip_best_model/
#   models/clip_gpt2/clip_gpt2_best_model/
```

### Step 4 — Sweep inference hyperparameters (after training)

```powershell
# Sweep beam width / ngram size / max tokens against existing checkpoint
python -m models.hparam_sweep --model blip
python -m models.hparam_sweep --model clip_gpt2
python -m models.hparam_sweep --model blip --max-samples 30   # quick test
```

Use the best config reported for all subsequent evaluation runs.

### Step 5 — Evaluate

```powershell
python -m models.blip.evaluate
python -m models.clip_gpt2.evaluate
python -m models.compare_results

# Two-stage pipeline (Stage 1 VLM + Stage 2 GPT-2 + Gemma judge)
python -m models.two_stage_pipeline --model blip
python -m models.two_stage_pipeline --model clip_gpt2
```

### Step 6 — Error analysis & reporting

```powershell
# C2: per-category ROUGE-L breakdown
python -m models.eval.category_breakdown

# C3: TP / FP / FN / Hard cases → qualitative_report.html
python -m models.eval.qualitative_sampler --model blip

# C3: Grad-CAM attention maps (3 samples)
python -m models.eval.attention_viz
```

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
| Fine-tuning | GPU, 8GB+ VRAM | ~60–90 min/model |
| Inference / evaluation | GPU preferred, CPU works | ~2 min GPU vs ~30 min CPU |
| Two-stage pipeline | GPU + internet (OpenRouter) | ~5 min GPU for 150 samples |

---

## Key Technical Details

- **CAPTCHA-resilient scraper** — Playwright-stealth + automated Alibaba slider CAPTCHA solver
- **Two-pass deduplication** — RapidFuzz title similarity (threshold 82) + pHash image hashing (Hamming ≤ 8)
- **Loss masking** — Only description tokens contribute to loss; metadata prefix tokens are masked (`labels = -100`)
- **Stage 1 minimal prompt** — VLM receives `Category: X` only, forcing visual grounding rather than metadata echoing
- **LLM judge** — Gemma 4 31B scores Stage 1 outputs on visual_grounding / fluency / relevance (JSON, 1–5 scale)

---

## Reference

> *"Automated Product Description Generation for E-commerce via Vision-Language Model Fine-tuning"*  
> Stanford University CS231N — methodological blueprint for this work.
