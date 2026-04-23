# Daraz Product Description Generator
### Automated E-commerce Description Generation via Vision-Language Model Fine-tuning

> A replication of the Stanford *"Automated Product Description Generation for E-commerce via Vision-Language Model Fine-tuning"* paper, applied to a novel **Pakistani Daraz e-commerce dataset**.

---

## 📋 Project Overview

This project builds an end-to-end pipeline to automatically generate product descriptions from product images and metadata, trained on data scraped from [Daraz.pk](https://www.daraz.pk) — Pakistan's largest e-commerce platform.

### Models Implemented
| Model | Type | Training |
|---|---|---|
| **BLIP** (`blip-image-captioning-base`) | Encoder-Decoder VLM | Fine-tuned on Google Colab T4 |
| **CLIP + GPT-2** | Vision prefix + Causal LM | Fine-tuned on Google Colab T4 |
| **Gemini 1.5 Flash** | API (zero-shot baseline) | Free tier inference |
| **Metadata-Only** | Text baseline | No model needed |

---

## 🏗️ Pipeline Architecture

```
Phase 1: Web Scraping          ✅  scraper/daraz_scraper.py
Phase 2: Data Cleaning         ✅  pipeline/cleaner.py
Phase 3: Deduplication         ✅  dedup/deduplicator.py
Phase 4: Dataset Building      ✅  organizer/dataset_builder.py
Phase 5: Model Fine-tuning     ✅  models/blip/, models/clip_gpt2/
Phase 6: Evaluation            ✅  models/shared/metrics.py
```

---

## 📁 Repository Structure

```
├── config.py                        # Central config (paths, thresholds, categories)
├── run.py                           # CLI orchestrator (scrape → clean → build)
├── requirements.txt
│
├── scraper/
│   └── daraz_scraper.py             # Playwright + CAPTCHA-solving scraper
│
├── pipeline/
│   └── cleaner.py                   # Quality filtering + subcategory labelling
│
├── dedup/
│   └── deduplicator.py              # Fuzzy title + perceptual hash deduplication
│
├── organizer/
│   └── dataset_builder.py           # Image downloader + train/val/test splits
│
└── models/
    ├── COLAB_GUIDE.md               # Step-by-step Colab training guide
    ├── prepare_colab_zip.py         # Zip dataset for Colab upload
    ├── compare_results.py           # Print combined metrics table
    ├── compare_results_colab.py     # Colab version of results table
    │
    ├── shared/
    │   ├── config.py                # Model paths + hyperparameters
    │   ├── dataset.py               # DarazProductDataset (PyTorch)
    │   └── metrics.py               # BLEU, ROUGE-L, METEOR, CIDEr
    │
    ├── blip/
    │   ├── train_colab.py           # BLIP fine-tuning (run on Colab)
    │   ├── evaluate.py              # Local CPU evaluation
    │   └── evaluate_colab.py        # Colab/Kaggle evaluation cell
    │
    ├── clip_gpt2/
    │   ├── model.py                 # ClipGPT2Model architecture
    │   ├── train_colab.py           # CLIP-GPT2 fine-tuning (run on Colab)
    │   ├── evaluate.py              # Local CPU evaluation
    │   └── evaluate_colab.py        # Colab/Kaggle evaluation cell
    │
    └── api_baseline/
        └── gemini_baseline.py       # Gemini API zero-shot baseline
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Scrape Daraz data
```bash
python run.py --steps scrape
```

### 3. Clean and build dataset
```bash
python run.py --steps clean build
```

### 4. Run Gemini API baseline (locally, free)
```bash
set GEMINI_API_KEY=your_key_here          # Windows
python -m models.api_baseline.gemini_baseline
```
Get a free key at: https://aistudio.google.com/app/apikey

### 5. Fine-tune models (Google Colab)
See [`models/COLAB_GUIDE.md`](models/COLAB_GUIDE.md) for the complete step-by-step guide.

### 6. Evaluate
```bash
# After downloading checkpoints from Colab:
python -m models.blip.evaluate
python -m models.clip_gpt2.evaluate
python -m models.compare_results
```

---

## 📊 Dataset

- **Source:** [Daraz.pk](https://www.daraz.pk) (Pakistani e-commerce)
- **Categories:** Consumer Electronics, Smartphones, Tablets, Women's Fashion, Home Appliances
- **Size:** ~1,370 products with multi-angle images
- **Splits:** 80% train / 10% val / 10% test

> **Note:** Raw data and images are excluded from this repo (too large). Run the scraper to regenerate.

---

## 🔑 Key Technical Contributions

- **CAPTCHA-resilient scraper** — Playwright-stealth + automated slider CAPTCHA solver
- **Two-pass deduplication** — RapidFuzz title similarity + pHash image comparison
- **Novel Daraz dataset** — First Pakistani e-commerce VLM dataset
- **Memory-efficient fine-tuning** — Gradient accumulation + gradient checkpointing for 15GB VRAM

---

## 🛠️ Hardware Requirements

| Task | Where to run | Requirements |
|---|---|---|
| Scraping + cleaning | Local | 8GB RAM, any CPU |
| API baseline (Gemini) | Local | Internet + free API key |
| Fine-tuning | Google Colab | Free T4 GPU (15GB VRAM) |
| Evaluation | Colab or Local | GPU preferred, CPU works |

---

## 📖 Reference Paper

> *"Automated Product Description Generation for E-commerce via Vision-Language Model Fine-tuning"*  
> Stanford University CS231N — used as the methodological blueprint for this work.

---

## 📦 Dependencies

See `requirements.txt`. Key packages:
- `playwright` — headless browser scraping
- `transformers` — BLIP, CLIP, GPT-2 models
- `torch` / `torchvision` — training (install on Colab)
- `nltk`, `rouge-score` — evaluation metrics
- `google-generativeai` — Gemini API baseline
