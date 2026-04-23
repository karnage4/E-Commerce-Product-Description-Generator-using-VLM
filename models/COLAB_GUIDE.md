# Google Colab Training Guide

> **For 8GB RAM laptops** — all heavy training runs on Colab's free T4 GPU (16 GB VRAM).  
> Your local machine is only used for data prep, API inference, and evaluation.

---

## Step 0 — Prepare Dataset Zip (run locally)

```powershell
cd "d:\downloads\Semester 6\Introduction to Computer Vision\Project\Project\Project"
python -m models.prepare_colab_zip
```

This creates `daraz_dataset_colab.zip` in the project root.  
Upload this file to **Google Drive** (My Drive → create a folder `daraz_cv_project/`).

---

## Step 1 — Open Google Colab

Go to [https://colab.research.google.com](https://colab.research.google.com)  
Create a new notebook. Go to **Runtime → Change runtime type → T4 GPU**.

---

## Step 2 — Mount Drive & Unzip Dataset

Paste this into the **first cell**:

```python
# Cell 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip dataset
import zipfile
zip_path = "/content/drive/MyDrive/daraz_cv_project/daraz_dataset_colab.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall("/content/daraz_data")

print("Dataset extracted!")
import os
print(os.listdir("/content/daraz_data"))
```

---

## Step 3 — Install Dependencies

```python
# Cell 2: Install packages
!pip install -q transformers accelerate pillow tqdm
# Verify GPU
import torch
print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
```

---

## Step 4A — Train BLIP Model

```python
# Cell 3a: Upload and run BLIP training
# First, upload models/blip/train_colab.py using the Colab file sidebar
# OR paste the entire content of train_colab.py directly here

# Then run:
# (The script auto-detects /content/daraz_data and uses T4 GPU)
exec(open("/content/train_colab.py").read())   # if uploaded
train()
```

**Expected training time:** ~60–90 minutes for 5 epochs on T4 GPU with ~1,000 samples.

---

## Step 4B — Train CLIP-GPT2 Model

```python
# Cell 3b: Upload models/clip_gpt2/model.py AND models/clip_gpt2/train_colab.py
# Run in the same Colab session (or a new one after BLIP is done)
exec(open("/content/train_colab_clipgpt2.py").read())
train()
```

---

## Step 5 — Download Checkpoints

After training finishes, download the best model to your local machine:

```python
# Cell 4: Zip and download checkpoint
import shutil
shutil.make_archive("/content/blip_best_model", "zip", "/content/checkpoints/blip/best_model")

from google.colab import files
files.download("/content/blip_best_model.zip")
```

Place the unzipped `best_model/` folder at:
```
models/checkpoints/blip/best_model/
models/checkpoints/clip_gpt2/best_model/
```

---

## Step 6 — Run Evaluation Locally (on your 8GB laptop)

```powershell
# BLIP evaluation (CPU inference, ~15-30 min for 150 samples)
python -m models.blip.evaluate --max-samples 150

# View combined results table
python -m models.compare_results
```

---

## Tips for Colab Free Tier

| Tip | Detail |
|---|---|
| **Save frequently** | Colab disconnects after ~90 min idle or ~12h total |
| **Use Drive for checkpoints** | Save directly to Drive so disconnects don't lose work |
| **T4 is free** | Always select T4 GPU, not CPU |
| **Watch VRAM** | Run `!nvidia-smi` to check VRAM usage |
| **Batch size** | If OOM, reduce BATCH_SIZE from 8 to 4 |

---

## Expected VRAM Usage

| Model | Batch Size | fp16 | VRAM |
|---|---|---|---|
| BLIP base | 8 | Yes | ~8 GB |
| CLIP-GPT2 | 8 | Yes | ~7 GB |
| BLIP-2 + LoRA | 4 | Yes | ~14 GB |

All fit within Colab T4's 15 GB VRAM.
