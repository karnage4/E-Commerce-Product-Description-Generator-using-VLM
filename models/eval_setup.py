"""
BLIP + CLIP-GPT2 Evaluation — Self-contained Colab/Kaggle cells.

SETUP OPTIONS:

Option A (Same Colab session as training — easiest):
  - Data is already at /content/daraz_data/
  - Checkpoints are already at /content/checkpoints/
  - Just run the cells below directly.

Option B (New Colab session):
  - Mount Google Drive
  - Data zip and checkpoints were backed up to Drive in previous steps
  - Unzip / copy them as shown in Cell 0B below

Option C (Kaggle):
  - Upload daraz_dataset_colab.zip as a Kaggle Dataset
  - Upload blip_best_model.zip and clip_gpt2_best_model.zip as another Dataset
  - Change DATA_ROOT and CHECKPOINT paths to /kaggle/input/...
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 0A — (Skip if same session as training)
# Mount Drive and restore data + checkpoints in a NEW Colab session
# ══════════════════════════════════════════════════════════════════════════════

"""
from google.colab import drive
import zipfile, shutil, os

drive.mount('/content/drive')
DRIVE = "/content/drive/MyDrive/daraz_cv_project"

# Restore dataset (only if /content/daraz_data doesn't exist yet)
if not os.path.exists("/content/daraz_data"):
    print("Unzipping dataset...")
    with zipfile.ZipFile(f"{DRIVE}/daraz_dataset_colab.zip") as z:
        z.extractall("/content/daraz_data")
    print("Done.")

# Restore BLIP checkpoint
if not os.path.exists("/content/checkpoints/blip/best_model"):
    print("Unzipping BLIP checkpoint...")
    with zipfile.ZipFile(f"{DRIVE}/blip_best_model.zip") as z:
        z.extractall("/content/checkpoints/blip/best_model")
    print("Done.")

# Restore CLIP-GPT2 checkpoint
if not os.path.exists("/content/checkpoints/clip_gpt2/best_model"):
    print("Unzipping CLIP-GPT2 checkpoint...")
    with zipfile.ZipFile(f"{DRIVE}/clip_gpt2_best_model.zip") as z:
        z.extractall("/content/checkpoints/clip_gpt2/best_model")
    print("Done.")
"""

# ══════════════════════════════════════════════════════════════════════════════
# CELL 0B — Kaggle path setup (use instead of 0A if on Kaggle)
# ══════════════════════════════════════════════════════════════════════════════

"""
# On Kaggle, datasets are mounted read-only under /kaggle/input/
# Adjust dataset slugs to match what you named them when uploading.

import zipfile, os

KAGGLE_DATA   = "/kaggle/input/daraz-cv-data/daraz_dataset_colab.zip"
KAGGLE_BLIP   = "/kaggle/input/daraz-cv-checkpoints/blip_best_model.zip"
KAGGLE_CLIP   = "/kaggle/input/daraz-cv-checkpoints/clip_gpt2_best_model.zip"

for src, dst in [
    (KAGGLE_DATA, "/tmp/daraz_data"),
    (KAGGLE_BLIP, "/tmp/checkpoints/blip/best_model"),
    (KAGGLE_CLIP, "/tmp/checkpoints/clip_gpt2/best_model"),
]:
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
        with zipfile.ZipFile(src) as z:
            z.extractall(dst)
        print(f"Extracted -> {dst}")

# Then set DATA_ROOT and CHECKPOINT_BLIP/CLIP below to /tmp/... paths
"""


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1 — Install only what Colab/Kaggle doesn't have (30 seconds)
# ══════════════════════════════════════════════════════════════════════════════

"""
# Paste and run this cell first:
!pip install -q nltk rouge-score
import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
print("Done. torch, transformers, pillow already pre-installed on Colab/Kaggle.")
"""
