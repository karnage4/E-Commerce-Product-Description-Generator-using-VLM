"""
Qualitative Evaluation — Section 6.6 of Report

Paste this ENTIRE file as ONE Colab cell and run it.

What it does:
  1. Auto-restores checkpoints + dataset from Google Drive if not already in /content/
  2. Loads BLIP and CLIP-GPT2 from their checkpoints
  3. Picks 5 diverse test samples (one per category if possible)
  4. For each sample: displays product image + prints a clean comparison box
  5. Saves a combined HTML report you can screenshot or print to PDF
"""

# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — AUTO-RESTORE CHECKPOINTS AND DATA (handles fresh Colab sessions)
# ══════════════════════════════════════════════════════════════════════════════

import os, zipfile, sys
from pathlib import Path

BLIP_CKPT_PATH  = Path("/content/checkpoints/blip/best_model")
CLIP_CKPT_PATH  = Path("/content/checkpoints/clip_gpt2/best_model")
DATA_PATH       = Path("/content/daraz_data")

# What we need and where to find it on Drive
DRIVE_DIR = "/content/drive/MyDrive/daraz_cv_project"
RESTORE_MAP = {
    BLIP_CKPT_PATH:  (f"{DRIVE_DIR}/blip_best_model.zip",         "model.safetensors"),
    CLIP_CKPT_PATH:  (f"{DRIVE_DIR}/clip_gpt2_best_model.zip",    "model.pt"),
    DATA_PATH:       (f"{DRIVE_DIR}/daraz_dataset_colab.zip",     "metadata/listings_final.jsonl"),
}

def check_missing():
    missing = []
    for dst, (_, marker) in RESTORE_MAP.items():
        if not (dst / marker).exists():
            missing.append(dst)
    return missing

missing = check_missing()

if missing:
    print(f"[!] Missing: {[str(m) for m in missing]}")
    print("    Attempting to restore from Google Drive...")

    # Mount Drive (no-op if already mounted)
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        drive_available = os.path.exists(DRIVE_DIR)
    except Exception as e:
        drive_available = False
        print(f"    Drive mount failed: {e}")

    if drive_available:
        for dst, (zip_src, marker) in RESTORE_MAP.items():
            if (dst / marker).exists():
                print(f"    ✓ Already present: {dst.name}")
                continue
            if not os.path.exists(zip_src):
                print(f"    [!] Zip not found on Drive: {zip_src}")
                print(f"        --> Manually upload {Path(zip_src).name} to Drive/{DRIVE_DIR.split('/')[-1]}/")
                continue
            print(f"    Unzipping {Path(zip_src).name} -> {dst} ...")
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_src) as z:
                z.extractall(dst)
            print(f"    ✓ Done: {dst}")
    else:
        print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CHECKPOINTS NOT FOUND — Manual steps required:                 ║
    ║                                                                  ║
    ║  Option A — Upload from your laptop:                            ║
    ║    1. In Colab sidebar → Files icon → Upload                    ║
    ║       - blip_best_model.zip      → /content/                   ║
    ║       - clip_gpt2_best_model.zip → /content/                   ║
    ║       - daraz_dataset_colab.zip  → /content/                   ║
    ║    2. Then run this cell in Colab to unzip:                     ║
    ║       import zipfile                                             ║
    ║       for f, d in [                                              ║
    ║           ("blip_best_model.zip",      "/content/checkpoints/blip/best_model"),       ║
    ║           ("clip_gpt2_best_model.zip", "/content/checkpoints/clip_gpt2/best_model"),  ║
    ║           ("daraz_dataset_colab.zip",  "/content/daraz_data"),                        ║
    ║       ]:                                                         ║
    ║           with zipfile.ZipFile(f"/content/{f}") as z:           ║
    ║               z.extractall(d)                                    ║
    ║    3. Re-run this cell after unzipping.                          ║
    ║                                                                  ║
    ║  Option B — Re-run training in a new session:                   ║
    ║    Follow COLAB_GUIDE.md from Step 2 onwards.                   ║
    ╚══════════════════════════════════════════════════════════════════╝
        """)
        sys.exit("Stopping — restore checkpoints first, then re-run.")

# Final check — abort clearly if still missing
still_missing = check_missing()
if still_missing:
    for dst in still_missing:
        _, marker = RESTORE_MAP[dst]
        print(f"[!] Still missing after restore attempt: {dst / marker}")
    sys.exit("Cannot proceed without checkpoints. See instructions above.")

print("\n✓ All checkpoints and data are present. Starting evaluation...\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN IMPORTS (after restore check)
# ══════════════════════════════════════════════════════════════════════════════

import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer,
)
from IPython.display import display, HTML, Image as IPImage
import base64
import io

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT       = Path("/content/daraz_data")
METADATA_FILE   = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_DIR      = DATA_ROOT / "images"
TEST_SPLIT_FILE = DATA_ROOT / "splits" / "test.txt"

_BLIP_CANDIDATES = [
    Path("/content/checkpoints/blip/best_model"),
    Path("/content/blip_best_model"),
]
_CLIP_CANDIDATES = [
    Path("/content/checkpoints/clip_gpt2/best_model"),
    Path("/content/clip_gpt2_best_model"),
]
BLIP_CKPT = next((p for p in _BLIP_CANDIDATES if p.exists()), _BLIP_CANDIDATES[0])
CLIP_CKPT = next((p for p in _CLIP_CANDIDATES if p.exists()), _CLIP_CANDIDATES[0])

PREFIX_LENGTH = 10
MAX_TEXT_LEN  = 64
NUM_SAMPLES   = 5          # how many products to show in report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# CLIP-GPT2 MODEL CLASS (must match training)
# ══════════════════════════════════════════════════════════════════════════════

class ClipGPT2Model(nn.Module):
    def __init__(self, prefix_length=PREFIX_LENGTH):
        super().__init__()
        self.prefix_length = prefix_length
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_dim  = self.clip.config.hidden_size
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_dim  = self.gpt2.config.n_embd
        self.gpt2_embed_dim = gpt2_dim
        self.visual_projection = nn.Sequential(
            nn.Linear(clip_dim, gpt2_dim * prefix_length), nn.Tanh())

    def get_visual_prefix(self, pixel_values):
        out = self.clip(pixel_values=pixel_values)
        cls = out.pooler_output
        return self.visual_projection(cls).view(-1, self.prefix_length, self.gpt2_embed_dim)

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask,
                 max_new_tokens=120, num_beams=4, no_repeat_ngram_size=3):
        prefix  = self.get_visual_prefix(pixel_values)
        embeds  = self.gpt2.transformer.wte(input_ids)
        combined = torch.cat([prefix, embeds], dim=1)
        prefix_mask = torch.ones(input_ids.size(0), self.prefix_length, device=pixel_values.device)
        full_attn   = torch.cat([prefix_mask, attention_mask], dim=1)
        return self.gpt2.generate(
            inputs_embeds=combined, attention_mask=full_attn,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True, pad_token_id=self.gpt2.config.eos_token_id)


CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std= [0.26862954, 0.26130258, 0.27577711]),
])

# ══════════════════════════════════════════════════════════════════════════════
# LOAD BOTH MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading BLIP...")
blip_processor = BlipProcessor.from_pretrained(str(BLIP_CKPT))
blip_model = BlipForConditionalGeneration.from_pretrained(
    str(BLIP_CKPT), torch_dtype=torch.float16 if device.type=="cuda" else torch.float32
).to(device).eval()
print("  BLIP loaded ✓")

print("Loading CLIP-GPT2...")
clip_tokenizer = GPT2Tokenizer.from_pretrained(str(CLIP_CKPT))
clip_tokenizer.pad_token = clip_tokenizer.eos_token
clip_model = ClipGPT2Model()
clip_model.load_state_dict(torch.load(CLIP_CKPT / "model.pt", map_location="cpu"))
clip_model = clip_model.to(device).eval()
print("  CLIP-GPT2 loaded ✓")

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_metadata_prompt(rec):
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    for k, v in list((rec.get("attributes") or {}).items())[:3]:
        if k and v: parts.append(f"{k}: {v}")
    return ". ".join(parts)


def load_image(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try: return Image.open(p).convert("RGB")
            except: pass
    return Image.new("RGB", (224, 224), (200, 200, 200))


def generate_blip(rec, image):
    prompt = build_metadata_prompt(rec)
    inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            ids = blip_model.generate(**inputs, max_new_tokens=120,
                                      num_beams=4, no_repeat_ngram_size=3)
    return blip_processor.decode(ids[0], skip_special_tokens=True).strip()


def generate_clip_gpt2(rec, image):
    pixel_values = CLIP_TRANSFORM(image).unsqueeze(0).to(device)
    enc = clip_tokenizer(build_metadata_prompt(rec), max_length=MAX_TEXT_LEN,
                         truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            ids = clip_model.generate(
                pixel_values, enc["input_ids"].to(device), enc["attention_mask"].to(device))
    return clip_tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def image_to_base64(pil_image, max_size=(300, 300)):
    pil_image.thumbnail(max_size, Image.LANCZOS)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def wrap_text(text, max_chars=120):
    """Add line breaks for display."""
    words = text.split()
    lines, cur = [], []
    for w in words:
        cur.append(w)
        if sum(len(x)+1 for x in cur) > max_chars:
            lines.append(" ".join(cur[:-1]))
            cur = [w]
    if cur: lines.append(" ".join(cur))
    return "<br>".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# PICK 5 DIVERSE TEST SAMPLES
# ══════════════════════════════════════════════════════════════════════════════

test_ids = set(TEST_SPLIT_FILE.read_text().strip().splitlines())
all_test_records = []
with open(METADATA_FILE, encoding="utf-8") as f:
    for line in f:
        try: rec = json.loads(line.strip())
        except: continue
        if rec.get("item_id") not in test_ids: continue
        if not rec.get("images") or not rec.get("description","").strip(): continue
        all_test_records.append(rec)

# Try to pick one from each unique category, fall back to first 5
seen_cats = set()
selected = []
for rec in all_test_records:
    cat = rec.get("category", "unknown")
    if cat not in seen_cats:
        seen_cats.add(cat)
        selected.append(rec)
    if len(selected) >= NUM_SAMPLES:
        break
# Fill up to NUM_SAMPLES if categories exhausted
if len(selected) < NUM_SAMPLES:
    for rec in all_test_records:
        if rec not in selected:
            selected.append(rec)
        if len(selected) >= NUM_SAMPLES:
            break

print(f"\nSelected {len(selected)} samples from categories: {[r.get('category','?') for r in selected]}")

# ══════════════════════════════════════════════════════════════════════════════
# GENERATE AND DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

html_blocks = []

for i, rec in enumerate(selected, 1):
    print(f"\n[{i}/{len(selected)}] Generating for: {rec.get('item_name','?')[:60]}...")

    image = load_image(rec)
    blip_out    = generate_blip(rec, image)
    clip_out    = generate_clip_gpt2(rec, image)
    reference   = rec["description"].strip()
    metadata    = build_metadata_prompt(rec)
    img_b64     = image_to_base64(image)

    # ── Print to console (easy to read in Colab) ──────────────────────────────
    print(f"  Metadata:   {metadata[:100]}...")
    print(f"  Reference:  {reference[:120]}...")
    print(f"  BLIP:       {blip_out[:120]}...")
    print(f"  CLIP-GPT2:  {clip_out[:120]}...")

    # ── Build HTML block for this sample ─────────────────────────────────────
    block = f"""
    <div style="border:2px solid #333; border-radius:10px; margin:20px 0; padding:20px;
                font-family:monospace; background:#1a1a2e; color:#eee;">

      <h3 style="color:#00d4ff; margin:0 0 12px 0;">
        Sample {i} — {rec.get('category', 'Unknown Category')}
      </h3>

      <div style="display:flex; gap:20px; align-items:flex-start;">

        <!-- Image -->
        <div style="flex-shrink:0;">
          <img src="data:image/png;base64,{img_b64}"
               style="width:240px; border-radius:8px; border:1px solid #555;" />
        </div>

        <!-- Text columns -->
        <div style="flex:1; font-size:13px; line-height:1.7;">

          <div style="margin-bottom:10px;">
            <span style="color:#aaa; font-weight:bold;">📋 METADATA INPUT</span><br>
            <span style="color:#ccc;">{wrap_text(metadata, 100)}</span>
          </div>

          <div style="margin-bottom:10px; background:#0f3460; padding:10px; border-radius:6px;">
            <span style="color:#ffd700; font-weight:bold;">📖 REFERENCE DESCRIPTION</span><br>
            <span style="color:#e0e0e0;">{wrap_text(reference[:400] + ("..." if len(reference)>400 else ""), 100)}</span>
          </div>

          <div style="margin-bottom:10px; background:#162447; padding:10px; border-radius:6px;">
            <span style="color:#4fc3f7; font-weight:bold;">🔵 BLIP Fine-tuned</span><br>
            <span style="color:#e0e0e0;">{wrap_text(blip_out, 100)}</span>
          </div>

          <div style="background:#1b2838; padding:10px; border-radius:6px;">
            <span style="color:#a5d6a7; font-weight:bold;">🟢 CLIP-GPT2 Fine-tuned</span><br>
            <span style="color:#e0e0e0;">{wrap_text(clip_out, 100)}</span>
          </div>

        </div>
      </div>
    </div>
    """
    html_blocks.append(block)

    # Also display inline (shows each one as it's generated)
    display(HTML(block))


# ── Save combined HTML file and download ──────────────────────────────────────
full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Qualitative Evaluation — Daraz VLM</title>
  <style>
    body {{ background: #0d0d1a; color: #eee; font-family: monospace; padding: 30px; }}
    h1   {{ color: #00d4ff; text-align: center; }}
    h2   {{ color: #aaa;    text-align: center; font-size: 14px; margin-top: -10px; }}
  </style>
</head>
<body>
  <h1>Qualitative Evaluation — Product Description Generation</h1>
  <h2>Daraz.pk Dataset &nbsp;|&nbsp; BLIP vs CLIP-GPT2 &nbsp;|&nbsp; Section 6.6</h2>
  {''.join(html_blocks)}
</body>
</html>"""

report_path = Path("/content/qualitative_report.html")
report_path.write_text(full_html, encoding="utf-8")
print(f"\n✓ HTML report saved: {report_path}")

from google.colab import files
files.download(str(report_path))
print("Downloaded qualitative_report.html — open in browser and take screenshots!")

# Also save to Drive if mounted
try:
    import shutil
    shutil.copy(str(report_path), "/content/drive/MyDrive/daraz_cv_project/qualitative_report.html")
    print("Backed up to Drive!")
except Exception as e:
    print(f"Drive backup skipped: {e}")
