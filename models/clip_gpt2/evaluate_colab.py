"""
CLIP-GPT2 Evaluation — Self-contained Colab/Kaggle cell.

Paste the ENTIRE content of this file into ONE Colab cell and run it.
Run AFTER the BLIP evaluation cell (metrics module is already defined there,
but this cell redefines everything so it's fully standalone).

Prerequisites:
  - Cell 0A or 0B from eval_setup.py (data + checkpoints accessible)
  - Cell 1 from eval_setup.py  (nltk + rouge-score installed)
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer

# ── PATHS — edit if using Kaggle (/tmp/...) ───────────────────────────────────
DATA_ROOT       = Path("/content/daraz_data")
METADATA_FILE   = DATA_ROOT / "metadata" / "listings_final.jsonl"
IMAGES_DIR      = DATA_ROOT / "images"
TEST_SPLIT_FILE = DATA_ROOT / "splits" / "test.txt"
RESULTS_DIR     = Path("/content/results")

_CLIP_CANDIDATES = [
    Path("/content/checkpoints/clip_gpt2/best_model"),   # same session
    Path("/content/clip_gpt2_best_model"),               # if unzipped directly
]
CHECKPOINT_DIR = next((p for p in _CLIP_CANDIDATES if p.exists()), _CLIP_CANDIDATES[0])

MAX_SAMPLES   = 150
PREFIX_LENGTH = 10      # must match training config
MAX_TEXT_LEN  = 64      # must match training config

print(f"Checkpoint: {CHECKPOINT_DIR}")
print(f"Exists:     {CHECKPOINT_DIR.exists()}")


# ── CLIP image transform ──────────────────────────────────────────────────────
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std= [0.26862954, 0.26130258, 0.27577711]),
])


# ── Model (must exactly match the architecture used in train_colab.py) ────────

class ClipGPT2Model(nn.Module):
    def __init__(self, prefix_length: int = PREFIX_LENGTH):
        super().__init__()
        self.prefix_length = prefix_length
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_dim  = self.clip.config.hidden_size      # 768

        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_dim  = self.gpt2.config.n_embd           # 768

        self.visual_projection = nn.Sequential(
            nn.Linear(clip_dim, gpt2_dim * prefix_length),
            nn.Tanh(),
        )
        self.gpt2_embed_dim = gpt2_dim

    def get_visual_prefix(self, pixel_values):
        out = self.clip(pixel_values=pixel_values)
        cls = out.pooler_output                                            # (B,768)
        return self.visual_projection(cls).view(-1, self.prefix_length, self.gpt2_embed_dim)

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask,
                 max_new_tokens=150, num_beams=4, no_repeat_ngram_size=3):
        prefix  = self.get_visual_prefix(pixel_values)
        embeds  = self.gpt2.transformer.wte(input_ids)
        combined = torch.cat([prefix, embeds], dim=1)

        prefix_mask = torch.ones(input_ids.size(0), self.prefix_length, device=pixel_values.device)
        full_attn   = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.gpt2.generate(
            inputs_embeds=combined, attention_mask=full_attn,
            max_new_tokens=max_new_tokens, num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True, pad_token_id=self.gpt2.config.eos_token_id,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_metadata_prompt(rec: dict) -> str:
    parts = []
    if rec.get("item_name"):   parts.append(f"Product: {rec['item_name']}")
    if rec.get("brand"):       parts.append(f"Brand: {rec['brand']}")
    if rec.get("category"):    parts.append(f"Category: {rec['category']}")
    if rec.get("price_pkr"):   parts.append(f"Price: PKR {rec['price_pkr']:.0f}")
    for k, v in list((rec.get("attributes") or {}).items())[:3]:
        if k and v: parts.append(f"{k}: {v}")
    return ". ".join(parts)


def load_test_records(max_samples: int) -> list[dict]:
    test_ids = set(TEST_SPLIT_FILE.read_text().strip().splitlines())
    records  = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            try: rec = json.loads(line.strip())
            except: continue
            if rec.get("item_id") not in test_ids: continue
            if not rec.get("images") or not rec.get("description", "").strip(): continue
            records.append(rec)
            if len(records) >= max_samples: break
    print(f"Loaded {len(records)} test records")
    return records


def load_image_tensor(rec: dict, device) -> torch.Tensor:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                img = Image.open(p).convert("RGB")
                return CLIP_TRANSFORM(img).unsqueeze(0).to(device)
            except: pass
    return CLIP_TRANSFORM(Image.new("RGB", (224,224),(255,255,255))).unsqueeze(0).to(device)


def compute_metrics(hypotheses: list[str], references: list[str]) -> dict:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rouge_lib

    smooth   = SmoothingFunction().method4
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]

    bleu1 = corpus_bleu(refs_tok, hyps_tok, weights=(1,0,0,0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tok, hyps_tok, weights=(.25,.25,.25,.25), smoothing_function=smooth)

    scorer  = rouge_lib.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r,h)["rougeL"].fmeasure for r,h in zip(references,hypotheses)) / len(references)
    meteor  = sum(meteor_score([r.split()], h.split()) for r,h in zip(references,hypotheses)) / len(references)

    return {
        "BLEU-1":  round(bleu1  * 100, 2),
        "BLEU-4":  round(bleu4  * 100, 2),
        "ROUGE-L": round(rouge_l * 100, 2),
        "METEOR":  round(meteor  * 100, 2),
    }


def print_table(model_name: str, metrics: dict):
    header = f"{'Model':<35}" + "".join(f"{k:>10}" for k in metrics)
    row    = f"{model_name:<35}" + "".join(f"{v:>10.2f}" for v in metrics.values())
    print("\n" + "=" * len(header))
    print(header + "\n" + "-" * len(header))
    print(row + "\n" + "=" * len(header))


# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

print(f"\nLoading CLIP-GPT2 from {CHECKPOINT_DIR} ...")
model = ClipGPT2Model(prefix_length=PREFIX_LENGTH)
state_dict = torch.load(CHECKPOINT_DIR / "model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print("Model loaded!")

tokenizer = GPT2Tokenizer.from_pretrained(str(CHECKPOINT_DIR))
tokenizer.pad_token = tokenizer.eos_token


# ── Evaluation loop ───────────────────────────────────────────────────────────

records    = load_test_records(MAX_SAMPLES)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_file = RESULTS_DIR / "clip_gpt2_results.jsonl"

hypotheses: list[str] = []
references: list[str] = []

t0 = time.time()
with open(results_file, "w", encoding="utf-8") as out_f:
    for rec in tqdm(records, desc="CLIP-GPT2 inference", unit="item"):
        pixel_values = load_image_tensor(rec, device)

        prompt_enc = tokenizer(
            build_metadata_prompt(rec),
            max_length=MAX_TEXT_LEN, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        input_ids      = prompt_enc["input_ids"].to(device)
        attention_mask = prompt_enc["attention_mask"].to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast():
                    output_ids = model.generate(pixel_values, input_ids, attention_mask)
            else:
                output_ids = model.generate(pixel_values, input_ids, attention_mask)

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        reference = rec["description"].strip()

        entry = {"item_id": rec["item_id"], "category": rec.get("category",""),
                 "generated": generated, "reference": reference}
        out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_f.flush()

        hypotheses.append(generated)
        references.append(reference)

elapsed = time.time() - t0
print(f"\nDone in {elapsed/60:.1f} min  ({elapsed/len(hypotheses):.1f}s/sample)")

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = compute_metrics(hypotheses, references)
print_table("CLIP-GPT2 Fine-tuned", metrics)

metrics_file = RESULTS_DIR / "clip_gpt2_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nSaved results -> {results_file}")
print(f"Saved metrics -> {metrics_file}")

# ── Download to laptop ────────────────────────────────────────────────────────
from google.colab import files
files.download(str(metrics_file))
files.download(str(results_file))

try:
    import shutil
    shutil.copy(str(metrics_file),  "/content/drive/MyDrive/daraz_cv_project/clip_gpt2_metrics.json")
    shutil.copy(str(results_file),  "/content/drive/MyDrive/daraz_cv_project/clip_gpt2_results.jsonl")
    print("Backed up to Drive!")
except Exception as e:
    print(f"Drive backup skipped: {e}")
