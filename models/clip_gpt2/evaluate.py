"""
CLIP-GPT2 Local Inference — runs on your 8GB RAM laptop (CPU only).

Checkpoint format:
  - CLIP-GPT2 saves a raw PyTorch state dict (model.pt) + GPT-2 tokenizer files
  - This is different from BLIP which saves in HuggingFace format
  - We reconstruct the ClipGPT2Model class here and load the weights manually

Run:
    python -m models.clip_gpt2.evaluate
    python -m models.clip_gpt2.evaluate --max-samples 100
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT,
    RESULTS_DIR, build_metadata_prompt
)
from models.shared.metrics import compute_all_metrics, print_metrics_table, save_metrics

# ── Checkpoint location (checks both possible paths) ──────────────────────────
_HERE = Path(__file__).resolve().parent       # models/clip_gpt2/
_ROOT = _HERE.parent                          # models/

_CANDIDATE_PATHS = [
    _HERE / "clip_gpt2_best_model",           # models/clip_gpt2/clip_gpt2_best_model/
    _ROOT / "checkpoints" / "clip_gpt2" / "best_model",
    _HERE / "best_model",
]
CHECKPOINT_DIR = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[0])
RESULTS_FILE   = RESULTS_DIR / "clip_gpt2_results.jsonl"

# ── Must match the values used during training ─────────────────────────────────
CLIP_MODEL    = "openai/clip-vit-base-patch32"
GPT2_MODEL    = "gpt2"
PREFIX_LENGTH = 10
MAX_TEXT_LEN  = 64

CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std= [0.26862954, 0.26130258, 0.27577711],
    ),
])


# ── Model definition (must match train_colab.py exactly) ─────────────────────

class ClipGPT2Model(nn.Module):
    def __init__(self, prefix_length: int = PREFIX_LENGTH):
        super().__init__()
        self.prefix_length = prefix_length
        self.clip = CLIPVisionModel.from_pretrained(CLIP_MODEL)
        self.clip_embed_dim = self.clip.config.hidden_size       # 768

        self.gpt2 = GPT2LMHeadModel.from_pretrained(GPT2_MODEL)
        self.gpt2_embed_dim = self.gpt2.config.n_embd            # 768

        self.visual_projection = nn.Sequential(
            nn.Linear(self.clip_embed_dim, self.gpt2_embed_dim * prefix_length),
            nn.Tanh(),
        )

    def get_visual_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out           = self.clip(pixel_values=pixel_values)
        cls_embedding = out.pooler_output                         # (B, 768)
        prefix_flat   = self.visual_projection(cls_embedding)     # (B, P*768)
        return prefix_flat.view(-1, self.prefix_length, self.gpt2_embed_dim)

    @torch.no_grad()
    def generate(
        self,
        pixel_values:         torch.Tensor,
        input_ids:            torch.Tensor,
        attention_mask:       torch.Tensor,
        max_new_tokens:       int = 150,
        num_beams:            int = 4,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        visual_prefix = self.get_visual_prefix(pixel_values)
        prompt_embeds = self.gpt2.transformer.wte(input_ids)
        combined      = torch.cat([visual_prefix, prompt_embeds], dim=1)

        prefix_mask = torch.ones(input_ids.size(0), self.prefix_length,
                                 device=pixel_values.device)
        full_attn   = torch.cat([prefix_mask, attention_mask], dim=1)

        return self.gpt2.generate(
            inputs_embeds=combined,
            attention_mask=full_attn,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            pad_token_id=self.gpt2.config.eos_token_id,
        )


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: Path):
    """Load CLIP-GPT2 from raw model.pt state dict."""
    print(f"\n  Loading CLIP-GPT2 from {checkpoint_path}...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\n[!] Checkpoint not found at {checkpoint_path}\n"
            "    Download clip_gpt2_best_model/ from Colab and place it inside models/clip_gpt2/"
        )

    weights_file = checkpoint_path / "model.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"[!] model.pt not found in {checkpoint_path}")

    model = ClipGPT2Model(prefix_length=PREFIX_LENGTH)
    state_dict = torch.load(weights_file, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print("  Model loaded (CPU mode)")

    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def load_test_records(max_samples: int = 150) -> list[dict]:
    test_ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").strip().splitlines())
    records  = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if rec.get("item_id") not in test_ids:
                continue
            if not rec.get("images") or not rec.get("description", "").strip():
                continue
            records.append(rec)
            if len(records) >= max_samples:
                break
    print(f"  Loaded {len(records)} test records")
    return records


def load_image_tensor(rec: dict) -> torch.Tensor:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return CLIP_TRANSFORM(Image.open(p).convert("RGB")).unsqueeze(0)  # (1,3,224,224)
            except Exception:
                pass
    return CLIP_TRANSFORM(Image.new("RGB", (224, 224), (255, 255, 255))).unsqueeze(0)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_evaluation(max_samples: int = 150) -> None:
    tokenizer, model = load_model(CHECKPOINT_DIR)
    records          = load_test_records(max_samples=max_samples)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    done_ids:   set[str]  = set()
    hypotheses: list[str] = []
    references: list[str] = []

    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    done_ids.add(e["item_id"])
                    hypotheses.append(e["generated"])
                    references.append(e["reference"])
                except Exception:
                    pass
        print(f"  Resuming — {len(done_ids)} already evaluated")

    t0 = time.time()
    with open(RESULTS_FILE, "a", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="CLIP-GPT2 inference [CPU]", unit="item"):
            if rec["item_id"] in done_ids:
                continue

            pixel_values = load_image_tensor(rec)

            metadata = build_metadata_prompt(rec)
            prompt_enc = tokenizer(
                metadata,
                max_length=MAX_TEXT_LEN,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            with torch.no_grad():
                output_ids = model.generate(
                    pixel_values=pixel_values,
                    input_ids=prompt_enc["input_ids"],
                    attention_mask=prompt_enc["attention_mask"],
                    max_new_tokens=150,
                    num_beams=4,
                    no_repeat_ngram_size=3,
                )

            # Decode only the newly generated tokens
            generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            reference = rec["description"].strip()

            entry = {
                "item_id":   rec["item_id"],
                "category":  rec.get("category", ""),
                "generated": generated,
                "reference": reference,
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()

            hypotheses.append(generated)
            references.append(reference)

    elapsed = time.time() - t0
    print(f"\n  Inference done in {elapsed/60:.1f} min ({elapsed/max(len(hypotheses),1):.1f}s/sample)")

    metrics = compute_all_metrics(hypotheses, references)
    print_metrics_table({"CLIP-GPT2 Fine-tuned": metrics})
    save_metrics(metrics, str(RESULTS_DIR / "clip_gpt2_metrics.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--checkpoint",  type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()
    CHECKPOINT_DIR = Path(args.checkpoint)
    run_evaluation(max_samples=args.max_samples)
