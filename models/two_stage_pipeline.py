"""
Two-Stage Product Description Pipeline.

Stage 1 — Fine-tuned VLM (BLIP or CLIP-GPT2):
    Input : product image + category/subcategory only
    Output: visually-grounded description (model forced to use the image)
    Judge : Gemma 4 31B via OpenRouter scores the Stage 1 output on
            visual_grounding, fluency, and relevance (1-5 each)

Stage 2 — Fine-tuned GPT-2 (extracted from CLIP-GPT2 checkpoint):
    Input : Stage 1 description + full product metadata (text only)
    Output: polished description that blends visual detail with learned style

All three artefacts are saved per item. Resumable.

Run:
    python -m models.two_stage_pipeline --model blip
    python -m models.two_stage_pipeline --model clip_gpt2
    python -m models.two_stage_pipeline --model blip --max-samples 20

Output:
    models/results/two_stage_results.jsonl   — one JSON line per item:
        item_id, category, reference,
        description_stage1, stage1_scores,
        description_stage2
    models/results/two_stage_metrics.json    — NLP metrics for both stages

Note on retraining:
    For best Stage 1 quality, retrain the VLM using build_stage1_prompt()
    (category + subcategory only). With the original full-metadata checkpoint
    Stage 1 still runs but may lean on metadata it was trained with.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR,
    build_stage1_prompt,
)
from models.shared.metrics import compute_all_metrics, print_metrics_table, save_metrics
from models.stage2.openrouter_refiner import make_client, score as judge_score
from models.stage2.gpt2_refiner import load_gpt2, refine as gpt2_refine

RESULTS_FILE = RESULTS_DIR / "two_stage_results.jsonl"
METRICS_FILE = RESULTS_DIR / "two_stage_metrics.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32

JUDGE_DELAY  = 0.5   # seconds between OpenRouter calls


# ── Shared helpers ─────────────────────────────────────────────────────────────

def load_test_records(max_samples: int) -> list[dict]:
    test_ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").strip().splitlines())
    records: list[dict] = []
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


def load_done(output_file: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not output_file.exists():
        return done
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                done[e["item_id"]] = e
            except Exception:
                pass
    return done


def load_pil_image(rec: dict) -> Image.Image:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


# ── BLIP Stage 1 ──────────────────────────────────────────────────────────────

def load_blip(checkpoint_path: Path):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print(f"  Loading BLIP from {checkpoint_path} ...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"[!] Checkpoint not found: {checkpoint_path}\n"
            "    Download best_model/ from Colab and place it at the path above."
        )
    processor = BlipProcessor.from_pretrained(str(checkpoint_path))
    model = BlipForConditionalGeneration.from_pretrained(
        str(checkpoint_path), torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    print(f"  BLIP loaded ({DEVICE})")
    return processor, model


@torch.no_grad()
def blip_generate(processor, model, rec: dict, max_new_tokens: int = 200) -> str:
    image  = load_pil_image(rec)
    prompt = build_stage1_prompt(rec)
    inputs = {k: v.to(DEVICE) for k, v in
              processor(images=image, text=prompt, return_tensors="pt").items()}
    with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return processor.decode(ids[0], skip_special_tokens=True).strip()


# ── CLIP-GPT2 Stage 1 ─────────────────────────────────────────────────────────

def load_clip_gpt2(checkpoint_path: Path):
    from torchvision import transforms
    from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer

    PREFIX_LENGTH = 10

    clip_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    class _ClipGPT2(nn.Module):
        def __init__(self):
            super().__init__()
            self.prefix_length = PREFIX_LENGTH
            self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
            self.visual_projection = nn.Sequential(
                nn.Linear(self.clip.config.hidden_size,
                          self.gpt2.config.n_embd * PREFIX_LENGTH),
                nn.Tanh(),
            )

        def get_visual_prefix(self, pixel_values):
            out = self.clip(pixel_values=pixel_values)
            flat = self.visual_projection(out.pooler_output)
            return flat.view(-1, self.prefix_length, self.gpt2.config.n_embd)

        @torch.no_grad()
        def generate(self, pixel_values, input_ids, attention_mask,
                     max_new_tokens=150, num_beams=4, no_repeat_ngram_size=3):
            prefix = self.get_visual_prefix(pixel_values)
            prompt_embeds = self.gpt2.transformer.wte(input_ids)
            combined = torch.cat([prefix, prompt_embeds], dim=1)
            prefix_mask = torch.ones(input_ids.size(0), self.prefix_length)
            full_attn = torch.cat([prefix_mask, attention_mask], dim=1)
            return self.gpt2.generate(
                inputs_embeds=combined,
                attention_mask=full_attn,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
                pad_token_id=self.gpt2.config.eos_token_id,
            )

    print(f"  Loading CLIP-GPT2 from {checkpoint_path} ...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"[!] Checkpoint not found: {checkpoint_path}\n"
            "    Download best_model/ from Colab and place it at the path above."
        )

    model = _ClipGPT2()
    weights = checkpoint_path / "model.pt"
    if not weights.exists():
        raise FileNotFoundError(f"[!] model.pt not found in {checkpoint_path}")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.to(DEVICE).eval()
    print(f"  CLIP-GPT2 loaded ({DEVICE})")

    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model, clip_transform


@torch.no_grad()
def clip_gpt2_generate(tokenizer, model, clip_transform, rec: dict,
                        max_new_tokens: int = 150) -> str:
    image = load_pil_image(rec)
    pixel_values = clip_transform(image).unsqueeze(0).to(DEVICE)
    prompt = build_stage1_prompt(rec)
    enc = tokenizer(prompt, max_length=64, truncation=True,
                    padding="max_length", return_tensors="pt")
    ids = model.generate(
        pixel_values=pixel_values,
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE),
        max_new_tokens=max_new_tokens,
        num_beams=4,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _find_checkpoint(candidates: list[Path]) -> Path:
    found = next((p for p in candidates if p.exists()), None)
    return found or candidates[0]


def run(model_name: str, max_samples: int) -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    _CKPTS    = PROJECT_ROOT / "models" / "checkpoints"
    _BLIP_DIR = PROJECT_ROOT / "models" / "blip"
    _CG2_DIR  = PROJECT_ROOT / "models" / "clip_gpt2"

    # ── Load Stage 1 VLM ──────────────────────────────────────────────────────
    if model_name == "blip":
        ckpt = _find_checkpoint([
            _BLIP_DIR / "blip_best_model",
            _CKPTS / "blip" / "best_model",
            _BLIP_DIR / "best_model",
        ])
        processor, vlm = load_blip(ckpt)
        def stage1_fn(rec):
            return blip_generate(processor, vlm, rec)
    else:
        ckpt = _find_checkpoint([
            _CG2_DIR / "clip_gpt2_best_model",
            _CKPTS / "clip_gpt2" / "best_model",
            _CG2_DIR / "best_model",
        ])
        tokenizer_vlm, vlm, clip_tf = load_clip_gpt2(ckpt)
        def stage1_fn(rec):
            return clip_gpt2_generate(tokenizer_vlm, vlm, clip_tf, rec)

    # ── Load Stage 2 GPT-2 (fine-tuned, text-only) ────────────────────────────
    gpt2_tok, gpt2_model = load_gpt2()

    # ── Load Gemma judge (OpenRouter) ─────────────────────────────────────────
    judge_client = make_client()

    # ── Data ──────────────────────────────────────────────────────────────────
    records = load_test_records(max_samples)
    done    = load_done(RESULTS_FILE)
    todo    = [r for r in records if r["item_id"] not in done]

    print(f"  Already done : {len(done)}")
    print(f"  Remaining    : {len(todo)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    stage1_hyps: list[str] = [d["description_stage1"] for d in done.values()]
    stage2_hyps: list[str] = [d["description_stage2"] for d in done.values()]
    refs:        list[str] = [d["reference"]           for d in done.values()]

    t0 = time.time()
    with open(RESULTS_FILE, "a", encoding="utf-8") as out_f:
        for rec in tqdm(todo, desc="Two-stage inference", unit="item"):
            # ── Stage 1: VLM (image + category only) ──────────────────────────
            desc1 = stage1_fn(rec)

            # ── Judge: Gemma scores Stage 1 ───────────────────────────────────
            scores = judge_score(judge_client, desc1, rec.get("category", ""))
            time.sleep(JUDGE_DELAY)

            # ── Stage 2: fine-tuned GPT-2 (Stage 1 + full metadata) ───────────
            desc2 = gpt2_refine(gpt2_tok, gpt2_model, rec, desc1)

            reference = rec["description"].strip()

            entry = {
                "item_id":            rec["item_id"],
                "category":           rec.get("category", ""),
                "reference":          reference,
                "description_stage1": desc1,
                "stage1_scores":      scores,
                "description_stage2": desc2,
            }
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out_f.flush()

            stage1_hyps.append(desc1)
            stage2_hyps.append(desc2)
            refs.append(reference)

    elapsed = time.time() - t0
    n = max(len(stage1_hyps), 1)
    print(f"\n  Done in {elapsed/60:.1f} min ({elapsed/n:.1f}s/item)")

    # ── NLP metrics ───────────────────────────────────────────────────────────
    label1 = f"{model_name.upper()} Stage 1 (visual only)"
    label2 = "Stage 2 — fine-tuned GPT-2"

    results = {}
    if stage1_hyps:
        results[label1] = compute_all_metrics(stage1_hyps, refs)
    if stage2_hyps:
        results[label2] = compute_all_metrics(stage2_hyps, refs)

    print_metrics_table(results)
    save_metrics({k: v for d in results.values() for k, v in d.items()}, str(METRICS_FILE))

    # ── LLM judge summary ─────────────────────────────────────────────────────
    all_scores = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                    if e.get("stage1_scores"):
                        all_scores.append(e["stage1_scores"])
                except Exception:
                    pass

    if all_scores:
        avg = lambda key: round(sum(s[key] for s in all_scores if s) / len(all_scores), 2)
        print(f"\n  Gemma judge averages (n={len(all_scores)}):")
        print(f"    visual_grounding : {avg('visual_grounding')}")
        print(f"    fluency          : {avg('fluency')}")
        print(f"    relevance        : {avg('relevance')}")
        print(f"    overall          : {avg('overall')}")

    print(f"\n  Results : {RESULTS_FILE}")
    print(f"  Metrics : {METRICS_FILE}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage VLM + GPT-2 pipeline")
    parser.add_argument(
        "--model", choices=["blip", "clip_gpt2"], default="blip",
        help="Which fine-tuned VLM to use for Stage 1 (default: blip)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=150,
        help="Maximum test items to process (default: 150)",
    )
    args = parser.parse_args()
    run(args.model, args.max_samples)
