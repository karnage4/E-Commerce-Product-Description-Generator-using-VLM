"""
Inference hyperparameter sweep — no retraining required, runs on CPU.

Sweeps beam search parameters against the test set and reports ROUGE-L for
each configuration. Use this to find the best inference settings for an
already-trained checkpoint before doing any expensive training runs.

Sweeps (see INFERENCE_SWEEP in models/shared/config.py):
    num_beams            : [2, 4, 6]
    no_repeat_ngram_size : [2, 3, 4]
    max_new_tokens       : [100, 150, 200]

By default sweeps one parameter at a time (others held at baseline).
Use --full-grid to try all combinations (3×3×3 = 27 runs, ~5–8 hrs on CPU).

Run:
    python -m models.hparam_sweep --model blip
    python -m models.hparam_sweep --model clip_gpt2
    python -m models.hparam_sweep --model blip --max-samples 30   # quick smoke-test
    python -m models.hparam_sweep --model blip --full-grid
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR,
    INFERENCE_SWEEP, build_stage1_prompt,
)
from models.shared.metrics import rouge_l_score

SWEEP_OUT = RESULTS_DIR / "hparam_sweep.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32

# Baseline values held fixed when sweeping one parameter at a time
_BASELINE = {
    "num_beams":            4,
    "no_repeat_ngram_size": 3,
    "max_new_tokens":       150,
}


# ── Data ───────────────────────────────────────────────────────────────────────

def load_test_records(n: int) -> list[dict]:
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
            if len(records) >= n:
                break
    return records


def load_image(rec: dict) -> Image.Image:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


# ── BLIP inference ─────────────────────────────────────────────────────────────

def load_blip(ckpt: Path):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained(str(ckpt))
    model = BlipForConditionalGeneration.from_pretrained(
        str(ckpt), torch_dtype=DTYPE,
    ).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def run_blip(processor, model, records, cfg: dict) -> tuple[list[str], list[str]]:
    hyps, refs = [], []
    for rec in records:
        image  = load_image(rec)
        prompt = build_stage1_prompt(rec)
        inputs = {k: v.to(DEVICE) for k, v in
                  processor(images=image, text=prompt, return_tensors="pt").items()}
        with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
            ids = model.generate(
                **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            num_beams=cfg["num_beams"],
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            early_stopping=True,
        )
        hyps.append(processor.decode(ids[0], skip_special_tokens=True).strip())
        refs.append(rec["description"].strip())
    return hyps, refs


# ── CLIP-GPT2 inference ────────────────────────────────────────────────────────

def load_clip_gpt2(ckpt: Path):
    from torchvision import transforms
    from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Tokenizer

    PREFIX_LENGTH = 10
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275,  0.40821073],
            std= [0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    class _M(nn.Module):
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

        @torch.no_grad()
        def generate(self, pv, ids, mask, **kw):
            pfx = self.visual_projection(self.clip(pixel_values=pv).pooler_output)
            pfx = pfx.view(-1, self.prefix_length, self.gpt2.config.n_embd)
            emb = torch.cat([pfx, self.gpt2.transformer.wte(ids)], dim=1)
            am  = torch.cat([torch.ones(ids.size(0), self.prefix_length, device=pv.device), mask], dim=1)
            return self.gpt2.generate(
                inputs_embeds=emb, attention_mask=am,
                pad_token_id=self.gpt2.config.eos_token_id, **kw,
            )

    model = _M()
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location="cpu"))
    model.to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(str(ckpt))
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model, tf


@torch.no_grad()
def run_clip_gpt2(tokenizer, model, tf, records, cfg: dict) -> tuple[list[str], list[str]]:
    hyps, refs = [], []
    for rec in records:
        pv  = tf(load_image(rec)).unsqueeze(0).to(DEVICE)
        enc = tokenizer(
            build_stage1_prompt(rec), max_length=64,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        ids = model.generate(
            pv, enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE),
            max_new_tokens=cfg["max_new_tokens"],
            num_beams=cfg["num_beams"],
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            early_stopping=True,
        )
        hyps.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())
        refs.append(rec["description"].strip())
    return hyps, refs


# ── Sweep logic ────────────────────────────────────────────────────────────────

def _build_configs(full_grid: bool) -> list[dict]:
    if full_grid:
        keys   = list(INFERENCE_SWEEP.keys())
        combos = list(itertools.product(*[INFERENCE_SWEEP[k] for k in keys]))
        return [dict(zip(keys, c)) for c in combos]

    # One-at-a-time sweep
    configs = [_BASELINE.copy()]   # baseline first
    for param, values in INFERENCE_SWEEP.items():
        for v in values:
            if v == _BASELINE[param]:
                continue
            cfg = _BASELINE.copy()
            cfg[param] = v
            configs.append(cfg)
    # deduplicate preserving order
    seen, unique = set(), []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _find_ckpt(model_name: str) -> Path:
    root = Path(__file__).resolve().parent
    candidates = (
        [
            root / "blip" / "blip_best_model",
            root.parent / "models" / "checkpoints" / "blip" / "best_model",
            root / "blip" / "best_model",
        ]
        if model_name == "blip"
        else [
            root / "clip_gpt2" / "clip_gpt2_best_model",
            root.parent / "models" / "checkpoints" / "clip_gpt2" / "best_model",
            root / "clip_gpt2" / "best_model",
        ]
    )
    found = next((p for p in candidates if p.exists()), None)
    if found:
        return found
    raise FileNotFoundError(
        f"[!] No checkpoint found for {model_name}. "
        "Download best_model/ from Colab first."
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["blip", "clip_gpt2"], default="blip")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Test records to evaluate per config (default 50 ≈ ~5 min/config)")
    parser.add_argument("--full-grid",   action="store_true",
                        help="Try all parameter combinations (27 runs)")
    args = parser.parse_args()

    ckpt    = _find_ckpt(args.model)
    records = load_test_records(args.max_samples)
    configs = _build_configs(args.full_grid)

    print(f"\n  Model      : {args.model}")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Samples    : {len(records)}")
    print(f"  Configs    : {len(configs)}")
    print(f"  Mode       : {'full grid' if args.full_grid else 'one-at-a-time'}\n")

    if args.model == "blip":
        processor, model = load_blip(ckpt)
        def _run(cfg):
            return run_blip(processor, model, records, cfg)
    else:
        tokenizer, model, tf = load_clip_gpt2(ckpt)
        def _run(cfg):
            return run_clip_gpt2(tokenizer, model, tf, records, cfg)

    results = []
    col_w = 10

    header = (
        f"{'num_beams':>{col_w}}"
        f"{'ngram':>{col_w}}"
        f"{'max_tok':>{col_w}}"
        f"{'ROUGE-L':>{col_w}}"
        f"{'time(s)':>{col_w}}"
    )
    print(header)
    print("-" * len(header))

    for cfg in configs:
        t0 = time.time()
        hyps, refs = _run(cfg)
        rouge = rouge_l_score(hyps, refs)
        elapsed = round(time.time() - t0, 1)

        row = (
            f"{cfg['num_beams']:>{col_w}}"
            f"{cfg['no_repeat_ngram_size']:>{col_w}}"
            f"{cfg['max_new_tokens']:>{col_w}}"
            f"{rouge:>{col_w}.2f}"
            f"{elapsed:>{col_w}}"
        )
        # mark baseline
        is_baseline = cfg == _BASELINE
        print(row + ("  ← baseline" if is_baseline else ""))

        results.append({"config": cfg, "rouge_l": rouge, "time_s": elapsed})

    best = max(results, key=lambda r: r["rouge_l"])
    print(f"\n  Best config  : {best['config']}")
    print(f"  Best ROUGE-L : {best['rouge_l']:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_OUT, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"  Saved → {SWEEP_OUT}")


if __name__ == "__main__":
    main()
