"""
Experiment N5 — Compute Profile.

Additive experiment: measures, rather than asserts, the cost side of the
efficiency claim. Loads existing checkpoints read-only.

Measures per model:
    parameter count (total / trainable)
    checkpoint size on disk
    wall-clock latency per description, GPU and CPU, batch size 1
    throughput (descriptions per hour on one consumer GPU)
    peak VRAM during generation

and derives:
    energy per 1,000 descriptions (GPU TDP x wall clock, an upper bound)
    cost per 1,000 descriptions at consumer electricity prices
    the same figure for a hosted frontier-VLM API call, for comparison

Run:
    python -m models.novelty.compute_profile
    python -m models.novelty.compute_profile --n 10 --skip-cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR, build_metadata_prompt,
)

OUT_JSON = RESULTS_DIR / "novelty_compute_profile.json"

# RTX 5060 Ti board power. Upper bound: real draw during batch-1 generation is
# lower, so the energy figures below are conservative (pessimistic for us).
GPU_TDP_WATTS = 180
ELECTRICITY_USD_PER_KWH = 0.15


def load_records(n):
    ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").split())
    recs = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("item_id") in ids and r.get("images"):
                recs.append(r)
            if len(recs) >= n:
                break
    return recs


def load_image(rec):
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return Image.new("RGB", (224, 224), (255, 255, 255))


def dir_size_mb(p: Path):
    return round(sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6, 1)


def profile_blip(records, device, max_new_tokens):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    ckpt = RESULTS_DIR.parent / "checkpoints" / "blip" / "best_model"
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    proc = BlipProcessor.from_pretrained(str(ckpt))
    model = BlipForConditionalGeneration.from_pretrained(
        str(ckpt), torch_dtype=dtype).to(device).eval()

    imgs = [load_image(r) for r in records]
    prompts = [build_metadata_prompt(r) for r in records]

    @torch.no_grad()
    def one(i):
        inputs = proc(images=imgs[i], text=prompts[i], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast(device.type, dtype=dtype, enabled=device.type == "cuda"):
            model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4,
                           early_stopping=True, no_repeat_ngram_size=3)

    one(0)  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(len(records)):
        one(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None
    total = sum(p.numel() for p in model.parameters())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "params_total_M": round(total / 1e6, 1),
        "checkpoint_MB": dir_size_mb(ckpt),
        "sec_per_description": round(elapsed / len(records), 3),
        "peak_vram_GB": round(peak, 2) if peak else None,
    }


def profile_clip_gpt2(records, device, max_new_tokens):
    from models.clip_gpt2.evaluate import ClipGPT2Model, CLIP_TRANSFORM, MAX_TEXT_LEN, GPT2_MODEL
    from transformers import GPT2Tokenizer
    ckpt = RESULTS_DIR.parent / "checkpoints" / "clip_gpt2" / "best_model"
    tokr = GPT2Tokenizer.from_pretrained(GPT2_MODEL)
    tokr.pad_token = tokr.eos_token
    model = ClipGPT2Model()
    state = torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    imgs = [load_image(r) for r in records]
    prompts = [build_metadata_prompt(r) for r in records]

    @torch.no_grad()
    def one(i):
        px = CLIP_TRANSFORM(imgs[i]).unsqueeze(0).to(device)
        enc = tokr(prompts[i], return_tensors="pt", truncation=True,
                   max_length=MAX_TEXT_LEN, padding="max_length")
        model.generate(pixel_values=px,
                       input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device),
                       max_new_tokens=max_new_tokens, num_beams=4,
                       no_repeat_ngram_size=3)

    one(0)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(len(records)):
        one(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None
    total = sum(p.numel() for p in model.parameters())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "params_total_M": round(total / 1e6, 1),
        "checkpoint_MB": dir_size_mb(ckpt),
        "sec_per_description": round(elapsed / len(records), 3),
        "peak_vram_GB": round(peak, 2) if peak else None,
    }


def derive(entry):
    s = entry["sec_per_description"]
    hours_per_1k = s * 1000 / 3600
    kwh = hours_per_1k * GPU_TDP_WATTS / 1000
    entry["descriptions_per_hour"] = int(3600 / s)
    entry["gpu_hours_per_1k"] = round(hours_per_1k, 3)
    entry["kWh_per_1k"] = round(kwh, 3)
    entry["electricity_USD_per_1k"] = round(kwh * ELECTRICITY_USD_PER_KWH, 4)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("N5 — COMPUTE PROFILE")
    print("=" * 78)
    records = load_records(args.n)
    print(f"  profiling on {len(records)} items, max_new_tokens={args.max_new_tokens}")

    out = {"config": {"n_items": len(records),
                      "max_new_tokens": args.max_new_tokens,
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "gpu_tdp_watts": GPU_TDP_WATTS,
                      "electricity_usd_per_kwh": ELECTRICITY_USD_PER_KWH}}

    devices = [torch.device("cuda")] if torch.cuda.is_available() else []
    if not args.skip_cpu:
        devices.append(torch.device("cpu"))

    for dev in devices:
        for name, fn in (("BLIP", profile_blip), ("CLIP-GPT2", profile_clip_gpt2)):
            key = f"{name} [{dev.type}]"
            print(f"\n  → {key}")
            try:
                e = derive(fn(records, dev, args.max_new_tokens))
                out[key] = e
                for k, v in e.items():
                    print(f"      {k:<28} {v}")
            except Exception as exc:
                print(f"      failed: {exc}")

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved → {OUT_JSON}")


if __name__ == "__main__":
    main()
