"""
BLIP Local Inference — GPU-accelerated when available.

After fine-tuning:
  1. Place the best_model/ checkpoint at models/checkpoints/blip/best_model/
  2. Run: python -m models.blip.evaluate

Generates descriptions for the test split, computes BLEU/ROUGE-L/METEOR/CIDEr,
saves results to models/results/blip_results.jsonl.

GPU (RTX 5060 Ti): ~0.2s/image.  CPU fallback: ~5–15s/image.
"""

import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT,
    RESULTS_DIR, build_metadata_prompt
)
from models.shared.metrics import compute_all_metrics, print_metrics_table, save_metrics

# ── Checkpoint location (checks both possible paths) ─────────────────────────
# The guide said: models/checkpoints/blip/best_model/
# But if you extracted directly into models/blip/, it's here:
_HERE = Path(__file__).resolve().parent          # models/blip/
_ROOT = _HERE.parent                             # models/

_CANDIDATE_PATHS = [
    _HERE / "blip_best_model",                  # models/blip/blip_best_model/
    _ROOT / "checkpoints" / "blip" / "best_model",  # models/checkpoints/blip/best_model/
    _HERE / "best_model",                       # models/blip/best_model/
]
CHECKPOINT_DIR = next((p for p in _CANDIDATE_PATHS if p.exists()), _CANDIDATE_PATHS[0])
RESULTS_FILE   = RESULTS_DIR / "blip_results.jsonl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32


def load_model(checkpoint_path: Path):
    print(f"\n  Loading BLIP from {checkpoint_path}...")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"\n[!] Checkpoint not found at {checkpoint_path}\n"
            "    Fine-tune first and download the best_model/ folder."
        )
    processor = BlipProcessor.from_pretrained(str(checkpoint_path))
    model     = BlipForConditionalGeneration.from_pretrained(
        str(checkpoint_path), torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    print(f"  Model loaded ({DEVICE}, {DTYPE})")
    return processor, model


def load_test_records(max_samples: int = 150) -> list[dict]:
    test_ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").strip().splitlines())
    records  = []
    with open(METADATA_FILE, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if rec.get("item_id") not in test_ids: continue
            if not rec.get("images") or not rec.get("description","").strip(): continue
            records.append(rec)
            if len(records) >= max_samples: break
    print(f"  Loaded {len(records)} test records")
    return records


def load_image(rec: dict) -> Image.Image:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try: return Image.open(p).convert("RGB")
            except: pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


@torch.no_grad()
def generate_description(
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    rec: dict,
    max_new_tokens: int = 200,
) -> str:
    image  = load_image(rec)
    prompt = build_metadata_prompt(rec)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def run_evaluation(max_samples: int = 150) -> None:
    processor, model = load_model(CHECKPOINT_DIR)
    records          = load_test_records(max_samples=max_samples)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-done for resumption
    done_ids: set[str]   = set()
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
                except: pass
        print(f"  Resuming — {len(done_ids)} already evaluated")

    t0 = time.time()
    with open(RESULTS_FILE, "a", encoding="utf-8") as out_f:
        for rec in tqdm(records, desc="BLIP inference [CPU]", unit="item"):
            if rec["item_id"] in done_ids:
                continue
            generated = generate_description(processor, model, rec)
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
    print(f"\n  Inference done in {elapsed/60:.1f} min ({elapsed/len(hypotheses):.1f}s/sample)")

    # Compute and display metrics
    metrics = compute_all_metrics(hypotheses, references)
    print_metrics_table({"BLIP Fine-tuned": metrics})
    save_metrics(metrics, str(RESULTS_DIR / "blip_metrics.json"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=150)
    parser.add_argument("--checkpoint",  type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()
    CHECKPOINT_DIR = Path(args.checkpoint)
    run_evaluation(max_samples=args.max_samples)
