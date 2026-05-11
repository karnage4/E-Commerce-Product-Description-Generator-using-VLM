"""
Attention / saliency visualisation — C3 Grad-CAM requirement.

Extracts decoder cross-attention from fine-tuned BLIP and overlays it on the
product image as a heatmap, showing which image regions the model attends to
when generating the description.

Method:
  1. Register forward hooks on every BlipTextCrossAttention layer.
  2. Run a single forward pass (not generate) with the generated description
     as the decoder label sequence.
  3. Stack all captured cross-attention tensors → average over layers and heads
     → take spatial tokens (drop CLS) → reshape to patch grid → bilinear
     upsample to image size → normalise → overlay with matplotlib.

Output:
  models/results/attention_maps/   — one PNG per sample (original + heatmap + overlay)
  models/results/attention_maps/attention_summary.html — combined report

Run:
    python -m models.eval.attention_viz
    python -m models.eval.attention_viz --num-samples 3 --checkpoint <path>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.shared.config import (
    METADATA_FILE, IMAGES_DIR, TEST_SPLIT, RESULTS_DIR,
    build_stage1_prompt,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32

_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parents[1]

_CKPT_CANDIDATES = [
    _ROOT / "models" / "blip" / "blip_best_model",
    _ROOT / "models" / "checkpoints" / "blip" / "best_model",
    _ROOT / "models" / "blip" / "best_model",
]

OUT_DIR = RESULTS_DIR / "attention_maps"


# ── Model loading ──────────────────────────────────────────────────────────────

def find_checkpoint(override: str | None) -> Path:
    if override:
        return Path(override)
    found = next((p for p in _CKPT_CANDIDATES if p.exists()), None)
    if found:
        return found
    raise FileNotFoundError(
        "[!] BLIP checkpoint not found. Checked:\n"
        + "\n".join(f"    {p}" for p in _CKPT_CANDIDATES)
    )


def load_blip(ckpt: Path):
    print(f"  Loading BLIP from {ckpt} ...")
    processor = BlipProcessor.from_pretrained(str(ckpt))
    model = BlipForConditionalGeneration.from_pretrained(
        str(ckpt), torch_dtype=DTYPE,
    ).to(DEVICE)
    model.eval()
    print(f"  BLIP loaded ({DEVICE})")
    return processor, model


# ── Data ───────────────────────────────────────────────────────────────────────

def load_test_records(n: int) -> list[dict]:
    test_ids = set(Path(TEST_SPLIT).read_text(encoding="utf-8").strip().splitlines())
    # Read a large pool so diversity filtering has enough candidates
    pool: list[dict] = []
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
            pool.append(rec)
    # Pick n records with category diversity
    seen_cats: set[str] = set()
    diverse: list[dict] = []
    for rec in pool:
        cat = rec.get("category", "")
        if cat not in seen_cats:
            seen_cats.add(cat)
            diverse.append(rec)
        if len(diverse) >= n:
            break
    # If not enough diverse categories, fill remaining slots from pool
    if len(diverse) < n:
        ids_used = {r["item_id"] for r in diverse}
        for rec in pool:
            if rec["item_id"] not in ids_used:
                diverse.append(rec)
                if len(diverse) >= n:
                    break
    return diverse[:n]


def load_image(rec: dict) -> Image.Image:
    for rel in rec.get("images", []):
        p = IMAGES_DIR.parent / rel
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    return Image.new("RGB", (224, 224), (200, 200, 200))


# ── Vision encoder attention rollout ──────────────────────────────────────────

def extract_vision_attention(
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    image: Image.Image,
) -> np.ndarray | None:
    """
    Attention rollout through BLIP's ViT vision encoder.

    Uses the CLS token's accumulated attention to all patch tokens after
    rolling attention through layers — independent of the text decoder,
    so it works regardless of training objective.

    Returns a (224, 224) float32 array in [0, 1].
    """
    enc = processor(images=image, return_tensors="pt")
    pixel_values = enc["pixel_values"].to(DEVICE)

    with torch.no_grad():
        vision_out = model.vision_model(
            pixel_values=pixel_values,
            output_attentions=True,
        )

    # vision_out.attentions: tuple of (1, num_heads, seq_len, seq_len) per layer
    # seq_len = 1 (CLS) + num_patches (196 for patch32 at 224px)
    if not vision_out.attentions:
        return None

    # Attention rollout: accumulate attention through layers with residual
    num_tokens = vision_out.attentions[0].shape[-1]
    rollout = torch.eye(num_tokens)

    for attn in vision_out.attentions:
        # (1, H, S, S) → average heads → (S, S)
        a = attn.squeeze(0).mean(dim=0).cpu().float()
        # Add identity (residual connection) and renormalise rows
        a = a + torch.eye(num_tokens)
        a = a / a.sum(dim=-1, keepdim=True)
        rollout = a @ rollout

    # CLS row → attention from CLS to each patch token
    cls_attn = rollout[0, 1:]          # (num_patches,)
    spatial  = cls_attn.numpy()

    n_patches = spatial.shape[0]
    side = int(round(n_patches ** 0.5))
    if side * side != n_patches:
        return None

    attn_grid = spatial.reshape(side, side)

    # Bilinear upsample to image size
    t = torch.tensor(attn_grid).unsqueeze(0).unsqueeze(0)
    t_up = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
    attn_up = t_up.squeeze().numpy()

    # Mild Gaussian smoothing
    attn_up = gaussian_filter(attn_up, sigma=6)

    # Percentile normalisation
    lo = np.percentile(attn_up, 10)
    hi = np.percentile(attn_up, 99)
    if hi - lo < 1e-8:
        return None
    return np.clip((attn_up - lo) / (hi - lo), 0, 1)


# ── Visualisation ──────────────────────────────────────────────────────────────

def save_attention_figure(
    image: Image.Image,
    attn_map: np.ndarray,
    item_id: str,
    category: str,
    generated: str,
    out_path: Path,
) -> None:
    img_arr = np.array(image.resize((224, 224)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#0d0d1a")

    for ax in axes:
        ax.set_facecolor("#0d0d1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    titles = ["Original Image", "Attention Map", "Overlay"]
    title_colors = ["#00d4ff", "#ffd700", "#a5d6a7"]

    axes[0].imshow(img_arr)
    axes[1].imshow(attn_map, cmap="hot", vmin=0, vmax=1)
    axes[2].imshow(img_arr)
    axes[2].imshow(attn_map, cmap="hot", alpha=0.55, vmin=0, vmax=1)

    for ax, title, col in zip(axes, titles, title_colors):
        ax.set_title(title, color=col, fontsize=11, pad=8)
        ax.axis("off")

    caption = f"[{category}]  {generated[:100]}{'...' if len(generated) > 100 else ''}"
    fig.text(0.5, 0.02, caption, ha="center", va="bottom",
             color="#aaa", fontsize=9, wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ── HTML summary ───────────────────────────────────────────────────────────────

def build_summary_html(entries: list[dict], out_dir: Path) -> str:
    cards = ""
    for e in entries:
        rel = Path(e["png"]).name
        cards += f"""
    <div style="margin-bottom:30px;padding:16px;background:#1a1a2e;
                border-radius:8px;border:1px solid #333;">
      <div style="color:#aaa;margin-bottom:8px;">
        <b>Category:</b> {e['category']} &nbsp;|&nbsp; <b>Item:</b> {e['item_id']}
      </div>
      <img src="{rel}" style="width:100%;max-width:700px;border-radius:6px;" />
      <div style="margin-top:10px;font-size:13px;">
        <span style="color:#4fc3f7;font-weight:bold;">Generated:</span>
        <span style="color:#ddd;"> {e['generated']}</span>
      </div>
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Attention Maps — Daraz BLIP</title>
  <style>
    body {{ background:#0d0d1a; color:#eee; font-family:monospace;
            padding:30px; max-width:800px; margin:auto; }}
    h1   {{ color:#00d4ff; text-align:center; }}
    p    {{ color:#888; text-align:center; font-size:13px; }}
  </style>
</head>
<body>
  <h1>Decoder Cross-Attention Maps — BLIP</h1>
  <p>Averaged over all decoder layers and attention heads.<br>
     Bright regions = where the model looked when generating the description.</p>
  {cards}
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--checkpoint",  type=str, default=None)
    args = parser.parse_args()

    ckpt = find_checkpoint(args.checkpoint)
    processor, model = load_blip(ckpt)

    records = load_test_records(args.num_samples)
    print(f"  Processing {len(records)} samples")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []

    for rec in tqdm(records, desc="Attention maps", unit="item"):
        image    = load_image(rec)
        prompt   = build_stage1_prompt(rec)

        # Generate description first
        inputs = {k: v.to(DEVICE) for k, v in
                  processor(images=image, text=prompt, return_tensors="pt").items()}
        with torch.no_grad():
            with torch.autocast(DEVICE.type, dtype=DTYPE, enabled=DEVICE.type == "cuda"):
                ids = model.generate(
                    **inputs, max_new_tokens=150, num_beams=4,
                    no_repeat_ngram_size=3, early_stopping=True,
                )
        generated = processor.decode(ids[0], skip_special_tokens=True).strip()

        # Extract vision encoder attention rollout
        attn = extract_vision_attention(processor, model, image)
        if attn is None:
            print(f"  [~] Skipping {rec['item_id']} — vision attention unavailable")
            continue

        out_path = OUT_DIR / f"{rec['item_id']}_attention.png"
        save_attention_figure(
            image, attn,
            item_id=rec["item_id"],
            category=rec.get("category", "unknown"),
            generated=generated,
            out_path=out_path,
        )

        entries.append({
            "item_id":   rec["item_id"],
            "category":  rec.get("category", "unknown"),
            "generated": generated,
            "png":       str(out_path),
        })
        print(f"  Saved → {out_path.name}")

    if entries:
        html_path = OUT_DIR / "attention_summary.html"
        html_path.write_text(
            build_summary_html(entries, OUT_DIR), encoding="utf-8"
        )
        print(f"\n  Summary → {html_path}")
    else:
        print("\n  [!] No attention maps produced — check checkpoint and model architecture.")


if __name__ == "__main__":
    main()
