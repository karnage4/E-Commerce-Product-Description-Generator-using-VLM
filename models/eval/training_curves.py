"""
Reconstruct and plot training curves from saved epoch checkpoints.

For each epoch_N/ checkpoint, runs a forward pass on the val split to get val
loss, building a real curve from what was actually trained.  If
training_history_{model}.json already exists (written by train_colab.py) it
includes BOTH train and val loss; otherwise only val loss is recovered here.

Usage:
    python -m models.eval.training_curves              # both models
    python -m models.eval.training_curves --model blip
    python -m models.eval.training_curves --model clip_gpt2
    python -m models.eval.training_curves --plot-only  # skip eval, use cached JSON
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.shared.config import RESULTS_DIR

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
BLIP_CKPT_DIR = _PROJECT_ROOT / "models" / "checkpoints" / "blip"
CLIP_CKPT_DIR = _PROJECT_ROOT / "models" / "checkpoints" / "clip_gpt2"


# ── Val-loss evaluators ───────────────────────────────────────────────────────

def _eval_blip_val_loss(checkpoint_dir: Path) -> float:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from models.blip.train_colab import DarazBlipDataset, BATCH_SIZE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    processor = BlipProcessor.from_pretrained(str(checkpoint_dir))
    model     = BlipForConditionalGeneration.from_pretrained(
        str(checkpoint_dir), torch_dtype=dtype
    ).to(device).eval()

    val_ds     = DarazBlipDataset("val", processor)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=(device.type == "cuda"))

    total, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"  {checkpoint_dir.name}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
            total += out.loss.item()
            n += 1
            del batch, out
    return total / n if n else float("inf")


def _eval_clip_val_loss(checkpoint_dir: Path) -> float:
    from transformers import GPT2Tokenizer
    from models.clip_gpt2.train_colab import DarazCLIPGPT2Dataset, ClipGPT2Model, BATCH_SIZE

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_dir))
    tokenizer.pad_token = tokenizer.eos_token

    model = ClipGPT2Model().to(device).eval()
    state = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state)

    val_ds     = DarazCLIPGPT2Dataset("val", tokenizer)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=(device.type == "cuda"))

    total, n = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"  {checkpoint_dir.name}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = model(**batch)
            total += loss.item()
            n += 1
            del batch, loss
    return total / n if n else float("inf")


# ── Curve builder ─────────────────────────────────────────────────────────────

def build_curve(model_key: str) -> list[dict]:
    """Evaluate val loss for every epoch_N checkpoint, return history list."""
    ckpt_dir = BLIP_CKPT_DIR if model_key == "blip" else CLIP_CKPT_DIR
    eval_fn  = _eval_blip_val_loss if model_key == "blip" else _eval_clip_val_loss

    history = []
    epoch   = 1
    while (ep_dir := ckpt_dir / f"epoch_{epoch}").exists():
        print(f"\n  [{model_key.upper()}] Epoch {epoch} — evaluating val loss...")
        val_loss = eval_fn(ep_dir)
        history.append({"epoch": epoch, "val_loss": round(val_loss, 6)})
        print(f"    val_loss = {val_loss:.4f}")
        epoch += 1

    if not history:
        print(f"  [!] No epoch checkpoints found in {ckpt_dir}")
    return history


# ── Plotter ───────────────────────────────────────────────────────────────────

DARK_BG   = "#0d0d1a"
PANEL_BG  = "#1a1a2e"
TEXT      = "#eee"
SPINE     = "#555"
COLORS    = {"blip": "#00d4ff", "clip_gpt2": "#ffd700"}
TRAIN_CLR = {"blip": "#4fc3f7", "clip_gpt2": "#ffe082"}


def plot_curves(histories: dict[str, list[dict]], output_path: Path) -> None:
    n_models = len(histories)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 4.5), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    for ax, (model_key, history) in zip(axes[0], histories.items()):
        epochs   = [h["epoch"] for h in history]
        val_loss = [h["val_loss"] for h in history]
        best_ep  = epochs[val_loss.index(min(val_loss))]

        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)

        # val loss
        ax.plot(epochs, val_loss, "o-",
                color=COLORS[model_key], linewidth=2, markersize=6,
                label="val loss")

        # train loss (only if saved by train_colab.py)
        if all("train_loss" in h for h in history):
            train_loss = [h["train_loss"] for h in history]
            ax.plot(epochs, train_loss, "s--",
                    color=TRAIN_CLR[model_key], linewidth=1.5, markersize=5,
                    alpha=0.75, label="train loss")

        ax.axvline(best_ep, color="#ff7043", linestyle=":", linewidth=1.5,
                   label=f"best epoch {best_ep}")

        ax.set_title(model_key.replace("_", "-").upper(), color=TEXT, fontsize=13, pad=10)
        ax.set_xlabel("Epoch", color="#aaa", fontsize=11)
        ax.set_ylabel("Loss",  color="#aaa", fontsize=11)
        ax.tick_params(colors="#aaa", labelsize=9)
        ax.set_xticks(epochs)
        ax.legend(facecolor="#0f3460", labelcolor=TEXT, fontsize=9, framealpha=0.9)

    fig.suptitle("Training / Validation Loss Curves", color=TEXT, fontsize=14, y=1.03)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Plot saved → {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["blip", "clip_gpt2", "all"], default="all")
    ap.add_argument("--plot-only", action="store_true",
                    help="Skip forward-pass eval; load existing training_history_*.json")
    args = ap.parse_args()

    model_keys = ["blip", "clip_gpt2"] if args.model == "all" else [args.model]
    histories: dict[str, list[dict]] = {}

    for model_key in model_keys:
        hist_file = RESULTS_DIR / f"training_history_{model_key}.json"

        if args.plot_only:
            if not hist_file.exists():
                print(f"  [!] {hist_file} not found — run without --plot-only first")
                continue
            history = json.loads(hist_file.read_text(encoding="utf-8"))
            print(f"  [{model_key.upper()}] Loaded {len(history)} epochs from {hist_file.name}")
        else:
            history = build_curve(model_key)
            if not history:
                continue
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            hist_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(f"  Saved → {hist_file}")

        histories[model_key] = history

    if not histories:
        print("  Nothing to plot.")
        return

    plot_curves(histories, RESULTS_DIR / "training_curves.png")
    print("  Done.")


if __name__ == "__main__":
    main()
