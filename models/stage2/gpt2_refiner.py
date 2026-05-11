"""
Stage 2 refiner — fine-tuned GPT-2 (from CLIP-GPT2 checkpoint).

Extracts the GPT-2 language model head from the saved CLIP-GPT2 state dict
and runs text-only generation. Input is:

    "Visual: {stage1_description}. {full_metadata_prompt}"

The fine-tuned GPT-2 learned Daraz description style and will continue the
prompt into a product description.

Checkpoint expected at (first found wins):
    models/clip_gpt2/clip_gpt2_best_model/
    models/checkpoints/clip_gpt2/best_model/
    models/clip_gpt2/best_model/

Standalone test:
    python -m models.stage2.gpt2_refiner
"""

import sys
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.shared.config import build_metadata_prompt

MAX_NEW_TOKENS       = 150
NUM_BEAMS            = 4
NO_REPEAT_NGRAM_SIZE = 3
MAX_PROMPT_TOKENS    = 100   # truncate prompt so there's room for generation

_CANDIDATE_PATHS = [
    PROJECT_ROOT / "models" / "clip_gpt2" / "clip_gpt2_best_model",
    PROJECT_ROOT / "models" / "checkpoints" / "clip_gpt2" / "best_model",
    PROJECT_ROOT / "models" / "clip_gpt2" / "best_model",
]


def _find_checkpoint() -> Path:
    found = next((p for p in _CANDIDATE_PATHS if p.exists()), None)
    if found:
        return found
    raise FileNotFoundError(
        "[!] CLIP-GPT2 checkpoint not found. Checked:\n"
        + "\n".join(f"    {p}" for p in _CANDIDATE_PATHS)
        + "\n    Download best_model/ from Colab and place it at one of the paths above."
    )


def load_gpt2(checkpoint_path: Path | None = None):
    """
    Load only the GPT-2 weights from a CLIP-GPT2 checkpoint.

    The saved state dict has keys like:
        gpt2.transformer.wte.weight
        gpt2.lm_head.weight
        clip.*
        visual_projection.*

    We filter to gpt2.* keys, strip the prefix, and load into a plain
    GPT2LMHeadModel so there's no dependency on the CLIP encoder at Stage 2.
    """
    if checkpoint_path is None:
        checkpoint_path = _find_checkpoint()

    print(f"  Loading fine-tuned GPT-2 from {checkpoint_path} ...")

    weights_file = checkpoint_path / "model.pt"
    if not weights_file.exists():
        raise FileNotFoundError(f"[!] model.pt not found in {checkpoint_path}")

    full_state = torch.load(weights_file, map_location="cpu")
    gpt2_state = {
        k[len("gpt2."):]: v
        for k, v in full_state.items()
        if k.startswith("gpt2.")
    }

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.load_state_dict(gpt2_state)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(str(checkpoint_path))
    tokenizer.pad_token = tokenizer.eos_token

    print("  GPT-2 loaded (CPU mode)")
    return tokenizer, model


def _build_prompt(rec: dict, stage1_description: str) -> str:
    """
    Combine Stage 1 visual description + full metadata into a continuation prompt.
    Format mirrors training distribution: starts with product context, flows into description.
    """
    metadata = build_metadata_prompt(rec)
    return f"Visual: {stage1_description}. {metadata}"


@torch.no_grad()
def refine(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel,
           rec: dict, stage1_description: str) -> str:
    """
    Generate a Stage 2 description using the fine-tuned GPT-2.
    Returns the generated text (continuation after the prompt).
    """
    prompt = _build_prompt(rec, stage1_description)

    enc = tokenizer(
        prompt,
        max_length=MAX_PROMPT_TOKENS,
        truncation=True,
        return_tensors="pt",
    )
    prompt_len = enc["input_ids"].shape[1]

    output_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=MAX_NEW_TOKENS,
        num_beams=NUM_BEAMS,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode only the newly generated tokens (strip the prompt)
    new_ids = output_ids[0][prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy_rec = {
        "item_id":    "test-001",
        "item_name":  "Infinix Hot 40 Pro",
        "brand":      "Infinix",
        "category":   "smartphones",
        "subcategory": "android-phones",
        "attributes": {"RAM": "8GB", "Storage": "256GB", "Battery": "5000mAh"},
    }
    dummy_stage1 = (
        "A slim black smartphone with a glossy finish, triple camera module "
        "on the back, and a waterdrop notch display. The Infinix logo is "
        "visible on the rear panel."
    )

    tok, mdl = load_gpt2()
    result = refine(tok, mdl, dummy_rec, dummy_stage1)
    print("\nStage 2 (GPT-2) output:\n", result)
