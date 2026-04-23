"""
CLIP + GPT-2 model definition — runs on Google Colab (free T4 GPU).

Architecture:
  Product Image  →  CLIP Image Encoder  →  Image Embedding (512-dim)
                                                 ↓
                                  Linear Projection (learnable prefix)
                                                 ↓
  Product Metadata  →  Tokenize  →  Prompt Embedding
                                                 ↓
                            [Image Prefix] + [Prompt Tokens]
                                                 ↓
                                           GPT-2 LM head
                                                 ↓
                                      Generated Description

The CLIP image encoder is frozen for first 2 epochs, then unfrozen with
a 10× lower learning rate. GPT-2 and the projection layer are always trainable.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, GPT2LMHeadModel, GPT2Config


class ClipGPT2Model(nn.Module):
    """
    Vision-language model combining CLIP visual encoder with GPT-2 LM.

    Args:
        clip_model_name:  HuggingFace CLIP model ID
        gpt2_model_name:  HuggingFace GPT-2 model ID
        prefix_length:    Number of visual prefix tokens prepended to GPT-2
        freeze_clip:      Whether to freeze the CLIP encoder initially
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        gpt2_model_name: str = "gpt2",
        prefix_length:   int = 10,
        freeze_clip:    bool = True,
    ):
        super().__init__()
        self.prefix_length = prefix_length

        # ── CLIP Visual Encoder ───────────────────────────────────────────────
        self.clip = CLIPVisionModel.from_pretrained(clip_model_name)
        self.clip_embed_dim = self.clip.config.hidden_size   # 768 for ViT-B/32

        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # ── GPT-2 Language Model ──────────────────────────────────────────────
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.gpt2_embed_dim = self.gpt2.config.n_embd        # 768 for gpt2-base

        # ── Visual Prefix Projection ──────────────────────────────────────────
        # Projects CLIP CLS token → prefix_length GPT-2 embedding vectors
        self.visual_projection = nn.Sequential(
            nn.Linear(self.clip_embed_dim, self.gpt2_embed_dim * prefix_length),
            nn.Tanh(),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def get_visual_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images → visual prefix embeddings.

        Args:
            pixel_values: (B, 3, 224, 224)

        Returns:
            (B, prefix_length, gpt2_embed_dim)
        """
        clip_out     = self.clip(pixel_values=pixel_values)
        cls_embedding = clip_out.pooler_output                      # (B, 768)
        prefix_flat   = self.visual_projection(cls_embedding)       # (B, P*768)
        prefix         = prefix_flat.view(-1, self.prefix_length, self.gpt2_embed_dim)
        return prefix

    def forward(
        self,
        pixel_values:   torch.Tensor,   # (B, 3, 224, 224)
        input_ids:      torch.Tensor,   # (B, T_text)  — tokenized metadata prompt
        attention_mask: torch.Tensor,   # (B, T_text)
        labels:         torch.Tensor,   # (B, T_label) — tokenized description
    ) -> torch.Tensor:
        """
        Forward pass for training. Returns scalar cross-entropy loss.

        The full sequence fed to GPT-2 is:
          [visual_prefix (P tokens)] + [metadata_prompt (T_text tokens)] + [description (T_label tokens)]

        Labels are shifted internally by GPT-2's causal LM head.
        """
        B = pixel_values.size(0)
        device = pixel_values.device

        # ── Visual prefix ─────────────────────────────────────────────────────
        visual_prefix = self.get_visual_prefix(pixel_values)  # (B, P, 768)

        # ── Text embeddings ───────────────────────────────────────────────────
        gpt2_embeds = self.gpt2.transformer.wte   # word token embedding layer
        prompt_embeds = gpt2_embeds(input_ids)    # (B, T_text, 768)
        label_embeds  = gpt2_embeds(labels.clamp(min=0))  # (B, T_label, 768)

        # ── Concatenate: [prefix | prompt | label] ───────────────────────────
        combined_embeds = torch.cat([visual_prefix, prompt_embeds, label_embeds], dim=1)

        # ── Build label mask ─────────────────────────────────────────────────
        # Only compute loss on description tokens, not on prefix or prompt
        P       = self.prefix_length
        T_text  = input_ids.size(1)
        T_label = labels.size(1)

        # Build labels for the full sequence: -100 for prefix+prompt, real ids for description
        prefix_ignore = torch.full((B, P + T_text), -100, dtype=torch.long, device=device)
        full_labels   = torch.cat([prefix_ignore, labels], dim=1)  # (B, P+T_text+T_label)

        # Build attention mask for the full sequence
        prefix_mask  = torch.ones(B, P, device=device)
        label_mask   = (labels != -100).long()
        full_attn    = torch.cat([prefix_mask, attention_mask, label_mask], dim=1)

        outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            attention_mask=full_attn,
            labels=full_labels,
        )
        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        pixel_values:      torch.Tensor,
        input_ids:         torch.Tensor,
        attention_mask:    torch.Tensor,
        max_new_tokens:    int = 150,
        num_beams:         int = 4,
        no_repeat_ngram_size: int = 3,
    ) -> torch.Tensor:
        """Generate token IDs for a batch at inference time."""
        visual_prefix = self.get_visual_prefix(pixel_values)
        gpt2_embeds   = self.gpt2.transformer.wte
        prompt_embeds = gpt2_embeds(input_ids)

        combined = torch.cat([visual_prefix, prompt_embeds], dim=1)
        prefix_mask = torch.ones(input_ids.size(0), self.prefix_length, device=visual_prefix.device)
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

    # ──────────────────────────────────────────────────────────────────────────

    def unfreeze_clip(self, lr_multiplier: float = 0.1):
        """
        Unfreeze the CLIP encoder for joint fine-tuning.
        Call this after initial epochs to allow visual encoder adaptation.
        """
        for p in self.clip.parameters():
            p.requires_grad = True
        print(f"  CLIP encoder unfrozen (use lr × {lr_multiplier} for its params)")

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":      total,
            "trainable":  trainable,
            "frozen":     total - trainable,
            "pct_trainable": f"{100 * trainable / total:.2f}%",
        }
