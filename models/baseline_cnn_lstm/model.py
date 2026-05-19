"""
Show-and-Tell baseline: frozen MobileNetV2 image encoder + LSTM decoder
that consumes [BOS] + metadata_tokens + [SEP] + description_tokens.

The image feature is projected and used as the initial hidden state of the
LSTM, so visual context is available at every decoding step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ── Image encoder (frozen) ────────────────────────────────────────────────────

class ImageFeatureExtractor(nn.Module):
    """MobileNetV2 backbone -> 1280-d global-pooled feature."""

    FEATURE_DIM = 1280

    def __init__(self):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        net = mobilenet_v2(weights=weights)
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.preprocess = weights.transforms()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) — already preprocessed
        feat = self.features(x)
        feat = self.pool(feat).flatten(1)   # (B, 1280)
        return feat


# ── Caption decoder ───────────────────────────────────────────────────────────

class ShowAndTellModel(nn.Module):
    """
    Image-conditioned LSTM language model.

    The image feature is projected to `hidden_dim` and used as the initial
    hidden state (h_0). The input sequence is the embedded tokens of
    [BOS] + metadata + [SEP] + description.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        feature_dim: int = 1280,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.img_proj_h = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.img_proj_c = nn.Linear(feature_dim, hidden_dim * num_layers)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def _init_state(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = feat.size(0)
        h0 = self.img_proj_h(feat).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        c0 = self.img_proj_c(feat).view(b, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()
        return torch.tanh(h0), torch.tanh(c0)

    def forward(self, feat: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """
        feat:   (B, feature_dim)
        tokens: (B, T)  — input token ids (teacher-forcing)
        returns logits: (B, T, V)
        """
        h0, c0 = self._init_state(feat)
        emb = self.embed(tokens)                # (B, T, E)
        out, _ = self.lstm(emb, (h0, c0))       # (B, T, H)
        return self.out(self.dropout(out))

    @torch.no_grad()
    def generate(
        self,
        feat: torch.Tensor,
        prefix_ids: torch.Tensor,
        eos_id: int,
        pad_id: int,
        max_new_tokens: int = 80,
        banned_ids: list[int] | None = None,
        no_repeat_ngram_size: int = 3,
    ) -> list[list[int]]:
        """
        Greedy decode with optional n-gram repetition blocking.

        feat:        (B, feature_dim)
        prefix_ids:  (B, P) — already-padded prefix, e.g. [BOS, meta..., SEP]
        Returns a list of B token-id lists (excluding the prefix).
        """
        self.eval()
        device = feat.device
        b = feat.size(0)

        h, c = self._init_state(feat)
        emb = self.embed(prefix_ids)
        out, (h, c) = self.lstm(emb, (h, c))
        last_tok = prefix_ids[:, -1:]           # (B, 1)

        generated: list[list[int]] = [[] for _ in range(b)]
        done = torch.zeros(b, dtype=torch.bool, device=device)
        banned = list(banned_ids or [])
        n = no_repeat_ngram_size

        for _ in range(max_new_tokens):
            emb_step = self.embed(last_tok)
            out, (h, c) = self.lstm(emb_step, (h, c))
            logits = self.out(out[:, -1, :])    # (B, V)

            for b_id in banned:
                logits[:, b_id] = float("-inf")

            # n-gram repetition penalty: forbid completing any n-gram
            # whose (n-1)-prefix already appears earlier in generated[i].
            if n > 1:
                for i in range(b):
                    if done[i] or len(generated[i]) < n - 1:
                        continue
                    prefix = tuple(generated[i][-(n - 1):])
                    for j in range(len(generated[i]) - (n - 1)):
                        if tuple(generated[i][j : j + n - 1]) == prefix:
                            logits[i, generated[i][j + n - 1]] = float("-inf")

            next_tok = logits.argmax(dim=-1)    # (B,)

            for i in range(b):
                if done[i]:
                    continue
                tok = int(next_tok[i].item())
                if tok == eos_id:
                    done[i] = True
                    continue
                generated[i].append(tok)

            if bool(done.all().item()):
                break
            last_tok = next_tok.masked_fill(done, pad_id).unsqueeze(1)

        return generated
