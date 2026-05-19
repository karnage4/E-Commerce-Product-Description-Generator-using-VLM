"""
Word-level vocabulary for the Show-and-Tell baseline.

Built once from training descriptions + metadata prompts, then saved
to disk and reused at eval time.
"""

import json
import re
from collections import Counter
from pathlib import Path


PAD, BOS, EOS, UNK, SEP = "<pad>", "<bos>", "<eos>", "<unk>", "<sep>"
SPECIALS = [PAD, BOS, EOS, UNK, SEP]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")


def tokenize(text: str) -> list[str]:
    """Lowercased word/punct tokeniser. Keeps numbers, splits punctuation."""
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(t, unk) for t in tokens]

    def decode(self, ids: list[int], strip_specials: bool = True) -> str:
        out = []
        for i in ids:
            w = self.itos[i] if 0 <= i < len(self.itos) else UNK
            if strip_specials and w in (PAD, BOS, EOS, SEP):
                if w == EOS:
                    break
                continue
            out.append(w)
        return " ".join(out)

    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]
    @property
    def sep_id(self) -> int: return self.stoi[SEP]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    @classmethod
    def build(cls, corpus: list[str], min_freq: int = 2, max_size: int | None = None) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in corpus:
            counter.update(tokenize(text))

        words = [w for w, c in counter.most_common() if c >= min_freq]
        if max_size is not None:
            words = words[: max(0, max_size - len(SPECIALS))]
        itos = list(SPECIALS) + [w for w in words if w not in SPECIALS]
        return cls(itos)
