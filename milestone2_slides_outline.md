# Milestone 2 — 5-Minute Video Presentation

**Vision-Language Models for E-Commerce Product Description Generation**
**Karam Hussain · Hassan Jabbar · Shaheer Shahid**

- Total runtime: **5:00**
- 18 slides total
- Speaker split: Karam (slides 1–7, 0:00 – 1:40) · Hassan (slides 8–13, 1:40 – 3:20) · Shaheer (slides 14–18, 3:20 – 5:00)

The LaTeX source for these slides lives at `milestone2_slides.tex`.

---

## 🎤 Karam — slides 1–6 (0:00 – 1:40)

### Slide 1 — Title

```
Vision-Language Models for E-Commerce
Product Description Generation
─────────────────────────────────────
Computer Vision · Milestone 2 · 2026
Karam Hussain · Hassan Jabbar · Shaheer Shahid
```

**Cue:** "Hi, we're Karam, Hassan, and Shaheer. Our project is automated product-description generation for Daraz.pk."

---

### Slide 2 — Problem & Dataset

**Inputs**
- Product image
- Structured metadata (brand, category, price, specs)

**Output** — fluent 3–5 sentence description

**DPD (Daraz Product Description) Dataset**
- 1,370 unique listings
- 5 categories (smartphones, tablets, consumer-electronics, home-appliances, womens-fashion)
- 80 / 10 / 10 train / val / test splits
- Custom-scraped + cleaned + deduplicated

**Cue:** "Sellers add thousands of listings daily and writing each description by hand doesn't scale."

---

### Slide 3 — 6-Phase Data Pipeline

| 1. Scrape | 2. Clean | 3. Dedup | 4. Build | 5. Train | 6. Eval |
|---|---|---|---|---|---|
| Playwright | HTML unescape | Fuzzy title | 80/10/10 | Colab T4 | Local CPU |
| Slider CAPTCHA | Quality filters | + Perceptual | splits | 5 epochs | BLEU |
| AJAX intercept | (≥20 words, ≤5 emojis) | image hash (Hamming ≤8) | Images + metadata | batch 8, FP16 | ROUGE-L, METEOR |

- **Output:** 1,370 deduplicated records ready for training.
- Orchestrated by `run.py` — each phase resumable.

**Cue:** "Six phases coordinated by `run.py` — from raw scraping to a reproducible 80/10/10 split."

---

### Slide 4 — Models We Used — The Lineup

| Model | Family | Role in this project |
|---|---|---|
| Metadata-only | Heuristic | Sanity floor — concatenates metadata fields, no learning. |
| **CNN+LSTM** (Show-and-Tell, Vinyals 2015) | From-scratch generative | Generative *floor*. MobileNetV2 image features + LSTM decoder, no pretrained language model. |
| **BLIP** (Salesforce, ICML 2022) | Pretrained VLM | End-to-end transformer (ViT-B/16 + cross-attention decoder), fine-tuned on our data. |
| **CLIP + GPT-2** (OpenAI, 2021) | Hybrid VLM | Frozen CLIP vision encoder fused with a GPT-2 language decoder via a learned 10-token visual prefix. |
| Two-Stage Pipeline | Refinement on top | VLM produces image-grounded draft → Gemma-4-31B judge → GPT-2 refiner. |

*Karam* explains the baseline. *Hassan* explains BLIP. *Shaheer* explains CLIP+GPT-2 and the two-stage pipeline.

**Cue:** "Before we dive in — here's the model lineup. One generative baseline from scratch, two pretrained vision-language models we fine-tuned, and a two-stage refinement on top."

---

### Slide 5 — Single source of truth: `build_metadata_prompt`

```python
# models/shared/config.py
def build_metadata_prompt(record: dict) -> str:
    parts = []
    if record.get("item_name"):   parts.append(f"Product: {record['item_name']}")
    if record.get("brand"):       parts.append(f"Brand: {record['brand']}")
    if record.get("category"):    parts.append(f"Category: {record['category']}")
    if record.get("subcategory"): parts.append(f"Subcategory: {record['subcategory']}")
    if record.get("price_pkr"):   parts.append(f"Price: PKR {record['price_pkr']:.0f}")
    if record.get("rating"):      parts.append(f"Rating: {record['rating']:.1f}/5")
    return ". ".join(parts)
```

**Cue:** "Every model in our project reads metadata through this one function — that's how the input format stays consistent across BLIP, CLIP+GPT-2, and our baseline."

---

### Slide 6 — Show-and-Tell — Our Generative Baseline

```
                ┌──────────────────────────────┐
                │ Frozen MobileNetV2 (ImageNet)│
   Image ──→    │ global avg pool → 1280-d     │
                └──────────────┬───────────────┘
                               │  Linear × 2 + tanh
                               ▼
                       (h₀, c₀) ← initial state
                               │
                ┌──────────────┴───────────────┐
                │  LSTM  (256 embed / 512 hid) │
                └──────────────┬───────────────┘
                               ▼
       <BOS> meta tokens <SEP> description ... <EOS>
       ←── loss masked ──→ ←─── loss only here ───→
```

- 7.8 M trainable params — runs locally on CPU
- Trained from scratch (no pretrained language model) for 20 epochs
- Pure measurement of what an image feature alone can teach an LSTM

**Cue:** "This baseline isolates exactly how much pretrained vision-language knowledge contributes to performance."

---

### Slide 7 — Show-and-Tell — Core Code

```python
# models/baseline_cnn_lstm/model.py
class ShowAndTellModel(nn.Module):
    def __init__(self, vocab_size, pad_id,
                 feature_dim=1280, embed_dim=256, hidden_dim=512):
        ...
        self.img_proj_h = nn.Linear(feature_dim, hidden_dim)   # → h₀
        self.img_proj_c = nn.Linear(feature_dim, hidden_dim)   # → c₀
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out   = nn.Linear(hidden_dim, vocab_size)

    def forward(self, feat, tokens):
        h0 = torch.tanh(self.img_proj_h(feat)).unsqueeze(0)   # vision → state
        c0 = torch.tanh(self.img_proj_c(feat)).unsqueeze(0)
        emb = self.embed(tokens)
        out, _ = self.lstm(emb, (h0, c0))                     # condition every step
        return self.out(out)
```

**Cue:** "Two linear layers convert the image feature into the LSTM's initial state. Hassan will explain how we trained this and how BLIP fixes its obvious weakness."

---

## 🎤 Hassan — slides 8–13 (1:40 – 3:20)

### Slide 8 — Loss Masking + `<unk>` Skip — The Training Trick

```python
# models/baseline_cnn_lstm/data_utils.py
def encode_sample(record, vocab, ...):
    meta_ids = vocab.encode(tokenize(build_metadata_prompt(record)))
    desc_ids = vocab.encode(tokenize(record["description"]))

    seq        = [vocab.bos_id] + meta_ids + [vocab.sep_id] + desc_ids + [vocab.eos_id]
    input_ids  = seq[:-1]
    target_ids = [-100] * len(input_ids)         # mask the prefix entirely

    sep_pos = 1 + len(meta_ids)
    for i in range(sep_pos, len(target_ids)):
        tok = seq[i + 1]
        # Don't train the model to emit <unk> -- that's giving up.
        target_ids[i] = -100 if tok == vocab.unk_id else tok
```

**Cue:** "Two label masks: minus-100 on metadata so the model isn't graded on copying it, and minus-100 on `<unk>` so it never learns 'unknown' as a safe answer."

---

### Slide 9 — Show-and-Tell — Results + Failure Modes

**Training:** 20 epochs CPU — val loss 6.68 → 4.86 (ppl 797 → 129)
**Test (n=135):** BLEU-1 = 4.65 · ROUGE-L = 7.97 · METEOR = 5.22

**Sample output** (every fashion item, fan, battery, and most phones produce this near-verbatim):

```
"brand : pre - wrap ; } • perfect for women and comfortable .
 • perfect and comfortable for women . • ideal for women
 • new original packaging warranty is a ..."
```

**Three failure modes:**
1. Mode-collapse to a generic "perfect for women" template
2. Fake spec-sheet template for phones / tablets (hallucinated GHz, RAM)
3. HTML tags leaked from raw seller copy (`pre-wrap;}`)

**Conclusion:** no real image grounding → justifies VLM pretraining.

**Cue:** "The image isn't really conditioning the output. That's our floor."

---

### Slide 10 — BLIP — Improved Model #1

**Context:** BLIP (*Bootstrapping Language–Image Pre-training*, Salesforce, ICML 2022) is a unified vision–language transformer pretrained on 129 M image–text pairs. We fine-tune the captioning checkpoint `Salesforce/blip-image-captioning-base` on our 1,100-sample Daraz training set.

```
  ┌─────────────────┐    ┌──────────────────────────┐    ┌──────────────────────┐
  │ ViT-B/16 vision │ ─→ │ Cross-attention decoder  │ ─→ │ Generated description │
  │   encoder       │    │ (BLIP MED, 12 layers)    │    │                       │
  └─────────────────┘    └──────────────────────────┘    └──────────────────────┘
```

**Salesforce/blip-image-captioning-base** — 224 M params, fine-tuned end-to-end.

**Two important code-level fixes:**
- **Loss masking** — only description tokens contribute to loss
- **Image augmentation** — flip + colour jitter + rotation, train split only

**Cue:** "BLIP is one of the two models that breaks through our floor."

---

### Slide 11 — BLIP — Loss-Masking Code

```python
# models/blip/train_colab.py
combined_text = f"{metadata}. {description}"

encoding = self.processor(images=image, text=combined_text,
                          padding="max_length", truncation=True,
                          max_length=MAX_SEQ_LENGTH, return_tensors="pt")

input_ids = encoding["input_ids"].squeeze(0)

# Find where metadata ends so we mask the prefix
prefix_ids = self.processor.tokenizer(f"{metadata}. ", add_special_tokens=False)["input_ids"]
prefix_len = min(1 + len(prefix_ids), MAX_SEQ_LENGTH)    # +1 for [CLS]

labels = input_ids.clone()
labels[:prefix_len]                                     = -100   # mask metadata
labels[labels == self.processor.tokenizer.pad_token_id] = -100   # mask padding
```

**Cue:** "Before this fix BLIP was learning to echo metadata. Setting those labels to minus-100 forces it to grade itself only on the description."

---

### Slide 12 — BLIP — Image Augmentation (Train Split Only)

```python
# models/blip/train_colab.py
TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.RandomRotation(degrees=10),
])

# Applied at the PIL level -- BEFORE the BlipProcessor, so the model still
# sees the processor's own resize + normalize on top.
if split == "train":
    image = TRAIN_AUGMENT(image)
```

- **Flip** — captures left-vs-right product photos
- **Jitter** — simulates seller-lighting variation
- **Rotation** — handles tilted product placements

**Cue:** "Applied only on the training split — never on val/test — so our metric estimates stay unbiased."

---

### Slide 13 — From BLIP to CLIP+GPT-2

| **BLIP** | **CLIP + GPT-2 (hybrid)** |
|---|---|
| Single end-to-end VLM | Frozen CLIP vision encoder |
| Cross-attention decoder | GPT-2 language decoder |
| Strong **recall** (ROUGE-L 12.70) | Strong **precision** (BLEU-1 12.09) |

Both built from `models/shared/dataset.py` + `build_metadata_prompt` so the input pipeline is identical.

**Cue:** "That's our first improved model. Shaheer will walk through the second, the CLIP+GPT-2 hybrid, and the two-stage pipeline."

---

## 🎤 Shaheer — slides 14–18 (3:20 – 5:00)

### Slide 14 — CLIP + GPT-2 — Architecture

**Context:** CLIP (*Contrastive Language–Image Pre-training*, OpenAI 2021) is a 400 M-pair contrastively-trained image encoder; GPT-2 (OpenAI 2019) is a 117 M-parameter autoregressive language model. We fuse them with a learned visual-prefix projection — a parameter-efficient way to condition a frozen language model on an image.

```
         ┌──────────────────┐
 Image ─→│ CLIP ViT-B/32     │── CLS (768-d) ──┐
         │ (frozen → ep 3)   │                  │
         └──────────────────┘                  ▼
                                  ┌────────────────────────────┐
                                  │ Linear(768 → 10 × 768)     │
                                  │ + Tanh   (visual prefix)   │
                                  └─────────────┬──────────────┘
                                                ▼
                          [10 visual tokens] + [metadata tokens]
                                                │
                                                ▼
                                       GPT-2 (117 M, fully trainable)
                                                │
                                                ▼
                                       Generated description
```

**Cue:** "A frozen CLIP encoder produces one CLS embedding — we project that into ten GPT-2-shaped tokens."

---

### Slide 15 — CLIP + GPT-2 — Visual Prefix Code

```python
# models/clip_gpt2/model.py
class ClipGPT2Model(nn.Module):
    def __init__(self, ..., prefix_length=10):
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.visual_projection = nn.Sequential(
            nn.Linear(self.clip_embed_dim, self.gpt2_embed_dim * prefix_length),
            nn.Tanh(),
        )

    def get_visual_prefix(self, pixel_values):
        cls_embedding = self.clip(pixel_values=pixel_values).pooler_output
        prefix_flat   = self.visual_projection(cls_embedding)
        return prefix_flat.view(-1, self.prefix_length, self.gpt2_embed_dim)

    def forward(self, pixel_values, input_ids, attention_mask, labels):
        visual_prefix = self.get_visual_prefix(pixel_values)
        prompt_embeds = self.gpt2.transformer.wte(input_ids)
        combined      = torch.cat([visual_prefix, prompt_embeds, ...], dim=1)
        # ... GPT-2 forward with masked labels
```

**Cue:** "One linear layer maps the CLIP CLS embedding into ten visual tokens that we prepend to the prompt — the language model sees them as if they were words."

---

### Slide 16 — Two-Stage Pipeline

```
                            STAGE 1                              STAGE 2
       ┌─────────────────────────────────┐         ┌────────────────────────┐
       │  VLM sees IMAGE + category only │   ───→  │ Fine-tuned GPT-2 reads │
       │  → visually-grounded draft       │         │ draft + FULL metadata │ → final
       └─────────────────────────────────┘         └────────────────────────┘
                       │
                       ▼
            ┌────────────────────────────┐
            │ Judge: Gemma-4-31B reads   │   scores: visual_grounding /
            │ draft + image              │           fluency / relevance
            └────────────────────────────┘
```

- Per item we save: Stage 1 text, judge scores, Stage 2 text
- Output: `models/results/two_stage_results.jsonl`

**Cue:** "Stripping metadata in Stage 1 forces the VLM to actually use the image. The judge gives a quality signal that doesn't depend on n-gram overlap."

---

### Slide 17 — Stage 1 Prompt Builder + Judge Call

```python
# models/shared/config.py
def build_stage1_prompt(record):
    """Minimal prompt -- forces the model to ground description in pixels."""
    parts = []
    if record.get("category"):    parts.append(f"Category: {record['category']}")
    if record.get("subcategory"): parts.append(f"Subcategory: {record['subcategory']}")
    return ". ".join(parts)
```

```python
# models/stage2/openrouter_refiner.py
def score(client, image_path, generated_text):
    """Gemma-4-31B judges Stage 1 output on three axes (1-5)."""
    return client.score_multimodal(
        image=image_path, text=generated_text,
        rubric=["visual_grounding", "fluency", "relevance"],
        model="google/gemma-4-31b-it",
    )
```

**Cue:** "The minimal Stage-1 prompt is what makes the two-stage pipeline genuinely image-grounded."

---

### Slide 18 — Final Results — Takeaway

| Model | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|---|---|---|---|---|---|
| Metadata-only heuristic | 1.04 | 0.30 | 13.71 | 8.80 | — |
| CNN+LSTM Show-and-Tell (scratch) | 4.65 | 0.34 | 7.97 | 5.22 | 0.46 |
| CLIP+GPT-2 (fine-tuned) | **12.09** | **0.92** | 10.28 | 10.53 | — |
| BLIP (fine-tuned) | 7.94 | 0.54 | **12.70** | **12.02** | — |
| Two-Stage: BLIP → GPT-2 refiner | 10.84 | 0.37 | 8.87 | 9.53 | **0.63** |
| Two-Stage: CLIP+GPT-2 → GPT-2 | 8.57 | 0.20 | 7.15 | 6.65 | 0.19 |

*"—" = CIDEr not computed (single-stage VLMs evaluated before `pycocoevalcap` was available).*

**Key takeaway:** Both pretrained VLMs beat the from-scratch CNN+LSTM on every single metric. → Vision-language **pretraining** is the dominant lever, not architecture. CIDEr ranks Two-Stage BLIP highest (0.63), confirming that the Stage-2 refiner driven by an image-grounded draft produces the most reference-aligned output.

**Main limitation:** both VLMs overfit after epoch 1 (only 1,100 train samples).
**Next step:** scale dataset to 10K+ products.

**Cue:** "Thanks for watching."

---

## Speaker workload — code coverage per member

| Speaker | Code / module owned |
|---|---|
| **Karam** | `scraper/`, `pipeline/cleaner.py`, `dedup/`, `models/shared/config.py:build_metadata_prompt`, `models/baseline_cnn_lstm/model.py` |
| **Hassan** | `models/baseline_cnn_lstm/data_utils.py:encode_sample`, `models/baseline_cnn_lstm/train.py` (results), `models/blip/train_colab.py` (loss masking + augmentation) |
| **Shaheer** | `models/clip_gpt2/model.py`, `models/two_stage_pipeline.py`, `models/shared/config.py:build_stage1_prompt`, `models/stage2/openrouter_refiner.py`, results table |

Each speaker owns one full model's architecture explanation plus one supporting piece of plumbing — workload is roughly equal.

## Practical recording tips

- **150 words/minute pace.** Each block above is trimmed to ~250 words ≈ 100 seconds. Add nothing; only cut.
- **Mono-space font** (JetBrains Mono / Fira Code) for every code block.
- **Bold the one line that matters** on each code slide so the audience eye lands there.
- **Real generated outputs** (slide 8) are the single most memorable moment — read it verbatim.
- **Rehearse the two handoffs**: "Hassan will walk through..." and "Shaheer will explain..." — these are the highest-risk transitions, practise them until they're automatic.

## Building the PDF

```
pdflatex milestone2_slides.tex
pdflatex milestone2_slides.tex      # second pass for TOC / cross-refs
```

The LaTeX uses only standard packages (beamer, listings, xcolor, tikz, booktabs) and works with `pdflatex`, `xelatex`, or `lualatex`.
