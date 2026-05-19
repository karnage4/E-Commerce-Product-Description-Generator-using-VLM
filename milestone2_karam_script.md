# Karam — Speaking Script (Milestone 2 Video)

**Your portion:** slides 1–7 · **0:00 – 1:55** · ~285 words at 150 wpm
**Total team video budget:** 5:00 max (you ≈ 110s, Hassan ≈ 95s, Shaheer ≈ 95s)

> **This version is the reasoning-first script.** Every cue now explains **why** we made a choice and **what impact** it had, not just what the code does. The slide handles the *what*; you handle the *why*.

Stay around **285 words / 110 seconds**. Read aloud once with a stopwatch — under 100s means you can slow down on slide 4 (the model lineup); over 120s means you should drop one of the bracketed `[optional]` sentences below.

---

## Slide 1 — Title (≈ 0:00 – 0:08)

> "Hi, I'm **Karam** — joined by Hassan and Shaheer. Our project is **automated product-description generation for Daraz dot p-k**."

**Delivery:**
- Smile, eye contact, names slowly so the audience pairs them with faces.
- Pause one beat after "Daraz" before clicking through.

---

## Slide 2 — Problem & Dataset (≈ 0:08 – 0:24)

> "The problem: existing vision-language datasets are all Western — Amazon, ABO. **No Pakistani e-commerce dataset exists.** Our first contribution was building one — 1{,}370 listings across five categories, custom-scraped, cleaned, and deduplicated."

**Why this framing matters:**
- "No dataset exists" is the academic gap. State it cleanly, the audience needs to hear the motivation.
- Slight emphasis on **"no Pakistani e-commerce dataset exists"** — that's the contribution claim.
- Don't list the five category names — they're on the slide and reading them costs 4 seconds.

---

## Slide 3 — Pipeline (≈ 0:24 – 0:43)

> "Each pipeline phase solves a real data-quality problem. Daraz hides product data behind dynamic JavaScript, so we **needed Playwright with a slider-CAPTCHA solver** — nothing simpler would scrape this. Cleaning removed roughly 30% of listings as HTML spam. And the **two-pass deduplication** catches the same product sold by multiple sellers — without it, the model would just memorise duplicates and our test set would be contaminated."

**Why this is now reasoning-heavy:**
- Each phase is justified by the problem it solves, not by its name.
- The "contaminated test set" line is the impact statement — say it slightly slower so the audience absorbs *why* dedup matters for our reported metrics.
- [optional, cut if over time] *"Two passes specifically because fuzzy title match alone misses image-duplicates with different captions."*

---

## Slide 4 — Models We Used (≈ 0:43 – 1:04)

> "**Why three model families** — to **isolate where performance comes from**. The CNN-plus-LSTM is trained from scratch with no pretrained language model, so it measures what an image feature alone can teach. BLIP and CLIP-plus-GPT-2 add large-scale pretraining on top — the **gap between them and the baseline is precisely the value of pretraining**, which is what we wanted to quantify. The two-stage pipeline tests whether a refiner can fix metadata-echoing without retraining the VLM."

**This is your most important slide.** Speak it slowly.

**Why this matters as a speech:**
- The audience needs to understand the *scientific structure*. Three families = three different hypotheses being tested. Without this framing the rest of the video looks like four random models.
- When you say "Hassan" and "Shaheer", glance at them — that cues the audience for who's coming next.
- Hit the phrase **"the gap is precisely the value of pretraining"** — that's the central claim of the whole project.

---

## Slide 5 — `build_metadata_prompt` (≈ 1:04 – 1:18)

> "All our models read metadata through **this one function** — a deliberate reproducibility decision. If each model had its own prompt logic, any difference in metrics could come from the prompt rather than the model itself. **One shared function means the only variable left is the model.**"

**Why this is now reasoning-heavy:**
- The reasoning is: *experimental control*. That's the under-the-hood detail the audience would otherwise miss.
- Hit **"the only variable left is the model"** — that's the impact statement.
- Don't read the function body. The audience can see it.

---

## Slide 6 — Show-and-Tell architecture (≈ 1:18 – 1:39)

> "Three design choices, each deliberate. **Why MobileNetV2 frozen** — it's cheap enough to run on CPU and its ImageNet features transfer well to product photos, so we get good visual features almost for free. **Why initialise the LSTM state from the image** — that's the original Show-and-Tell trick from 2015, well-studied and simple to reason about. **Why train completely from scratch** — so any output comes purely from the image plus our 1{,}100 training samples, with no help from a pretrained language model."

**Why this is now reasoning-heavy:**
- Each of the three choices is paired with its motivation. "Three design choices, each deliberate" is the framing line — it tells the audience to expect three justifications in a row.
- Slow down on **"with no help from a pretrained language model"** — that's the controlled-comparison setup that justifies the entire baseline.
- [optional, cut if over time] *"It's cheap enough to run on CPU"* — drop "cheap enough to run on CPU" if you're over budget; the rest still works.

---

## Slide 7 — Show-and-Tell code + handoff (≈ 1:39 – 1:55)

> "**Two separate linear projections** — one for h-zero, one for c-zero — so the LSTM has independent paths to encode visual information. **Tanh** keeps both states in the LSTM's natural activation range, which stops the optimiser from blowing up early in training.
>
> **Hassan will explain how we trained this, and how BLIP improves on it.**"

**Why this is now reasoning-heavy:**
- The reasoning behind **two** linear layers (instead of one) is *capacity*: more parameters to encode richer image info.
- The reasoning behind **tanh** is *training stability* — without it, the un-bounded image features destabilise the LSTM gradients.
- The handoff line is the single most important sentence you say. Pause one full beat before it, slow down, look at Hassan if you're recording together. Rehearse this line three times before you record.

---

## Total word count: ~285 · target time: ≈ 110 s

| Slide | Topic | Key reasoning beat | Words | Target s |
|---|---|---|---:|---:|
| 1 | Title | (intro only) | 20 | 8 |
| 2 | Problem & Dataset | *No Pakistani e-commerce dataset exists* | 40 | 16 |
| 3 | Pipeline | *Without dedup the test set is contaminated* | 56 | 22 |
| 4 | **Models We Used** (NEW) | *The gap to the baseline is the value of pretraining* | 64 | 25 |
| 5 | `build_metadata_prompt` | *Only variable left is the model* | 38 | 15 |
| 6 | Show-and-Tell architecture | *No help from a pretrained language model* | 50 | 20 |
| 7 | Show-and-Tell code + handoff | *Two projections = more capacity; tanh = stability* | 35 | 14 |
| **Total** | | | **303** | **≈ 120** |

> **Reality check:** the deeper script is denser, so the natural reading rate slows. Expect ~120s on first run-through. Trim the bracketed `[optional]` sentences on slides 3 and 6 to land at 110s. If you're still over, the easiest sentence to drop is *"Cleaning removed roughly 30% of listings as HTML spam"* on slide 3 (saves ~5s).

---

## Three things to rehearse separately

1. **The opening 20 words (slide 1).** First impression. Practise until it sounds unrehearsed.
2. **The model-family framing (slide 4).** "Three model families to isolate where performance comes from" → "the gap is precisely the value of pretraining" — this two-sentence pair is the spine of the entire video. Drill it.
3. **The handoff on slide 7.** Speak it three times slowly. This is the moment most likely to be re-shot.

---

## What to avoid in your delivery

- **Don't read any code line-by-line.** The slide shows the code; you explain the design decision behind it.
- **Don't read the lineup table on slide 4.** Narrate the *structure* (three families, three hypotheses, one refiner) and let the audience scan the names.
- **Don't say BLEU, ROUGE, CIDEr, or any metric number.** Those are Shaheer's.
- **Don't say "Show-and-Tell is a paper from 2015 by Vinyals et al."** — the citation is on the slide. You just need "the original Show-and-Tell trick" to anchor it.
- **Don't say "umm" between slides.** A silent click beats filler.

---

## Backup buffer

If you land at 1:50 (5s under budget), don't extend on the fly — just hand off to Hassan. Spare seconds redistribute automatically.

If you go *over* 1:55, the easiest cuts in order are:
1. The 30%-cleaning line on slide 3 (~5s).
2. "It's cheap enough to run on CPU" on slide 6 (~3s).
3. The final tanh-stability sentence on slide 7 (~4s) — leave just the two-linear-projections explanation and the handoff.
