# CNN+LSTM Show-and-Tell Baseline — Qualitative Test Samples

**Checkpoint:** `models/checkpoints/baseline_cnn_lstm/best.pt` (20 epochs, val_loss 4.86).
**Decoding:** greedy + `no_repeat_ngram_size=3`, banned ids `{<unk>, <pad>, <bos>, <sep>}`, `max_new_tokens=80`.
**Test split:** 135 samples.
**Metrics:** BLEU-1 4.65 / BLEU-4 0.34 / ROUGE-L 7.97 / METEOR 5.22 / CIDEr 0.00.
**Full predictions:** `models/results/baseline_cnn_lstm_predictions.jsonl`.

Samples below were drawn with `random.seed(7)` from each category (2 each except
home-appliances which has only one test row).

---

## 1. consumer-electronics — item_id 469947358 (LED grille light)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new original packaging warranty is void in good condition . • easy to wear . • suitable for women , and comfortable and comfortable • perfect . • casual wear .

**REFERENCE:**
> Easy to Install: Plug & play, no drilling required. Just insert the light into a mesh hole, you can mount it flexibly anywhere on your vehicle's exterior surface or grille. Recommend hooking up the power before installation to make sure the grille light works. Note: The two grille lights installed in the middle can be distinguished according to the shape.

---

## 2. consumer-electronics — item_id 3155603 (UPS battery additive)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new . • new original packaging warranty is a

**REFERENCE:**
> Safe Blister Packing Gives maximum battery performance and get the most out of your batteries with increased conductivity and corrosion prevention This product can work on any 12 volts UPS Battery Great for Big vehicle, like trucks, trailer and more

---

## 3. home-appliances — item_id 470823582 (rechargeable fan)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new . • new original packaging warranty is a

**REFERENCE:**
> PLEASE NOTE: Fan are measured according to blade dia, 6 inch size means that blade diameter would be 6 inch Available Sizes: 6 inch, 8inch (Select from varaint) Power Type: Rechargeable Specification: 66.5*42.5 * 45.5cm Rated Voltage: 5 Rated Power: 7.5 Fan Speed Mode: 2 Gear Operation Mode: Mechanical Whether Charging Is Supported: Supported Packing Volume: 21.4*6.7 * 21.7cm Color: White, Gray, P…

---

## 4. smartphones — item_id 924652536 (Samsung Galaxy A07)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new . • new original packaging warranty is a

**REFERENCE:**
> Samsung Galaxy A07 detailed specifications Build OS Android 15 OS, up to 6 major Android upgrades UI OneUI 7 Dimensions 164.4 x 77.4 x 7.6 mm Weight 184 g SIM Nano-SIM + Nano-SIM Colors Gray, Light Violet, Dark Green Frequency 2G Band SIM1: GSM 850 / 900 / 1800 / 1900 SIM2: GSM 850 / 900 / 1800 / 1900 3G Band HSDPA 850 / 900 / 2100 4G Band LTE band 1(2100), 3(1800), 5(850), 7(2600), 8(900), 20(800), 38(2600)…

---

## 5. smartphones — item_id 538571318 (OnePlus, used)

**GENERATED:**
> brand : 6 . 0 ghz cortex - core ( 6 . 1 x 8 . 0 . 0 x 8 ghz cortex . 0 - inch display : 6 months warranty ( t - sim ) , dual sim , dual standby , dual , and a

**REFERENCE:**
> We Import Used Phones From Overseas, It's Not Used In Pakistan. All one plus models are comming with online approved Display: 6.41″ Optic AMOLED, HDR10 Processor: Qualcomm Snapdragon 855 Ram/Rom: 6-8-12GB/128-256GB Camera: 48 MP Main/16 MP Selfie Sim: Dual Sim | PTA Server Approve Processor Sensors Special Features

---

## 6. tablets — item_id 783005346 (Amazon Fire 7)

**GENERATED:**
> brand – samsung model : pre - core ( 6 . 3 ghz cortex - inch display : 6 . 5 x 8 . 0 ghz cortex . 0 - inch screen size : 6 months warranty ( t - sim ) , dual sim , dual standby , dual , and a preinstalled operating system product will be in daraz like new products are in good condition . they feel like new . fully functional with all features working

**REFERENCE:**
> Brand – Amazon Model – Fire 7 CPU – Quad-core 1.3 GHz Cortex-A7 Chipset – Mediatek MT8127 (28 nm) Camera – 2 MP main camera and rear-facing cameras with 720p HD video recording Memory – RAM- 1GB | ROM 16GB with microSD card slot Display – Screen size 7 inches with 600 x 1024 pixels Battery – Non-Removable Lithium-Ion (Li-Ion) 2980 mAh Operating System – Android 9.0 UI Dimensions – 191 x 115 x 10.6…

---

## 7. tablets — item_id 350628990 (Samsung Galaxy Tab A)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new original packaging warranty is void in good condition . • easy to wear . • suitable for women , and comfortable and comfortable • perfect . • casual wear .

**REFERENCE:**
> Brand - Samsung Model - Samsung Galaxy Tab A Model: Sm-T387 8.0 inch Display BLACK Color 2GB RAM 32GB Memory Android 9 5MP main camera 2MP selfie camera Processor Quad-core 1.3 GHz Cortex-A53 6 Months Warranty (T&C apply) Daraz Like New products are in Good Condition. They feel like new. Fully functional with all features working, with minimal scratches, minor usage but no damage. A preinstalled o…

---

## 8. womens-fashion — item_id 509476737 (embroidered kurti)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • easy to wear . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new original packaging warranty is void in good condition . • no damage . • suitable for women , and comfortable , and durable . • high - quality and comfortable fabric and

**REFERENCE:**
> Brand: No Brnd White Embroidered Kurti 3 Piece Suit Embroidered Organza Dupatta Fabric/Stuff: Organza (inner Malai) Shirt Chest: 20/21" Shirt Length: 37/38" Trouser Length: 38/39"

---

## 9. womens-fashion — item_id 252197633 (fleece hoodie)

**GENERATED:**
> brand : pre - wrap ; } • perfect for women and comfortable . • perfect and comfortable for women . • easy to wear . • ideal for women • perfect , and it ' s a preinstalled operating system product will be in daraz like new original packaging warranty is void in good condition . • no damage . • suitable for women , and comfortable , and durable . • high - quality and comfortable fabric and

**REFERENCE:**
> Size (small, medium, large, extra large) regular fit Fleece fabric Hooded Full-sleeves Winter collection Casual wear 96 percent Fleece colours (black, blue, grey Stay cozy and stylish with our Black Always Forever Fleece Full Sleeves Zipper Hoodie for Women. Made from soft and warm fleece fabric, this hoodie is perfect for those chilly days. The full sleeves provide extra warmth and comfort, whil…

---

## Three failure modes observed

1. **Mode collapse to a generic commercial template** — ~⅔ of test items receive
   the same `brand : pre - wrap ; } • perfect for women and comfortable...` output
   regardless of category. The image feature initialises the LSTM state but is too
   weak to override the language-model prior past the first token.

2. **Category-conditioned spec-sheet template** (smartphones/tablets) — the model
   has learned that phone/tablet descriptions follow a
   `<brand> <model> <CPU> <inch display> <warranty> <SIM> <Daraz Like New>` slot
   pattern. It correctly emits the slot structure but the values are randomly
   recombined and hallucinated (`brand – samsung` for an Amazon Fire 7, fake GHz
   numbers, fake screen sizes).

3. **Unescaped HTML leakage** — `pre - wrap ; }` traces back to the literal CSS
   string `<pre>{ white-space: pre-wrap; }` embedded unescaped in many raw Daraz
   listings. The baseline tokeniser treats it as ordinary text and learns it as
   high-frequency vocabulary. Data-quality signal:
   `pipeline/cleaner.py` should strip raw HTML/CSS fragments before retraining.

## What it is *not* doing

No genuine image grounding. Across all nine inspected outputs, not a single
colour, material, or visually-derived attribute appears that could only have
come from the image. Every fashion item, every fan, every battery additive,
every smartphone gets either the same generic blurb (mode 1) or the same
randomly-instantiated spec template (mode 2).

## Why this is the correct baseline behaviour for the report

This is exactly what "what you get without large-scale vision-language
pretraining" looks like. The BLIP and CLIP+GPT-2 numbers in Table 2 of
`milestone2_report.tex` now have a defensible floor to be compared against.
The BLEU-1 jump from 4.65 (CNN+LSTM) → 7.94 (BLIP) → 12.09 (CLIP+GPT-2)
quantifies the contribution of vision-language pretraining on this dataset.
