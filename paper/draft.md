# Seeing or Restating? A Blind-Baseline Audit of Product Description Generation for Low-Resource Marketplaces

**Karam Hussain, Hassan Jabbar, Shaheer Shahid**

*Draft v1 — all numbers below are measured on this repository's code and data. Scripts to reproduce every table are in `models/novelty/`; result JSONs are in `models/results/`.*

---

## Abstract

Product-description generation for e-commerce is evaluated, in every prior work we identify, by n-gram similarity to the seller's existing description. We show this protocol is blind in two independent ways. **First, it cannot distinguish a model that sees the product photograph from one that does not:** on an identical Amazon Berkeley Objects (ABO) split, a text-only GPT-2 that never receives the image exceeds a fine-tuned BLIP on BLEU-4 (125.7% of the vision model's score, 95% CI [120.2, 130.9]) and is statistically indistinguishable on METEOR and BLEU-1; the partition replicates on a Pakistani marketplace corpus, where the blind model wins BLEU-1 outright by 2.9×. Image counterfactuals confirm the vision models genuinely use the photograph — replacing it with a blank rewrites 53% of a description while moving ROUGE-L by 0.5%. **Second, it cannot distinguish improving a listing from reproducing it:** because similarity to the incumbent text is the definition of success, the protocol structurally cannot reward a system for writing something better than what is on the platform — and models trained under it inherit the same ceiling. Measuring improvement directly with reference-free CLIPScore, converged VLMs fine-tuned on seller text plateau at or below their own incumbent (−7.3% to +0.7% across three configurations on a Pakistani marketplace corpus; −4.2% on Amazon's curated catalogue): a model trained to imitate its references cannot beat them. Repairing the supervision — rewriting training labels once with a large VLM, CapFilt-style — flips the same architecture to **+6.4% above the incumbent, while *lowering* its ROUGE-L**; the standard protocol ranks the only improving model last. We trace both failures to the same root cause — seller-authored references on open marketplaces are metadata restatements (20–24% recoverable from the model's own input) and mass-duplicated copy (56.8% of listings are re-listings of another item) — and release a two-corpus audit toolkit, a leakage-controlled 3,091-listing Pakistani marketplace dataset (DPD), and a corrected evaluation protocol. The system under audit runs at 1.2 s/description in under 1 GB of VRAM on one consumer GPU.

---

## 1. Introduction

### 1.1 Problem statement

Open multi-seller marketplaces — Daraz, Jumia, Shopee, Mercado Libre — serve most of the world's online shoppers, and their listings are notoriously poor: descriptions are spec-sheet dumps, warranty boilerplate, or verbatim copies of a competitor's listing even when the product differs. A seller-side generation system for this setting must satisfy three constraints simultaneously:

1. **It must improve on the incumbent listing**, not resemble it — the incumbent is the problem.
2. **It must run at marketplace economics** — per-listing frontier-API calls (≈$0.85 per 1,000 descriptions at current Gemini Flash rates) do not amortise at third-world seller margins; a local model at ≈$0.01 per 1,000 does.
3. **Its evaluation must not depend on trusting the platform's own text**, because that text is what is being replaced.

### 1.2 The gap

Prior work fails constraint 3 by construction, and therefore cannot measure constraints 1 at all. Every paper we identify on this task — from the Stanford CS231N study we replicate [1], through ModICT's 300K-sample Chinese-marketplace system [3], to MMPCBench's 2026 evaluation of six frontier multimodal LLMs [2] — scores generated descriptions by BLEU/ROUGE/CIDEr (or embedding similarity) against the seller-authored description. None runs the control that has been standard in VQA since Goyal et al. (2017) [4]: *remove the image and see what survives.* None asks whether the reference itself is good enough to be a target.

We run both controls. The results invalidate the standard protocol for this task and, in doing so, locate where a low-compute generation system actually earns its keep.

### 1.3 Contributions

1. **The first blind-baseline audit of e-commerce description generation** (§5): a text-only model trained under identical conditions matches or beats a fine-tuned VLM on three of five standard metrics, with bootstrap confidence intervals, on two corpora.
2. **An image-counterfactual protocol** (§6) separating *how much the output changes* from *how much the score notices*: blanking the photograph rewrites 53% of a CLIP-GPT2 description while moving ROUGE-L 0.5%, and removes 34% of a BLIP description's visual content while moving ROUGE-L 2.5%. The models are image-dependent; the metrics are not. The failure is in the measurement, not the models.
3. **A root-cause analysis** (§4): 20.3–23.6% of reference content tokens are recoverable from the metadata prompt; 56.8% of marketplace listings are re-listings (stable across a 2.8× corpus expansion); image-hash deduplication alone is only ~51% precise at detecting them.
4. **A corrected protocol** (§7): reference-free CLIPScore correlates ρ = +0.517 with independently judged visual grounding where ROUGE-L correlates ρ = −0.153, and a *headroom* measure (CLIPScore of output minus CLIPScore of incumbent) that expresses "the listing got better" — the sentence no reference-based metric can say.
5. **DPD, a leakage-controlled Pakistani marketplace dataset** (§3): 3,091 listings, 9 balanced categories, item-level (not image-level) deduplicated splits, plus a reproducible scraper.
6. **A deployment result** (§8): fine-tuning on seller text cannot beat the seller text — converged models sit at −7.3% to +0.7% headroom on the marketplace corpus and −4.2% on Amazon — while one round of supervision repair lifts the identical 224M-parameter architecture to **+6.4% above the incumbent**, delivered at 1.2 s/description, <1 GB VRAM, ≈$0.01/1,000 descriptions on one consumer GPU. Reference-based metrics order these models backwards.

---

## 2. Related work, and proof of the gap

We claim two specific absences in the literature: (a) no blind/no-image control in e-commerce description generation, and (b) no evaluation of improvement over the incumbent listing. The table below is the evidence; for each system we list what it evaluates against and whether either control appears.

| Work | Task / scale | Evaluation | Blind control? | Incumbent-improvement measure? |
|---|---|---|---|---|
| Stanford CS231N [1] | ABO, 113K listings; CLIP-GPT2, BLIP, OFA fine-tuning | BLEU, ROUGE-L, METEOR, CIDEr vs seller bullets | **No** | **No** |
| ModICT [3] | MD2T, ~300K Chinese marketplace samples; in-context multimodal tuning | BLEU, ROUGE, BERTScore, diversity, human | **No** (confirmed by direct inspection: baselines are MMPG, M-kplug, Oscar variants — all multimodal) | **No** |
| MMPCBench [2] | 9 Amazon categories × ~1K items; Qwen2.5-VL and Gemma-3 families | cosine/BERTScore vs reference + downstream recommenders | **No** | **No** |
| eBay VLM adaptation [5] | 15M listings, up to 120×H100 | LLM-as-judge, F1 on extraction | **No** | **No** |
| PRAISE [6] | Review-mining to patch descriptions | Manual rubric, F1 | n/a (text-only) | Partially (adds review facts) but no visual grounding |
| EcomEval [7] | Multi-task e-commerce LLM benchmark | Reference + judge | **No** | **No** |
| ECLIP [8] / RA-CLIP [9] | E-commerce contrastive pretraining | Zero-shot classification/retrieval | n/a (not generative) | n/a |
| Hallucination detection in listings [10] | Faithfulness of LLM-enriched listings | Precision/recall of hallucination flags | **No** | **No** |
| **This work** | DPD (3,091) + ABO subset (15K) | Blind baseline, image counterfactuals, CLIPScore headroom, LLM judge | **Yes** | **Yes** |

The blind control itself has precedent outside this task: Goyal et al. [4] used question-only baselines to expose language priors in VQA, motivating balanced VQA v2; a 2026 study applies no-image ablations to modern VLM *question answering* [11]; and text-only LLMs have been shown to beat same-scale VLMs on human-centred *decision* tasks [12]. Its application to product-description **generation** — where the confounder is not a language prior but the reference itself — is, to the best of an extensive search, new. On the metric side, CLIPScore [13] is established for captioning; its use to measure *improvement over an incumbent listing* (headroom, §7.2) is the new element, not the metric.

Supervision repair via model-generated labels is **not** claimed as a contribution: it is BLIP's own CapFilt [14], and current e-commerce practice (SynthAVE [15]; eBay's verification pipeline [5]). We use it (§8.2) and cite it.

---

## 3. Datasets

### 3.1 DPD — Daraz Product Descriptions (released)

Scraped from Daraz.pk with a Playwright-based crawler (automated slider-CAPTCHA handling, AJAX interception; `scraper/`). Two versions:

| | DPD-v1 (Milestone 2) | **DPD-v2 (this paper)** |
|---|---|---|
| Listings (clean, deduplicated) | 1,095 | **3,091** |
| Categories | 5 (11–29% each) | **9 (8.8–13.2% each)** |
| Test split | 135 | **309** |
| Duplicate records | — | **0** |
| Split construction | random | **item-level leakage-controlled** |

### 3.2 The duplication structure of a marketplace corpus

Grouping all 3,091 v2 listings into same-item clusters (title similarity ≥82 ∨ description similarity ≥70 ∨ (pHash Hamming ≤4 ∧ title ≥60); connected components; whole groups assigned to one split):

| Quantity | v1 (1,370 raw) | v2 (3,091) |
|---|---|---|
| Listings inside a multi-item group | 56.4% | **56.8%** |
| Same-item edges from copied *descriptions* | 2,220 | **8,790** |
| Same-item edges from matching titles | 511 | 971 |
| Residual test items with a same-item match in train | 0 | **0** |

Two findings. **(i)** More than half the catalogue is re-listings, stable across a 2.8× expansion into four new categories — a structural property of open marketplaces, impossible on curated one-listing-per-product catalogues like ABO. The dominant duplication channel is *copied description text* (9:1 over titles): sellers paste each other's copy. **(ii)** Image-hash deduplication alone — the standard practice, and this project's own original pipeline — is insufficient: of 61 image-duplicate pairs in v1, only 50.8% were genuinely the same item (61.4% at Hamming 0); the rest were distinct products sharing a stock photograph. Image-level dedup therefore both misses text-copied re-listings and discards legitimate listings. In v1's random split this left **23% of the test set as re-listings of training items**.

### 3.3 ABO control corpus

15,000 English listings sampled from Amazon Berkeley Objects [16] (147,702 listings; bullet-point references; main images at ≤256px), split 12,001/1,500/1,500. ABO is the corpus of the paper we replicate [1] and the field's de-facto benchmark.

---

## 4. Why the references cannot serve as targets

**Metadata leakage.** Tokenising each reference and each metadata prompt (stemmed content words), the share of reference tokens recoverable from the model's own conditioning input is:

| Corpus | Mean | Median | n |
|---|---|---|---|
| ABO (Amazon) | **23.6%** | 22.7% | 40,000 |
| DPD (Daraz) | 20.3% | 16.1% | 135 |

A quarter of the "ground truth" is the input, echoed. Consequently the metadata prompt **submitted verbatim, with no model and no image**, scores ROUGE-L 15.4 on ABO and 13.15 on DPD — on DPD, above both fine-tuned VLMs.

**An evaluation-harness hazard that follows.** `BlipForConditionalGeneration.generate` treats the text prompt as a decoder prefix, and `processor.decode(output_ids[0])` returns prompt + continuation. In the original pipeline this meant **100% of BLIP's scored outputs began with a verbatim copy of the prompt; 41.9% of the scored text was the prompt**. Because the reference partially *is* the prompt (above), this echo scores. Removing it flips the reported model ranking on ROUGE-L, METEOR, and CIDEr (BLIP 12.70 → 8.24 ROUGE-L; CLIP-GPT2, generated via `inputs_embeds` and hence echo-free, wins all five metrics). Any evaluation calling this API with a prompt inherits the hazard.

**Reference visual content.** By a 149-stem visual lexicon (colour/material/texture/shape/part terms), ABO references are 15.2% visual tokens; DPD references are 8.8%. Strip the metadata-derivable part from a DPD reference and the remainder is 7.0% visual — warranty terms, dimensions, shipping boilerplate. There is little visual signal in the target for a reference-based metric to reward.

---

## 5. The blind baseline

**Setup.** On the identical ABO split: (a) *vision* — BLIP-base [14] (224M params), image + 64-token metadata prompt → description; (b) *blind* — GPT-2 (124M), the same 64-token prompt → description, **no image, no vision encoder, ever**. Identical target budget (128 tokens), loss masking (description tokens only), optimiser (AdamW 1e-5, cosine), epochs, and decoding (beam 4, no-repeat-3-gram). Scored on the same 600 held-out items with prompt prefixes stripped from both. Ratios bootstrap-resampled (600 resamples).

| Metric | Blind | Vision | Blind/Vision | 95% CI | Verdict |
|---|---|---|---|---|---|
| BLEU-4 | 35.11 | 27.93 | **125.7%** | [120.2, 130.9] | **blind wins** |
| METEOR | 52.77 | 51.72 | 102.0% | [98.7, 104.7] | indistinguishable |
| BLEU-1 | 40.98 | 41.66 | 98.4% | [94.2, 102.5] | indistinguishable |
| ROUGE-L | 45.20 | 63.13 | 71.6% | [69.5, 73.4] | vision wins |
| CIDEr | 7.65 | 23.84 | 32.1% | — | vision wins |

**Three of the five standard metrics cannot distinguish a model that sees the product from one that cannot.** On BLEU-4 the blind model *significantly wins*: with the image-metadata echo unavailable, it spends its full capacity matching reference phrasing, which BLEU-4 rewards. For context, the paper we replicate reports BLEU-4 = 36.32 for BLIP fine-tuned on 113K ABO listings [1]; our blind model reaches 35.11 on one-tenth the data with no image (indicative only — different split and code; our controlled comparison on a shared split is the table above).

**Replication on the marketplace corpus (DPD-v2, leakage-controlled, n = 309).** Blind GPT-2 and vision BLIP trained on the identical 2,472-item item-deduplicated split, matched epoch budget (12, best-validation), markup-cleaned text:

| Metric | Blind | Vision | Winner |
|---|---|---|---|
| BLEU-1 | **7.77** | 2.72 | blind |
| BLEU-4 | **0.70** | 0.18 | blind |
| METEOR | **9.11** | 8.37 | blind (≈tie) |
| ROUGE-L | 9.48 | **12.05** | vision |
| CIDEr | 1.00 | **3.29** | vision |

The metric partition **replicates across both corpora**: BLEU-1, BLEU-4, and METEOR are blind-solvable; ROUGE-L and CIDEr retain visual sensitivity. On the marketplace corpus the blind model does not merely tie the BLEU metrics — it wins them outright (2.9× on BLEU-1), because with more metadata echo in the references there is more for a text-only model to harvest. (On v1's leaky 135-item split the same qualitative pattern held: blind wins BLEU-1 5.74 vs 3.00 and CIDEr 1.06 vs 0.70 while BLIP's output carried 4.8× the visual vocabulary.)

---

## 6. The models see; the metrics don't look

If the blind result meant "vision is useless," it would be a model finding. The counterfactuals show it is a metric finding.

**Blank-image counterfactual (ABO, n=200–400).** The same trained BLIP, with its photograph replaced by a uniform grey square, loses **49.0% ROUGE-L, 67.2% BLEU-1, 54.2% METEOR, 79.4% CIDEr** (12K-sample model; 28.7–92.7% at the 873-sample size). The model is strongly image-dependent, and on ABO — whose references are professionally written and visually specific — the metrics register it.

**The divergence, on the marketplace corpus.** Here the same counterfactual separates *how much the output changes* from *how much the score notices*. VSS (Visual Sensitivity Score) is 100 − ROUGE-L(output with real image, output with blank image): it measures how much of the generated text is image-driven, independent of any reference.

| Model / prompt | VSS — how much the text changes | ROUGE-L change | n |
|---|---|---|---|
| CLIP-GPT2, full metadata prompt | **52.9** | **−0.5%** | 135 |
| CLIP-GPT2, image-forced prompt | **55.3** | −2.1% (*wrong direction*) | 135 |
| BLIP, full metadata prompt | 41.3 | −2.3% | 135 |
| BLIP, image-forced prompt | 74.0 | −25.3% | 135 |
| BLIP-v2, full prompt, converged | — | −2.5% | 309 |

Read the top row. **Replacing the product photograph with a grey square rewrites 53% of CLIP-GPT2's description, and ROUGE-L moves half a percent.** With the image-forced prompt the text changes even more (55%) and the score moves *the wrong way*. This is the paper's thesis in a single controlled measurement on one model: the output is demonstrably image-driven, and the field's headline metric is very nearly invariant to it.

Corroborating on the v2 corpus (n = 309, converged): blanking the image removes **34%** of the output's visual-lexicon content (8.29 → 5.50) and 20% of CIDEr, while ROUGE-L moves 2.5%.

The magnitude of the ROUGE-L drop is training-config-sensitive (2.5–28.7% across configurations; an early 1,095-sample run measured 28.7% [CI 16.6–40.1]). The stable and honest statement is not any single drop percentage but the **divergence itself**: across every model, prompt, corpus and training configuration we tested, the text changes substantially (VSS 41–87) while ROUGE-L moves ≤2.5% — except under the image-forced prompt, where metadata is withheld and the metric finally registers vision.

---

## 7. A protocol that measures the right thing

### 7.1 Metric validity against an independent judge

All 270 image-forced (category-prompt-only) generations from both DPD VLMs were independently scored 1–5 for visual grounding by Gemma-3-27B-class judge (Spearman ρ, n = 270):

| Metric | ρ vs judged visual grounding | Reference needed? |
|---|---|---|
| ROUGE-L | **−0.153*** | yes |
| Residual ROUGE-L (metadata removed) | −0.202* | yes |
| Visual-lexicon recall of reference | −0.282* | yes |
| Visual lexicon density (ours, deprecated) | +0.417* | no |
| **CLIPScore [13]** | **+0.517*** | **no** |

Every reference-based variant — including two corrections we designed ourselves — correlates *negatively* with judged grounding, because the reference itself is not visual (§4). CLIPScore, which compares the text to the *image*, correlates at +0.517. We accordingly deprecate our own lexicon metric in favour of the established one (they barely correlate with each other, ρ = 0.08: naming visible things ≠ matching this image). *Caveat:* CLIPScore rewards naming the depicted object, not prose quality — a bag-of-visible-nouns would score high — so it must be paired with a fluency/faithfulness judge; and the judge itself awaits human validation (§9).

### 7.2 Headroom: did the listing get better?

Reference-based metrics define success as *resembling* the incumbent. The deployment question is *beating* it:

**headroom = CLIPScore(generated, image) − CLIPScore(incumbent seller text, image)**

Measured across every properly-converged configuration (within-corpus deltas; absolute CLIPScores are not comparable across corpora):

| Model (training target) | Corpus / split | n | Headroom |
|---|---|---|---|
| BLIP, **seller labels** | DPD-v1 clean split | 137 | **−7.3%** |
| BLIP, seller labels | DPD-v2, full-metadata prompt | 309 | +0.2% |
| BLIP, seller labels | DPD-v2, image-forced prompt | 309 | +0.7% |
| BLIP, seller labels | ABO (curated) | 600 | **−4.2%** |
| BLIP, **repaired labels** [14-style] | DPD-v1 clean split | 137 | **+6.4%** |

Two findings, sharper than a corpus contrast. **(i) Imitative fine-tuning is ceiling-bounded by its own references.** Every model trained to reproduce seller text converges to at-or-below the incumbent's image alignment — on the marketplace *and* on Amazon. This is the training-side face of the evaluation problem: the reference is simultaneously the metric's yardstick and the model's target, so neither the metric nor the model can exceed it. (Earlier, undertrained checkpoints showed apparent +9–11% headroom; that gain vanishes at convergence and was an artifact of verbose, weakly-fit outputs. We report it retracted.) **(ii) Supervision repair breaks the ceiling and the standard metric punishes it.** The one configuration that beats the incumbent (+6.4%, and 13.7 points above its seller-label twin) is the model trained on VLM-rewritten labels — the model whose ROUGE-L is *lower* (8.33 vs 9.05). Under the field's protocol, the only model that improves the platform ranks last.

The refiner stage as currently trained also subtracts value (visual density 18.1 → 9.9 across stages on v1); for deployment the image-conditioned generator with repaired supervision is the system.

Differentiation, measured against the whole live corpus: **43.7% of incumbent seller listings are near-duplicates (fuzzy ≥70) of another listing on the platform; every generative system tested produces 0%** — generation per se solves the copying problem, and this too is invisible to reference-based scoring, which *rewards* proximity to the (duplicated) reference.

---

## 8. The audited system, at marketplace economics

### 8.1 Compute profile (measured, RTX 5060 Ti, batch 1, beam 4)

| | BLIP (vision) | CLIP-GPT2 |
|---|---|---|
| Parameters | 224M | 218M |
| s / description (GPU) | 1.35 | 1.20 |
| Peak VRAM | **0.60 GB** | 0.94 GB |
| Descriptions / hour | 2,662 | 2,997 |
| Electricity / 1,000 (180W TDP bound, $0.15/kWh) | **$0.010** | $0.009 |
| s / description (CPU only) | 4.17 | 3.69 |

One mid-range consumer GPU covers ≈64,000 listings/day — the order of a national marketplace's daily intake — at ~1% of frontier-API listed cost, and the system degrades gracefully to CPU-only. This is not a claim of superior quality to frontier models (untested here; §9); it is the deployment envelope in which the +6.4% headroom of §7.2 is delivered.

### 8.2 Supervision repair (the method is CapFilt's [14]; the measurement is ours)

Because the marketplace's own text is both training target and evaluation reference, and that text is corrupted (§4), we rewrite the 1,100 training labels once with a large VLM — a one-time cost that amortises over unlimited local inference. Against its seller-label twin (identical split, identical seller-text test references), the repaired-label model: output visual density 9.14 → **24.71** (2.7×); CLIPScore headroom over the incumbent **−7.3% → +6.4%**; ROUGE-L 9.05 → **8.33 (falls)**. The intervention that produces the only above-incumbent model in the study is the one the standard protocol scores as a regression — a self-contained instance of the paper's thesis, measured.

---

## 9. Limitations and open items

- **Human validation is outstanding.** The judge is one LLM; CLIPScore's ρ = +0.517 is judge-relative. A three-rater blinded study (materials built: `models/results/human_study/rating_sheet.html`, 60 items × 4 systems) is the single highest-priority remaining experiment.
- **No frontier-model comparison.** The Pareto claim (§8) currently bounds cost, not quality-per-cost; a Gemini-Flash pass over the DPD test set is scripted (`models/novelty/frontier_baseline.py`) and blocked only on a live API key.
- **Config-sensitive results are flagged, not hidden.** An early marketplace-vs-Amazon image-contribution contrast did not survive pipeline-matched retraining (bootstrap p = 0.91; retracted); the DPD image-contribution drop varies 2.5–28.7% across training configs; and apparent +9–11% headroom from undertrained checkpoints vanished at convergence (retracted, §7.2). The stable claims are the two-corpus blind-baseline metric partition, the leakage/duplication measurements, the seller-label headroom ceiling, and the repaired-label headroom gain.
- **The "training teaches echo" hypothesis is refuted by our own probe:** across the five saved training epochs, prompt-echo *falls* (58.8% → 54.1%) and image-sensitivity *rises* (VSS 40.1 → 45.5). Longer training modestly improves grounding; the headroom ceiling of §7.2 is a property of the training *target*, not of training duration.
- **CLIPScore limits:** object-naming bias; 77-token truncation; CLIP's known weakness on fine-grained attributes.
- **Two-stage refiner currently harms headroom** and leaks metadata field names into output; fixing it (e.g., echo-suppressed decoding, measured to cut prompt-echo 10% and raise visual density 9% with no retraining) is future work.
- Single platform per market type; Urdu/Roman-Urdu content untreated.

## 10. Conclusion

The standard evaluation for product-description generation cannot see the two things that matter: whether the model looked at the product, and whether the listing improved. We showed both blindnesses with controlled experiments on two corpora — a blind model wins or ties three of five metrics everywhere we test, and every model trained to imitate seller text plateaus at its incumbent's quality on marketplace and curated catalogues alike. We traced both to the same root: on open marketplaces the reference text is metadata echo and mass-duplicated copy, and it serves as training target and evaluation yardstick at once. Under the corrected protocol — blind baselines, image counterfactuals, CLIPScore headroom against the incumbent, item-level-deduplicated splits — the path to actual improvement is visible and cheap: one round of supervision repair lifts a 224M-parameter model 6.4% above the marketplace's own listings, on one consumer GPU, while the field's metrics score that model as the worst in the study. For the markets where automated description generation is worth the most, the field has been measuring — and training on — the wrong thing.

---

## References

1. Zhao, He, Liu. *Automated Product Description Generation for E-commerce via Vision-Language Model Fine-tuning.* Stanford CS231N, 2024. https://cs231n.stanford.edu/2024/papers/automated-product-description-generation-for-e-commerce-via-visi.pdf — **reference-based evaluation only; no blind control.**
2. *MMPCBench: Benchmarking Multimodal LLMs for Missing Modality Completion in Product Catalogues.* arXiv:2601.19750, 2026. https://arxiv.org/html/2601.19750 — **six modern MLLMs on image→description; confirmed no blind/no-image ablation.**
3. *ModICT: A Multimodal In-Context Tuning Approach for E-Commerce Product Description Generation.* arXiv:2402.13587, 2024. https://arxiv.org/abs/2402.13587 — **~300K-sample marketplace dataset; all baselines multimodal; confirmed no text-only control, no reference-quality discussion.**
4. Goyal, Khot, Summers-Stay, Batra, Parikh. *Making the V in VQA Matter.* CVPR 2017. https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html — the blind-control precedent, in VQA.
5. *Adapting Vision-Language Models for E-Commerce Understanding at Scale.* arXiv:2602.11733, 2026. https://arxiv.org/abs/2602.11733 — 15M listings, ≤120 H100; judge/F1 evaluation.
6. *PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights.* ACL 2025 demo. https://arxiv.org/html/2506.17314v1
7. *EcomEval: Towards Reliable Evaluation of LLMs for E-commerce.* arXiv:2510.20632, 2025. https://arxiv.org/pdf/2510.20632
8. *ECLIP: Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-commerce.* CVPR 2023. https://cvpr.thecvf.com/virtual/2023/poster/22582
9. Xie et al. *RA-CLIP: Retrieval Augmented Contrastive Language-Image Pre-training.* CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Xie_RA-CLIP_Retrieval_Augmented_Contrastive_Language-Image_Pre-Training_CVPR_2023_paper.html
10. *Hallucination Detection in LLM-enriched Product Listings.* ECNLP @ 2024. https://aclanthology.org/2024.ecnlp-1.4.pdf
11. *Do Vision-Language Models See or Guess? Measuring and Reducing Textual-Prior Reliance.* arXiv:2606.10400, 2026. https://arxiv.org/html/2606.10400 — no-image ablation for VQA accuracy; **no generation, no e-commerce.**
12. *When Words Outperform Vision: VLMs Can Self-Improve Via Text-Only Training.* arXiv:2503.16965, 2025. https://arxiv.org/abs/2503.16965 — text-only beats VLM on decision tasks; nearest cross-domain precedent.
13. Hessel, Holtzman, Forbes, Le Bras, Choi. *CLIPScore: A Reference-free Evaluation Metric for Image Captioning.* EMNLP 2021. https://arxiv.org/abs/2104.08718
14. Li, Li, Xiong, Hoi. *BLIP: Bootstrapping Language-Image Pre-training.* ICML 2022. https://arxiv.org/abs/2201.12086 — CapFilt = the label-repair precedent.
15. *SynthAVE: Scalable Synthetic Labeling for E-Commerce with LLM-Arena Validation.* arXiv:2607.07469, 2026. https://arxiv.org/html/2607.07469
16. Collins et al. *ABO: Dataset and Benchmarks for Real-World 3D Object Understanding.* CVPR 2022. https://amazon-berkeley-objects.s3.amazonaws.com/index.html

### Novelty statement

We searched arXiv, ACL Anthology, CVF Open Access, and general web (July 2026) for: blind/no-image/text-only controls in e-commerce description generation; reference-quality audits of seller-authored targets; and incumbent-relative (headroom) evaluation of generated listings. The closest works are [4] (blind control, but VQA), [11] (no-image ablation, but VQA accuracy), [12] (text-only wins, but decision-making), and [13] (reference-free metric, but never applied against an incumbent listing). Direct inspection of [1], [2], and [3] — the three works nearest to our task — confirms none contains a text-only baseline, an image ablation, or any reference-quality analysis. We are not aware of any prior measurement of (a) blind-vs-vision parity on standard metrics for this task, (b) marketplace re-listing rates as a property of description-generation corpora, or (c) headroom-over-incumbent as an evaluation target.

### Reproducibility

Every number in this paper is produced by a script in `models/novelty/` and written to `models/results/novelty_*.json`. Nothing in the original pipeline was modified: the Milestone-2 checkpoints, splits, and result files remain on disk as the audited baseline, and every experiment is additive.

| Claim | Script | Output |
|---|---|---|
| Blind baseline, ABO (§5) | `abo_head2head.py` | `novelty_abo_head2head.json` |
| Blind-ratio confidence intervals (§5) | ad-hoc bootstrap over saved generations | `novelty_blind_ratio_ci.json` |
| Blind baseline, DPD-v2 (§5) | `blind_v2.py` + `dpd_clean_train.py --v2` | `novelty_blind_v2_metrics.json`, `novelty_dpd_clean_result_v2b.json` |
| Image counterfactuals (§6) | `visual_sensitivity.py` | `novelty_visual_sensitivity.json` |
| Counterfactual CIs (§6) | `bootstrap_ci.py` | `novelty_bootstrap_ci.json` |
| Metadata leakage (§4) | `echo_audit.py`, `abo_replication.py` | `novelty_echo_audit.json`, `novelty_abo_replication.json` |
| Prefix-leakage hazard (§4) | `prompt_prefix_audit.py` | `novelty_prompt_prefix_audit.json` |
| Duplication structure (§3) | `duplicate_audit.py`, `duplicate_diagnosis.py`, `clean_split.py --v2` | `novelty_clean_split_v2.json` |
| Metric validity vs judge (§7.1) | `metric_validity.py`, `clipscore_compare.py` | `novelty_clipscore_compare.json` |
| Headroom (§7.2) | `headroom.py` | `novelty_headroom.json`, `novelty_headroom_v2*.json` |
| Supervision repair (§8.2) | `dpd_clean_train.py --augmented` | `novelty_dpd_clean_result_aug.json` |
| Compute profile (§8.1) | `compute_profile.py` | `novelty_compute_profile.json` |
| Epoch dynamics (§9) | `epoch_dynamics.py` | `novelty_epoch_dynamics.json` |
| Frontier baseline (§9, pending) | `frontier_baseline.py` | blocked on API key |

Data: DPD v1 (`data/data/processed/`) and v2 (`data/processed/`, 3,091 listings, 9 categories, item-deduplicated splits in `splits_clean/`); ABO subset via `abo_build_subset.py`. Scraper, cleaner, and deduplicator in `scraper/`, `pipeline/`, `dedup/`.
