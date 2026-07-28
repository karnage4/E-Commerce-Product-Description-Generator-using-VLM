# Novelty experiments — assembled results

## N9 Prompt-prefix leakage (headline correction)

**BLIP fine-tuned** — 100.0% of outputs open with a verbatim prompt prefix; 41.9% of the scored text is the prompt.

| Metric | as scored | prompt removed |
|---|---|---|
| BLEU-1 | 7.94 | 3.0 |
| BLEU-4 | 0.54 | 0.1 |
| ROUGE-L | 12.7 | 8.24 |
| METEOR | 12.02 | 5.99 |
| CIDEr | 1.4 | 0.7 |

**CLIP-GPT2 fine-tuned** — 0.7% of outputs open with a verbatim prompt prefix; 0.0% of the scored text is the prompt.

| Metric | as scored | prompt removed |
|---|---|---|
| BLEU-1 | 12.09 | 10.99 |
| BLEU-4 | 0.92 | 0.43 |
| ROUGE-L | 10.28 | 10.28 |
| METEOR | 10.53 | 11.45 |
| CIDEr | 0.97 | 1.14 |

Rankings that flip once compared like-for-like: **ROUGE-L, METEOR, CIDEr**

## N7/N8 Metric validity against the LLM judge

| Metric | Spearman rho vs judged visual grounding | Reference-free? |
|---|---|---|
| ROUGE-L | -0.153 | no |
| Residual ROUGE-L | -0.202 | no |
| Visual Residual Recall | -0.282 | no |
| Echo Rate | 0.401 | no |
| **Visual Attribute Density** | **0.417** | yes |

| Text | Visual Attribute Density |
|---|---|
| Reference (seller-written) | 8.75 |
| Metadata-Only (template) | 4.85 |
| BLIP fine-tuned | 12.63 |
| CLIP-GPT2 fine-tuned | 15.54 |
| BLIP Stage-1 (category-only prompt) | 18.17 |
| CLIP-GPT2 Stage-1 (category-only) | 18.72 |
| Two-Stage BLIP (stage2) | 8.16 |

## N1 Metadata-echo audit

- 20.32% of the average reference's content tokens also occur in the metadata prompt (median 16.09%).

| System | ROUGE-L | Residual ROUGE-L | Metadata Recall | Visual-Residual Recall | Echo Rate |
|---|---|---|---|---|---|
| Metadata-Only (template) | 13.15 | 4.81 | 100.0 | 0.0 | 100.0 |
| BLIP fine-tuned | 12.7 | 8.52 | 100.0 | 4.37 | 56.68 |
| CLIP-GPT2 fine-tuned | 10.28 | 8.75 | 54.25 | 7.79 | 18.99 |
| Two-Stage BLIP (stage2) | 8.87 | 7.78 | 42.46 | 7.31 | 20.59 |
| Two-Stage BLIP (stage1) | 7.44 | 6.42 | 21.75 | 3.51 | 28.35 |
| Two-Stage CLIP-GPT2 (stage2) | 7.15 | 7.2 | 14.04 | 6.24 | 8.56 |

## N2 Catalogue near-duplicate contamination

- 61/135 test items have a train image within pHash Hamming 8.
- Hamming distribution: {'9-16': 64, '17+': 10, '0 (identical)': 44, '1-4': 10, '5-8': 7}

| System | Partition | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|---|---|---|---|---|---|---|
| pHash Retrieval (copy nearest train desc) | ALL | 30.43 | 23.76 | 26.09 | 26.16 | 135.71 |
| pHash Retrieval (copy nearest train desc) | CONTAMINATED | 49.96 | 43.0 | 47.41 | 47.39 | 296.16 |
| pHash Retrieval (copy nearest train desc) | CLEAN | 11.0 | 4.71 | 8.51 | 8.66 | 3.21 |
| BLIP fine-tuned | ALL | 7.94 | 0.54 | 12.7 | 12.02 | 1.4 |
| BLIP fine-tuned | CONTAMINATED | 8.33 | 0.3 | 13.25 | 12.52 | 0.72 |
| BLIP fine-tuned | CLEAN | 7.51 | 0.68 | 12.26 | 11.61 | 1.93 |
| CLIP-GPT2 fine-tuned | ALL | 12.09 | 0.92 | 10.28 | 10.53 | 0.97 |
| CLIP-GPT2 fine-tuned | CONTAMINATED | 13.73 | 0.94 | 11.18 | 11.35 | 1.1 |
| CLIP-GPT2 fine-tuned | CLEAN | 10.18 | 0.83 | 9.54 | 9.86 | 0.89 |

## N3 Visual counterfactual sensitivity

| Model / prompt | ROUGE-L real | ROUGE-L blank | ROUGE-L mismatch | VSS blank | VSS mismatch | ROUGE-L lost by deleting image |
|---|---|---|---|---|---|---|
| BLIP|full | 12.7 | 12.41 | 11.57 | 41.33 | 45.12 | 0.29 |
| BLIP|stage1 | 7.44 | 5.56 | 5.5 | 73.95 | 74.78 | 1.88 |

## N4 Echo-suppressed decoding

| lambda | ROUGE-L | Residual ROUGE-L | Visual-Residual Recall | Echo Rate | VSS | CIDEr | mean words |
|---|---|---|---|---|---|---|---|
| 0.0 | 11.63 | 7.69 | 3.77 | 58.6 | 39.59 | 1.34 | 91.4 |
| 2.0 | 11.29 | 7.62 | 3.57 | 54.55 | 41.99 | 0.49 | 91.4 |
| 4.0 | 11.16 | 7.58 | 3.89 | 52.7 | 42.48 | 0.64 | 92.5 |
| 8.0 | 11.13 | 7.54 | 3.8 | 53.0 | 42.32 | 0.62 | 91.7 |

## N5 Compute profile

| Config | params (M) | s/description | descriptions/hour | kWh per 1k | USD electricity per 1k | peak VRAM (GB) |
|---|---|---|---|---|---|---|
| BLIP [cuda] | 224.0 | 1.352 | 2662 | 0.068 | 0.0101 | 0.6 |
| CLIP-GPT2 [cuda] | 217.8 | 1.201 | 2997 | 0.06 | 0.009 | 0.94 |
| BLIP [cpu] | 224.0 | 4.166 | 864 | 0.208 | 0.0312 | None |
| CLIP-GPT2 [cpu] | 217.8 | 3.694 | 974 | 0.185 | 0.0277 | None |
