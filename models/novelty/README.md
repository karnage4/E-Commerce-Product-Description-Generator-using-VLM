# Novelty experiments

Additive diagnostics for the DPD pipeline. **Nothing here modifies training code,
evaluation code, or checkpoints** — the Milestone 2 scripts and results stay intact
as the benchmark. Every script reads existing artefacts and writes new files into
`models/results/novelty_*`.

## Why these exist

Milestone 2 reported BLIP as the improved model (ROUGE-L 12.70) over CLIP-GPT2
(10.28). These experiments test whether that number, and the benchmark it comes
from, measure description quality at all. They do not.

## The findings, shortest form

| # | Script | Finding |
|---|---|---|
| N9 | `prompt_prefix_audit.py` | **100%** of BLIP's saved outputs open with a verbatim copy of the metadata prompt; **41.9%** of the scored text *is* the prompt. CLIP-GPT2 has no prefix (it generates via `inputs_embeds`). Comparing like-for-like, **CLIP-GPT2 wins all five metrics** — ROUGE-L, METEOR and CIDEr flip. |
| N1 | `echo_audit.py` | **20.3%** of the average reference's content words also appear in the metadata prompt. The metadata template alone scores ROUGE-L 13.15, beating both fine-tuned models. Ranking by ROUGE-L ≈ ranking by echo rate. |
| N2 | `duplicate_audit.py` | **33%** of test products have a pixel-identical image in train (pHash Hamming 0); 45% within Hamming 8. A pHash copy baseline scores CIDEr **296** on contaminated items and **3.2** on clean ones. Generative models are flat across both. |
| N3 | `visual_sensitivity.py` | Blanking the image costs BLIP **0.29** ROUGE-L out of 12.70 (2.3%) with the full metadata prompt, but **1.88** out of 7.44 (25.3%) with the category-only prompt. Measured on the continuation alone, BLIP's own text changes by **68%** when the image is swapped — the model sees, the metric does not count it. |
| N7 | `metric_validity.py` | Against the Gemma-4-31B judge's per-item grounding score: ROUGE-L ρ = **−0.15**, Residual ROUGE-L **−0.20**, Visual-Residual Recall **−0.28**. Every reference-based metric, including our own proposed corrections, is negatively related to judged grounding. |
| N8 | `visual_lexicon.py` | The references are only **8.75%** visual; strip the metadata part and the remainder is **7.0%** visual (it is warranty, dimensions, shipping). There is no visual signal in the reference to recover. **Visual Attribute Density**, which needs no reference, correlates **+0.42** with the judge (p ≈ 9e−13). |
| N4 | `echo_suppressed_decoding.py` | Training-free logit penalty on prompt tokens at decode time. Reduces echo and raises visual density and image-sensitivity, at the cost of ROUGE-L — which is the expected trade, since ROUGE-L is paying for the echo. |
| N5 | `compute_profile.py` | Measured latency, VRAM, energy and cost per 1,000 descriptions. |
| N6 | `epoch_dynamics.py` | Runs the same decomposition over the five saved BLIP epoch checkpoints, to test whether Milestone 2's "overfitting from data scarcity" is really the model learning to echo. Written, not yet run. |

`assemble_report.py` collects all of the above into
`models/results/novelty_summary.md` plus figures in `models/results/novelty_figures/`.

## Running them

CPU only, seconds each:

```powershell
python -m models.novelty.prompt_prefix_audit
python -m models.novelty.echo_audit
python -m models.novelty.metric_validity
python -m models.novelty.visual_lexicon
python -m models.novelty.duplicate_audit     # ~3 min, hashes every image
```

GPU:

```powershell
python -m models.novelty.visual_sensitivity --max-samples 135 --models blip
python -m models.novelty.echo_suppressed_decoding --max-samples 90 --lambdas 0 2 4 8
python -m models.novelty.compute_profile --n 12
python -m models.novelty.epoch_dynamics --max-samples 70
```

Then:

```powershell
python -m models.novelty.assemble_report
```

## The one-line fix N9 implies

In `models/blip/evaluate.py`, `processor.decode(output_ids[0], ...)` returns the
prompt *and* the continuation, because for `BlipForConditionalGeneration` the text
prompt is the decoder prefix. To score only what the model produced:

```python
input_len = inputs["input_ids"].shape[1]
text = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
```

Re-running evaluation after that change is what makes the BLIP vs CLIP-GPT2
comparison honest. Keep the current `blip_results.jsonl` as the "as-published"
baseline so the correction is documentable.
