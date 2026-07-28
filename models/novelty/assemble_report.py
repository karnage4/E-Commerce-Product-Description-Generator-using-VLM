"""
Assembles every novelty experiment into one master table plus figures.

Reads only the JSON files written by the N1-N6 scripts; runs nothing itself, so
it is safe to re-run at any time and requires no GPU.

Run:
    python -m models.novelty.assemble_report
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.shared.config import RESULTS_DIR

FIG_DIR = RESULTS_DIR / "novelty_figures"
OUT_MD = RESULTS_DIR / "novelty_summary.md"


def load(name):
    p = RESULTS_DIR / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def fig_rank_flip(echo, out):
    """Standard ROUGE-L vs Visual-Residual Recall — the ranking flip."""
    if not echo:
        return
    sysd = echo["systems"]
    names = [n for n in ("Metadata-Only (template)", "BLIP fine-tuned",
                         "CLIP-GPT2 fine-tuned", "Two-Stage BLIP (stage2)",
                         "Two-Stage CLIP-GPT2 (stage2)") if n in sysd]
    short = {"Metadata-Only (template)": "Metadata\ntemplate",
             "BLIP fine-tuned": "BLIP",
             "CLIP-GPT2 fine-tuned": "CLIP-GPT2",
             "Two-Stage BLIP (stage2)": "Two-Stage\nBLIP",
             "Two-Stage CLIP-GPT2 (stage2)": "Two-Stage\nCLIP-GPT2"}
    rl = [sysd[n]["ROUGE-L"] for n in names]
    vrr = [sysd[n]["Visual Residual Recall"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(range(len(names)), rl, color="#8899bb")
    axes[0].set_title("Reported metric: ROUGE-L\n(rewards metadata echo)")
    axes[0].set_ylabel("ROUGE-L")
    axes[1].bar(range(len(names)), vrr, color="#bb7755")
    axes[1].set_title("Corrected metric: Visual-Residual Recall\n(metadata credit removed)")
    axes[1].set_ylabel("VRR (%)")
    for ax in axes:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([short[n] for n in names], fontsize=8)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "rank_flip.png", dpi=150)
    plt.close(fig)


def fig_contamination(dup, out):
    if not dup:
        return
    sc = dup["scores"]
    names = list(sc.keys())
    parts = ["CONTAMINATED", "CLEAN"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.35
    for j, part in enumerate(parts):
        vals = [sc[n][part]["CIDEr"] if sc[n].get(part) else 0 for n in names]
        ax.bar([i + j * w for i in range(len(names))], vals, w, label=part.title())
    ax.set_xticks([i + w / 2 for i in range(len(names))])
    ax.set_xticklabels(["pHash\nretrieval", "BLIP", "CLIP-GPT2"][:len(names)], fontsize=9)
    ax.set_ylabel("CIDEr")
    ax.set_yscale("symlog")
    ax.set_title("Retrieval collapses once catalogue duplicates are removed;\n"
                 "generation does not")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "contamination.png", dpi=150)
    plt.close(fig)


def fig_vss(vss, out):
    if not vss:
        return
    keys = list(vss.keys())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.35
    ax.bar([i for i in range(len(keys))], [vss[k]["VSS_blank"] for k in keys], w,
           label="VSS (blank image)")
    ax.bar([i + w for i in range(len(keys))], [vss[k]["DeltaROUGE_real_minus_blank"] for k in keys],
           w, label="ROUGE-L lost by deleting the image")
    ax.set_xticks([i + w / 2 for i in range(len(keys))])
    ax.set_xticklabels(keys, fontsize=8)
    ax.set_title("Image dependence vs. what the metric charges for it")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "visual_sensitivity.png", dpi=150)
    plt.close(fig)


def fig_esd(esd, out):
    if not esd:
        return
    lams = sorted(esd.keys(), key=float)
    x = [float(l) for l in lams]
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(x, [esd[l]["Echo Rate"] for l in lams], "o-", color="#bb5555", label="Echo Rate")
    ax1.plot(x, [esd[l]["ROUGE-L"] for l in lams], "s-", color="#8899bb", label="ROUGE-L")
    ax1.set_xlabel("echo-suppression strength  $\\lambda$")
    ax1.set_ylabel("%")
    ax2 = ax1.twinx()
    ax2.plot(x, [esd[l]["Visual Residual Recall"] for l in lams], "^-",
             color="#337755", label="Visual-Residual Recall")
    ax2.set_ylabel("VRR (%)", color="#337755")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax1.grid(alpha=.3)
    ax1.set_title("Echo-Suppressed Decoding: trading echo for visual content")
    fig.tight_layout()
    fig.savefig(out / "esd.png", dpi=150)
    plt.close(fig)


def fig_epochs(ep, out):
    if not ep:
        return
    keys = sorted(ep.keys(), key=lambda k: int(k.split("_")[1]))
    x = [int(k.split("_")[1]) for k in keys]
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(x, [ep[k]["Echo Rate"] for k in keys], "o-", color="#bb5555", label="Echo Rate")
    ax1.set_xlabel("training epoch"); ax1.set_ylabel("Echo Rate (%)", color="#bb5555")
    ax2 = ax1.twinx()
    ax2.plot(x, [ep[k]["Visual Residual Recall"] for k in keys], "^-",
             color="#337755", label="Visual-Residual Recall")
    ax2.set_ylabel("VRR (%)", color="#337755")
    ax1.set_title("What extra training buys: more echo or more vision?")
    ax1.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "epoch_dynamics.png", dpi=150)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    echo = load("novelty_echo_audit.json")
    dup = load("novelty_duplicate_audit.json")
    vss = load("novelty_visual_sensitivity.json")
    esd = load("novelty_echo_suppressed_decoding.json")
    ep = load("novelty_epoch_dynamics.json")
    cp = load("novelty_compute_profile.json")

    prefix = load("novelty_prompt_prefix_audit.json")
    validity = load("novelty_metric_validity.json")
    lexicon = load("novelty_visual_lexicon.json")

    lines = ["# Novelty experiments — assembled results\n"]

    if prefix:
        lines.append("## N9 Prompt-prefix leakage (headline correction)\n")
        for name in ("BLIP fine-tuned", "CLIP-GPT2 fine-tuned"):
            e = prefix.get(name)
            if not e:
                continue
            lines.append(f"**{name}** — {e['items_starting_with_prompt_prefix_pct']}% of outputs "
                         f"open with a verbatim prompt prefix; "
                         f"{e['mean_pct_of_output_that_is_prompt_prefix']}% of the scored text is the prompt.\n")
            lines.append("| Metric | as scored | prompt removed |")
            lines.append("|---|---|---|")
            for m in ("BLEU-1", "BLEU-4", "ROUGE-L", "METEOR", "CIDEr"):
                lines.append(f"| {m} | {e['as_scored_in_milestone2'][m]} | {e['prompt_prefix_removed'][m]} |")
            lines.append("")
        if prefix.get("ranking_changes"):
            flipped = [m for m, v in prefix["ranking_changes"].items() if v["flipped"]]
            lines.append(f"Rankings that flip once compared like-for-like: **{', '.join(flipped)}**\n")

    if validity or lexicon:
        lines.append("## N7/N8 Metric validity against the LLM judge\n")
        lines.append("| Metric | Spearman rho vs judged visual grounding | Reference-free? |")
        lines.append("|---|---|---|")
        if validity:
            for m, c in validity.get("correlations", {}).items():
                lines.append(f"| {m} | {c['judge_visual_grounding']['spearman_rho']} | no |")
        if lexicon:
            lines.append(f"| **Visual Attribute Density** | "
                         f"**{lexicon['vad_vs_judge']['visual_grounding']['spearman_rho']}** | yes |")
        lines.append("")
        if lexicon:
            lines.append("| Text | Visual Attribute Density |")
            lines.append("|---|---|")
            for n, s in lexicon["systems"].items():
                lines.append(f"| {n} | {s['VAD']} |")
            lines.append("")

    if echo:
        d = echo["reference_decomposition"]
        lines.append("## N1 Metadata-echo audit\n")
        lines.append(f"- {d['mean_pct_of_reference_derivable_from_metadata']}% of the average "
                     f"reference's content tokens also occur in the metadata prompt "
                     f"(median {d['median_pct']}%).\n")
        lines.append("| System | ROUGE-L | Residual ROUGE-L | Metadata Recall | Visual-Residual Recall | Echo Rate |")
        lines.append("|---|---|---|---|---|---|")
        for n, s in echo["systems"].items():
            lines.append(f"| {n} | {s['ROUGE-L']} | {s['Residual ROUGE-L']} | "
                         f"{s['Metadata Recall']} | {s['Visual Residual Recall']} | {s['Echo Rate']} |")
        lines.append("")

    if dup:
        lines.append("## N2 Catalogue near-duplicate contamination\n")
        lines.append(f"- {dup['n_contaminated']}/{dup['n_test']} test items have a train image "
                     f"within pHash Hamming {dup['hamming_threshold']}.")
        lines.append(f"- Hamming distribution: {dup['hamming_distribution']}\n")
        lines.append("| System | Partition | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr |")
        lines.append("|---|---|---|---|---|---|---|")
        for n, parts in dup["scores"].items():
            for p, s in parts.items():
                if s:
                    lines.append(f"| {n} | {p} | {s['BLEU-1']} | {s['BLEU-4']} | "
                                 f"{s['ROUGE-L']} | {s['METEOR']} | {s['CIDEr']} |")
        lines.append("")

    if vss:
        lines.append("## N3 Visual counterfactual sensitivity\n")
        lines.append("| Model / prompt | ROUGE-L real | ROUGE-L blank | ROUGE-L mismatch | "
                     "VSS blank | VSS mismatch | ROUGE-L lost by deleting image |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, c in vss.items():
            lines.append(f"| {k} | {c['ROUGE-L_real']} | {c['ROUGE-L_blank']} | "
                         f"{c['ROUGE-L_mismatch']} | {c['VSS_blank']} | {c['VSS_mismatch']} | "
                         f"{c['DeltaROUGE_real_minus_blank']} |")
        lines.append("")

    if esd:
        lines.append("## N4 Echo-suppressed decoding\n")
        lines.append("| lambda | ROUGE-L | Residual ROUGE-L | Visual-Residual Recall | Echo Rate | VSS | CIDEr | mean words |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for l in sorted(esd.keys(), key=float):
            c = esd[l]
            lines.append(f"| {l} | {c['ROUGE-L']} | {c['Residual ROUGE-L']} | "
                         f"{c['Visual Residual Recall']} | {c['Echo Rate']} | {c.get('VSS')} | "
                         f"{c['CIDEr']} | {c['mean_gen_words']} |")
        lines.append("")

    if ep:
        lines.append("## N6 Echo dynamics across epochs\n")
        lines.append("| Checkpoint | ROUGE-L | Residual ROUGE-L | Visual-Residual Recall | Metadata Recall | Echo Rate | VSS |")
        lines.append("|---|---|---|---|---|---|---|")
        for k in sorted(ep.keys(), key=lambda k: int(k.split('_')[1])):
            c = ep[k]
            lines.append(f"| {k} | {c['ROUGE-L']} | {c['Residual ROUGE-L']} | "
                         f"{c['Visual Residual Recall']} | {c['Metadata Recall']} | "
                         f"{c['Echo Rate']} | {c.get('VSS')} |")
        lines.append("")

    if cp:
        lines.append("## N5 Compute profile\n")
        lines.append("| Config | params (M) | s/description | descriptions/hour | kWh per 1k | USD electricity per 1k | peak VRAM (GB) |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, v in cp.items():
            if k == "config" or not isinstance(v, dict):
                continue
            lines.append(f"| {k} | {v.get('params_total_M')} | {v.get('sec_per_description')} | "
                         f"{v.get('descriptions_per_hour')} | {v.get('kWh_per_1k')} | "
                         f"{v.get('electricity_USD_per_1k')} | {v.get('peak_vram_GB')} |")
        lines.append("")

    fig_rank_flip(echo, FIG_DIR)
    fig_contamination(dup, FIG_DIR)
    fig_vss(vss, FIG_DIR)
    fig_esd(esd, FIG_DIR)
    fig_epochs(ep, FIG_DIR)

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved → {OUT_MD}")
    print(f"Figures → {FIG_DIR}")


if __name__ == "__main__":
    main()
