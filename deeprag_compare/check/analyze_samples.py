"""
Analyze multi_sample_dump.jsonl:
  1. Boxplot: per-question SC distribution, grouped by stratum.
  2. Per-question stats table (markdown).
  3. Confident-hallucination case study: for each HI_wrong question, pick the
     sample with highest SC and dump its low-prob token positions.
  4. Token-level confidence breakdown: entity-token vs filler-token entropy.

Output:
  check/figures/sc_boxplot.png
  check/figures/sc_vs_em_scatter.png
  check/analysis_table.md
  check/confident_hallucination_cases.md
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/S113065528/Research/deeprag_compare")
DUMP = ROOT / "check/multi_sample_dump.jsonl"
SUMMARY = ROOT / "check/multi_sample_summary.json"
FIG_DIR = ROOT / "check/figures"
TABLE_MD = ROOT / "check/analysis_table.md"
CASES_MD = ROOT / "check/confident_hallucination_cases.md"

FIG_DIR.mkdir(exist_ok=True)

STRATUM_ORDER = ["HI_correct", "HI_wrong", "LO_correct", "LO_wrong"]
STRATUM_COLORS = {
    "HI_correct": "#2ca02c",  # green
    "HI_wrong": "#d62728",    # red — confident hallucination
    "LO_correct": "#1f77b4",  # blue
    "LO_wrong": "#7f7f7f",    # gray
}


def load():
    recs = [json.loads(l) for l in open(DUMP)]
    by_qid = defaultdict(list)
    for r in recs:
        by_qid[r["qid"]].append(r)
    summary = json.load(open(SUMMARY))
    return recs, by_qid, summary


def boxplot_sc(by_qid, summary):
    """One box per question, sorted by stratum then by greedy_nr_sc."""
    summary_sorted = sorted(
        summary, key=lambda s: (STRATUM_ORDER.index(s["stratum"]), s["greedy_nr_sc"])
    )
    sc_values = [[r["sc"] for r in by_qid[s["qid"]]] for s in summary_sorted]
    labels = [f"Q{i+1}" for i in range(len(summary_sorted))]
    colors = [STRATUM_COLORS[s["stratum"]] for s in summary_sorted]

    fig, ax = plt.subplots(figsize=(14, 6))
    bp = ax.boxplot(sc_values, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Overlay greedy nr_sc as red star
    for i, s in enumerate(summary_sorted):
        ax.plot(i + 1, s["greedy_nr_sc"], marker="*", color="black",
                markersize=14, markeredgecolor="white", zorder=5)

    # Stratum group dividers
    for i in range(1, 4):
        ax.axvline(5 * i + 0.5, color="black", linestyle="--", alpha=0.3)

    ax.set_ylabel("Self-Certainty (SC)")
    ax.set_xlabel("Question (sorted: 5 per stratum)")
    ax.set_title("SC distribution under sampling (N=20, T=0.7)\n"
                 "Boxes colored by stratum; black star = greedy SC")

    # Custom legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=STRATUM_COLORS[s], alpha=0.6, label=s)
                      for s in STRATUM_ORDER]
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="black",
                                     markersize=12, linestyle="None",
                                     label="greedy SC"))
    ax.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    out = FIG_DIR / "sc_boxplot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
    return summary_sorted


def scatter_sc_vs_em(summary):
    fig, ax = plt.subplots(figsize=(8, 6))
    for stratum in STRATUM_ORDER:
        xs, ys = [], []
        for s in summary:
            if s["stratum"] == stratum:
                xs.append(s["sc_mean"])
                ys.append(s["em_rate"])
        ax.scatter(xs, ys, c=STRATUM_COLORS[stratum], s=80, alpha=0.7,
                   edgecolor="black", label=stratum)
    ax.set_xlabel("Mean SC across N=20 samples")
    ax.set_ylabel("EM rate across N=20 samples")
    ax.set_title("SC vs EM consistency under sampling")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIG_DIR / "sc_vs_em_scatter.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def write_table(summary):
    lines = [
        "# Per-question stats (N=20 samples, T=0.7)\n",
        "| stratum | qid | greedy_sc | greedy_em | sc_mean | sc_std | sc_min | sc_max | em_rate | n_uniq | question |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    summary_sorted = sorted(summary, key=lambda s: (STRATUM_ORDER.index(s["stratum"]), s["greedy_nr_sc"]))
    for s in summary_sorted:
        q = s["question"]
        if len(q) > 60:
            q = q[:57] + "..."
        lines.append(
            f"| {s['stratum']} | {s['qid'][:10]} | {s['greedy_nr_sc']:.2f} | {s['greedy_nr_em']} | "
            f"{s['sc_mean']:.2f} | {s['sc_std']:.2f} | {s['sc_min']:.2f} | {s['sc_max']:.2f} | "
            f"{s['em_rate']:.2f} | {s['n_unique_preds']} | {q} |"
        )
    TABLE_MD.write_text("\n".join(lines))
    print(f"Saved {TABLE_MD}")


def confident_hallucination_cases(by_qid, summary):
    """For each HI_wrong question, pick the sample with the HIGHEST SC
    (most confident) and dump its low-prob token positions."""
    lines = [
        "# Confident Hallucination Case Study\n",
        "For each HI_wrong question, we show the sample with **highest SC** "
        "(most confident) and break down per-token confidence.\n",
        "Columns:",
        "- `pos`: token position",
        "- `tok`: chosen token (greedy of sampling step)",
        "- `lp`: log-prob of chosen token",
        "- `top1−top2`: gap between top-1 and top-2 (smaller = more uncertain among alternatives)",
        "- `entropy`: entropy of full distribution at this position (nats)",
        "- `top5`: top-5 alternatives (truncated)\n",
    ]
    hi_wrong = [s for s in summary if s["stratum"] == "HI_wrong"]
    for q_idx, s in enumerate(hi_wrong, 1):
        recs = by_qid[s["qid"]]
        best = max(recs, key=lambda r: r["sc"])
        lines.append(f"## Q{q_idx}. qid={s['qid']}")
        lines.append(f"**Question:** {s['question']}")
        lines.append(f"**Gold:** {s['golds']}")
        lines.append(f"**Greedy pred (from baseline):** {s['greedy_nr_pred']}")
        lines.append(f"**Sample preds (N=20):** {s['unique_preds']}")
        lines.append(f"**SC range:** [{s['sc_min']:.2f}, {s['sc_max']:.2f}]  "
                     f"mean={s['sc_mean']:.2f}  std={s['sc_std']:.2f}")
        lines.append(f"**EM rate:** {s['em_rate']:.2f} ({sum(r['em'] for r in recs)}/{len(recs)})")
        lines.append("")
        lines.append(f"### Highest-SC sample (sc={best['sc']:.2f}, "
                     f"em={best['em']}, pred=`{best['pred']}`)")
        lines.append(f"Generated text: `{best['text'][:300]}{'...' if len(best['text']) > 300 else ''}`\n")
        lines.append("Token-level confidence (showing top-15 highest-entropy tokens):")
        toks = best["tokens"]
        # Sort by entropy descending, take top 15
        toks_by_ent = sorted(toks, key=lambda t: -t["entropy"])[:15]
        toks_by_ent = sorted(toks_by_ent, key=lambda t: t["pos"])  # display in order
        lines.append("| pos | tok | lp | top1−top2 | entropy | top5 |")
        lines.append("|---|---|---|---|---|---|")
        for t in toks_by_ent:
            gap = t["top1_lp"] - t["top2_lp"]
            top5 = ", ".join(f"`{x['tok']}`({x['lp']:.2f})" for x in t["topk"][:5])
            tok_disp = repr(t["tok"])
            lines.append(f"| {t['pos']} | {tok_disp} | {t['lp']:.2f} | {gap:.2f} | "
                         f"{t['entropy']:.2f} | {top5} |")
        lines.append("")
    CASES_MD.write_text("\n".join(lines))
    print(f"Saved {CASES_MD}")


def aggregate_stratum_stats(summary):
    print("\n=== Per-stratum aggregate ===")
    by_stratum = defaultdict(list)
    for s in summary:
        by_stratum[s["stratum"]].append(s)
    print(f"{'stratum':12s} {'mean(sc_mean)':>14s} {'mean(sc_std)':>14s} "
          f"{'mean(em_rate)':>14s} {'mean(n_uniq)':>14s}")
    for stratum in STRATUM_ORDER:
        items = by_stratum[stratum]
        if not items:
            continue
        ms = statistics.mean(s["sc_mean"] for s in items)
        mst = statistics.mean(s["sc_std"] for s in items)
        mem = statistics.mean(s["em_rate"] for s in items)
        mun = statistics.mean(s["n_unique_preds"] for s in items)
        print(f"{stratum:12s} {ms:14.2f} {mst:14.2f} {mem:14.2f} {mun:14.1f}")


def main():
    recs, by_qid, summary = load()
    print(f"Loaded {len(recs)} records, {len(summary)} questions")
    boxplot_sc(by_qid, summary)
    scatter_sc_vs_em(summary)
    write_table(summary)
    confident_hallucination_cases(by_qid, summary)
    aggregate_stratum_stats(summary)


if __name__ == "__main__":
    main()
