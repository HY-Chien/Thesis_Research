"""
Exp-A: Failure-mode classification on the 1000Q × 4-dataset post-hoc data.

For each question we have (nr_em, rag_em, first_ent_min_lp). Split by confidence
median → cross-tab with (nr correct? × rag rescues?) → 6 failure-mode cells:

  T1: HIGH conf + NR correct           (gate must NOT retrieve)
  T2: HIGH conf + NR wrong + RAG saves (gate would MISS — false negative)
  T3: HIGH conf + NR wrong + RAG fails (Type A: confident hallucination)
  T4: LOW conf  + NR correct           (gate would over-retrieve — false positive)
  T5: LOW conf  + NR wrong + RAG saves (gate works as intended)
  T6: LOW conf  + NR wrong + RAG fails (low conf, retrieval still useless)

A good gate signal puts wrong answers in T5 (recoverable) rather than T2 (hidden by
high confidence) or T3 (true hallucination, RAG can't fix anyway).

Output:
    check/exp_a_output/exp_a_summary.md
    check/exp_a_output/figures/failure_modes_*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = Path("/home/S113065528/Research/deeprag_compare/check/exp1_output")
OUTPUT_DIR = Path("/home/S113065528/Research/deeprag_compare/check/exp_a_output")
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]

CELLS = ["T1", "T2", "T3", "T4", "T5", "T6"]
CELL_DESC = {
    "T1": "HIGH+NR✓ (correctly skipped)",
    "T2": "HIGH+NR✗+RAG✓ (gate MISS)",
    "T3": "HIGH+NR✗+RAG✗ (Type A)",
    "T4": "LOW+NR✓ (over-retrieve)",
    "T5": "LOW+NR✗+RAG✓ (gate WIN)",
    "T6": "LOW+NR✗+RAG✗ (low conf no fix)",
}


def classify_one(c, nr, rag, threshold):
    high = c >= threshold
    if high and nr == 1:
        return "T1"
    if high and nr == 0 and rag == 1:
        return "T2"
    if high and nr == 0 and rag == 0:
        return "T3"
    if (not high) and nr == 1:
        return "T4"
    if (not high) and nr == 0 and rag == 1:
        return "T5"
    return "T6"


def classify(records, conf_key, threshold):
    bins = {c: 0 for c in CELLS}
    for r in records:
        c = r.get(conf_key)
        if c is None:
            continue
        bins[classify_one(c, r["nr_em"], r["rag_em"], threshold)] += 1
    return bins


def percent(bins):
    n = sum(bins.values())
    return {k: (v / n if n else 0.0) for k, v in bins.items()}


def metrics_from_bins(bins):
    """Derived metrics that judge how 'good' the confidence signal is."""
    n = sum(bins.values())
    n_wrong = bins["T2"] + bins["T3"] + bins["T5"] + bins["T6"]
    n_high_wrong = bins["T2"] + bins["T3"]
    n_low_wrong = bins["T5"] + bins["T6"]

    # Type A prevalence: out of all wrong, how many are confident-and-unsalvageable?
    type_a_share = bins["T3"] / n_wrong if n_wrong else 0.0
    # Among high-confidence wrong, what fraction does RAG save?
    rag_save_high = bins["T2"] / n_high_wrong if n_high_wrong else 0.0
    # Among low-confidence wrong, what fraction does RAG save?
    rag_save_low = bins["T5"] / n_low_wrong if n_low_wrong else 0.0
    # Gate signal quality: P(wrong | low conf) − P(wrong | high conf)
    n_high = bins["T1"] + bins["T2"] + bins["T3"]
    n_low = bins["T4"] + bins["T5"] + bins["T6"]
    p_wrong_high = (bins["T2"] + bins["T3"]) / n_high if n_high else 0.0
    p_wrong_low = (bins["T5"] + bins["T6"]) / n_low if n_low else 0.0
    return {
        "n": n,
        "n_wrong": n_wrong,
        "type_a_share": type_a_share,
        "rag_save_high": rag_save_high,
        "rag_save_low": rag_save_low,
        "p_wrong_high": p_wrong_high,
        "p_wrong_low": p_wrong_low,
        "discrimination": p_wrong_low - p_wrong_high,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "figures").mkdir(exist_ok=True)

    all_stats = {}

    for ds in DATASETS:
        f = INPUT_DIR / f"{ds}_per_q.jsonl"
        records = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        records = [r for r in records if r["first_ent_min_lp"] is not None]
        if not records:
            continue

        ent_vals = [r["first_ent_min_lp"] for r in records]
        sc_vals = [r["nr_sc"] for r in records]
        ent_thr = float(np.median(ent_vals))
        sc_thr = float(np.median(sc_vals))

        bins_ent = classify(records, "first_ent_min_lp", ent_thr)
        bins_sc = classify(records, "nr_sc", sc_thr)

        all_stats[ds] = {
            "n": len(records),
            "ent_thr": ent_thr,
            "sc_thr": sc_thr,
            "bins_ent": bins_ent,
            "bins_sc": bins_sc,
            "metrics_ent": metrics_from_bins(bins_ent),
            "metrics_sc": metrics_from_bins(bins_sc),
        }

        # Plot side-by-side stacked bars
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, bins, label, thr in [
            (axes[0], bins_sc, "sc_orig", sc_thr),
            (axes[1], bins_ent, "first_ent_min_lp", ent_thr),
        ]:
            cells = CELLS
            counts = [bins[c] for c in cells]
            colors = ["#9ad", "#f88", "#f33", "#cdb", "#3a3", "#996"]
            bars = ax.bar(cells, counts, color=colors)
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.01,
                            str(count), ha="center", fontsize=8)
            ax.set_title(f"{label} gate (thr={thr:.3f})")
            ax.set_ylabel("# questions")
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle(f"{ds}: failure-mode breakdown")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"figures/failure_modes_{ds}.png", dpi=150)
        plt.close(fig)

    # ---- Markdown summary ----
    lines = ["# Exp-A: Failure-Mode Classification (1000Q × 4 datasets)\n"]
    lines.append("## Six failure modes\n")
    lines.append("Confidence is split at the median; each cell counts how many questions fall into it.\n")
    for k, v in CELL_DESC.items():
        lines.append(f"- **{k}**: {v}")

    lines.append("\n## Cell counts (first_ent_min_lp gate)\n")
    lines.append("| Dataset | T1 | T2 | T3 | T4 | T5 | T6 | total |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds, s in all_stats.items():
        b = s["bins_ent"]
        lines.append(
            f"| {ds} | {b['T1']} | {b['T2']} | {b['T3']} | {b['T4']} | {b['T5']} | {b['T6']} | {s['n']} |"
        )

    lines.append("\n## Cell counts (sc_orig gate)\n")
    lines.append("| Dataset | T1 | T2 | T3 | T4 | T5 | T6 | total |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds, s in all_stats.items():
        b = s["bins_sc"]
        lines.append(
            f"| {ds} | {b['T1']} | {b['T2']} | {b['T3']} | {b['T4']} | {b['T5']} | {b['T6']} | {s['n']} |"
        )

    lines.append("\n## Derived signal quality\n")
    lines.append("- `Type-A share`: fraction of wrong answers that are HIGH-confidence and unsalvageable by RAG (T3 / wrong)")
    lines.append("- `RAG save (HIGH)`: among HIGH-conf wrong, fraction RAG fixes (T2 / (T2+T3))")
    lines.append("- `RAG save (LOW)`: among LOW-conf wrong, fraction RAG fixes (T5 / (T5+T6))")
    lines.append("- `Discrimination`: P(wrong | LOW) − P(wrong | HIGH); higher = better gate signal\n")

    lines.append("| Dataset | Metric | Type-A share | RAG save HIGH | RAG save LOW | Discrim. |")
    lines.append("|---|---|---|---|---|---|")
    for ds, s in all_stats.items():
        for label, key in [("sc_orig", "metrics_sc"), ("first_ent_min_lp", "metrics_ent")]:
            m = s[key]
            lines.append(
                f"| {ds} | {label} "
                f"| {m['type_a_share']:.3f} "
                f"| {m['rag_save_high']:.3f} "
                f"| {m['rag_save_low']:.3f} "
                f"| {m['discrimination']:+.3f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append("- A larger `Discrimination` means LOW-conf questions are more likely to be wrong than HIGH-conf — i.e. the signal correlates with correctness in the right direction.")
    lines.append("- A larger `RAG save (LOW) − RAG save (HIGH)` means RAG actually works better when triggered by LOW conf — a gate based on this signal does what we want.")
    lines.append("- Type-A share is the **fundamental ceiling**: these questions are unsalvageable by RAG at all, no matter what gate we use. They need a different intervention (better retriever, better reasoning model, or refusal).")

    summary = "\n".join(lines)
    (OUTPUT_DIR / "exp_a_summary.md").write_text(summary)
    print(summary)
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
