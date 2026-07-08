#!/usr/bin/env python3
"""Make answer-level direct-vs-decompose 2x2 disagreement figure.

This figure is intentionally answer-level, not retrieval-level. It shows whether
saved direct/decomposed answers are correct, using the current final candidate
traces with path-aware retrieval, original-question evidence union, and
rule-based verifier replay.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
THESIS_FIG = ROOT.parent / "Thesis" / "figures" / "fig_direct_decompose_2x2_counts.png"
SUMMARY = RESULTS / "direct_decompose_2x2_current_final.md"

FILES = {
    "2Wiki": "2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "HotpotQA": "hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "MuSiQue": "musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "PopQA": "popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(rows: list[dict]) -> dict[str, int | float]:
    n = len(rows)
    both_correct = direct_only = decomp_only = both_wrong = 0
    for row in rows:
        d = int(row.get("direct_em", 0))
        c = int(row.get("decomp_em", 0))
        if d and c:
            both_correct += 1
        elif d:
            direct_only += 1
        elif c:
            decomp_only += 1
        else:
            both_wrong += 1
    direct_em = 100 * (both_correct + direct_only) / n
    decomp_em = 100 * (both_correct + decomp_only) / n
    oracle_em = 100 * (both_correct + direct_only + decomp_only) / n
    return {
        "n": n,
        "both_correct": both_correct,
        "direct_only": direct_only,
        "decomp_only": decomp_only,
        "both_wrong": both_wrong,
        "direct_em": direct_em,
        "decomp_em": decomp_em,
        "oracle_em": oracle_em,
        "net_decomp_minus_direct": decomp_only - direct_only,
    }


def write_summary(stats: dict[str, dict[str, int | float]]) -> None:
    lines = [
        "# Direct-vs-Decompose 2x2 Counts (Current Final Traces)",
        "",
        "Setting: current final candidate traces with path-aware retrieval, original-question evidence union, and type-aware rule-based verifier replay.",
        "",
        "This is an answer-level analysis. It is different from retrieval coverage: a passage can contain the gold answer string, while the final decomposed answer can still be wrong because of decomposition drift, reasoning error, or synthesis error.",
        "",
        "| Dataset | N | Both correct | Direct-only | Decomp-only | Both wrong | Direct EM | Decomp EM | Oracle EM | Decomp-only - Direct-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds, row in stats.items():
        lines.append(
            f"| {ds} | {row['n']} | {row['both_correct']} | {row['direct_only']} | "
            f"{row['decomp_only']} | {row['both_wrong']} | {row['direct_em']:.2f} | "
            f"{row['decomp_em']:.2f} | {row['oracle_em']:.2f} | {row['net_decomp_minus_direct']:+.0f} |"
        )
    lines.extend([
        "",
        "Interpretation:",
        "",
        "- Decomposition improves retrieval coverage, but answer-level correctness still depends on reasoning and final synthesis.",
        "- In the current final traces, HotpotQA is no longer strongly direct-favored: direct-only = 111 and decomp-only = 113.",
        "- The large both-wrong counts show that many errors cannot be fixed by routing alone; they need better retrieval, decomposition, or synthesis.",
    ])
    SUMMARY.write_text("\n".join(lines) + "\n")


def make_fig(stats: dict[str, dict[str, int | float]]) -> None:
    datasets = list(stats)
    direct_only = np.array([stats[d]["direct_only"] for d in datasets])
    decomp_only = np.array([stats[d]["decomp_only"] for d in datasets])
    both_correct = np.array([stats[d]["both_correct"] for d in datasets])
    both_wrong = np.array([stats[d]["both_wrong"] for d in datasets])

    x = np.arange(len(datasets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=170)

    bars_d = ax.bar(x - width / 2, direct_only, width, label="Direct-only correct", color="#244a8f")
    bars_c = ax.bar(x + width / 2, decomp_only, width, label="Decompose-only correct", color="#0b8f6a")

    for bars in (bars_d, bars_c):
        ax.bar_label(bars, padding=3, fontsize=9)

    for i, d in enumerate(datasets):
        net = stats[d]["net_decomp_minus_direct"]
        y = max(direct_only[i], decomp_only[i]) + 34
        ax.text(
            x[i], y, f"net {net:+.0f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333"
        )

    ax.set_title("Answer-level Direct vs. Decompose Disagreement", fontsize=15, pad=12)
    ax.set_ylabel("Number of questions (N=1000)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, max(decomp_only.max(), direct_only.max()) + 90)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2)

    note = "Both-correct / both-wrong counts: " + "; ".join(
        f"{d}: {int(both_correct[i])}/{int(both_wrong[i])}" for i, d in enumerate(datasets)
    )
    fig.text(0.08, 0.02, note, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    THESIS_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_FIG, bbox_inches="tight")


def main() -> None:
    stats = {ds: summarize(load_rows(RESULTS / fn)) for ds, fn in FILES.items()}
    write_summary(stats)
    make_fig(stats)
    print(f"Wrote {SUMMARY}")
    print(f"Wrote {THESIS_FIG}")


if __name__ == "__main__":
    main()
