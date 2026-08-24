#!/usr/bin/env python3
"""Make the accuracy-first direct-vs-decompose 2x2 figure.

Candidate answers are reconstructed from the tracked prompt-fix-A traces and
fixed-chain synthesis artifacts at the thesis headline step-gate threshold
(``tau=0.002305``), then passed through the final verifier.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from accuracy_first_replay import (  # noqa: E402
    DEFAULT_SYNTH_PATHS,
    load_all_accuracy_first_rows,
)

RESULTS = ROOT / "results"
DEFAULT_FIG = RESULTS / "reproduced/fig_direct_decompose_2x2_counts.png"
DEFAULT_SUMMARY = RESULTS / "reproduced/direct_decompose_2x2_accuracy_first.md"
DATASETS = [
    ("HotpotQA", "hotpotqa"),
    ("2Wiki", "2wiki"),
    ("MuSiQue", "musique"),
    ("PopQA", "popqa"),
]


def summarize(rows: list[dict]) -> dict[str, int | float]:
    n = len(rows)
    both_correct = direct_only = decomp_only = both_wrong = 0
    for row in rows:
        direct_ok = bool(row["direct_em"])
        decomp_ok = bool(row["decomp_em"])
        if direct_ok and decomp_ok:
            both_correct += 1
        elif direct_ok:
            direct_only += 1
        elif decomp_ok:
            decomp_only += 1
        else:
            both_wrong += 1
    return {
        "n": n,
        "both_correct": both_correct,
        "direct_only": direct_only,
        "decomp_only": decomp_only,
        "both_wrong": both_wrong,
        "direct_em": 100.0 * (both_correct + direct_only) / n,
        "decomp_em": 100.0 * (both_correct + decomp_only) / n,
        "oracle_em": 100.0 * (both_correct + direct_only + decomp_only) / n,
        "net_decomp_minus_direct": decomp_only - direct_only,
    }


def verify(stats: dict[str, dict[str, int | float]]) -> None:
    """Check that the four computed quadrants form a complete partition."""
    for dataset, row in stats.items():
        total = sum(
            int(row[key])
            for key in ("both_correct", "direct_only", "decomp_only", "both_wrong")
        )
        if total != int(row["n"]):
            raise AssertionError(f"{dataset}: quadrant sum {total} != n={row['n']}")


def write_summary(stats: dict[str, dict[str, int | float]], path: Path) -> None:
    lines = [
        "# Direct-vs-Decompose 2x2 Counts (Accuracy-First Replay)",
        "",
        "Setting: tracked prompt-fix-A traces replayed with the Tail-Leakage step gate at tau=0.002305, path-aware retrieval, original-question evidence union, and the final verifier.",
        "",
        "| Dataset | N | Both correct | Direct-only | Decomp-only | Both wrong | Direct EM | Decomp EM | Oracle EM | Decomp-only - Direct-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in stats.items():
        lines.append(
            f"| {dataset} | {row['n']} | {row['both_correct']} | {row['direct_only']} | "
            f"{row['decomp_only']} | {row['both_wrong']} | {row['direct_em']:.2f} | "
            f"{row['decomp_em']:.2f} | {row['oracle_em']:.2f} | "
            f"{row['net_decomp_minus_direct']:+.0f} |"
        )
    lines.extend([
        "",
        "The two paths remain complementary. Decompose-only wins dominate on all four datasets, "
        f"while HotpotQA retains {int(stats['HotpotQA']['direct_only'])} direct-only wins. "
        "Both-wrong cases remain outside the reach of routing alone.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_fig(stats: dict[str, dict[str, int | float]], path: Path) -> None:
    datasets = list(stats)
    direct_only = np.asarray([stats[d]["direct_only"] for d in datasets])
    decomp_only = np.asarray([stats[d]["decomp_only"] for d in datasets])
    both_correct = np.asarray([stats[d]["both_correct"] for d in datasets])
    both_wrong = np.asarray([stats[d]["both_wrong"] for d in datasets])
    x = np.arange(len(datasets))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=170)
    bars_direct = ax.bar(
        x - width / 2, direct_only, width,
        label="Direct-only correct", color="#244a8f",
    )
    bars_decomp = ax.bar(
        x + width / 2, decomp_only, width,
        label="Decompose-only correct", color="#0b8f6a",
    )
    for bars in (bars_direct, bars_decomp):
        ax.bar_label(bars, padding=3, fontsize=9)
    for i, dataset in enumerate(datasets):
        net = stats[dataset]["net_decomp_minus_direct"]
        y = max(direct_only[i], decomp_only[i]) + 28
        ax.text(
            x[i], y, f"net {net:+.0f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333",
        )
    ax.set_title("Answer-level Direct vs. Decompose Disagreement", fontsize=15, pad=12)
    ax.set_ylabel("Number of questions (N=1000)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, max(decomp_only.max(), direct_only.max()) + 85)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2)
    note = "Both-correct / both-wrong: " + "; ".join(
        f"{dataset} {int(both_correct[i])}/{int(both_wrong[i])}"
        for i, dataset in enumerate(datasets)
    )
    fig.text(0.08, 0.02, note, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synth", nargs="+", type=Path, default=list(DEFAULT_SYNTH_PATHS),
        help="tracked fixed-chain synthesis JSONL shards",
    )
    parser.add_argument(
        "--summary-out", type=Path, default=DEFAULT_SUMMARY,
        help="output Markdown summary (default: results/reproduced)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_FIG,
        help="output PNG (default: results/reproduced)",
    )
    args = parser.parse_args()

    loaded = load_all_accuracy_first_rows(args.synth)
    stats = {
        label: summarize(loaded[dataset])
        for label, dataset in DATASETS
    }
    verify(stats)
    write_summary(stats, args.summary_out)
    make_fig(stats, args.out)
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
