#!/usr/bin/env python3
"""Make the final all-tl_sum direct-vs-decompose routing-gap figure.

The prompt-fix-A candidate traces are loaded through ``build`` and the final
router is replayed through ``decide(..., 'both', 0.0005, 0.0158)``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_verifier_tlsum_fixA import build, decide  # noqa: E402

OUT = ROOT.parent / "Thesis" / "figures" / "fig_routing_gap_oracle.png"
SUMMARY_OUT = ROOT / "results" / "routing_gap_oracle.md"
DATASETS = [
    ("HotpotQA", "hotpotqa"),
    ("2Wiki", "2wiki"),
    ("MuSiQue", "musique"),
    ("PopQA", "popqa"),
]
DIRECT_TAU = 0.0005
POST_DECOMP_TAU = 0.0158
EXPECTED = {
    # direct correct, decomp correct, router correct, oracle correct, direct choices
    "HotpotQA": (303, 357, 375, 435, 167),
    "2Wiki": (180, 490, 487, 530, 26),
    "MuSiQue": (102, 267, 266, 293, 122),
    "PopQA": (378, 538, 537, 571, 10),
}


def summarize(rows: list[dict]) -> dict[str, float | int]:
    n = len(rows)
    choices = [
        decide(row, "both", DIRECT_TAU, POST_DECOMP_TAU)
        for row in rows
    ]
    direct = sum(int(row["direct_em"]) for row in rows)
    decomp = sum(int(row["decomp_em"]) for row in rows)
    router = sum(
        int(row["direct_em"] if choice == "direct" else row["decomp_em"])
        for row, choice in zip(rows, choices)
    )
    oracle = sum(int(row["direct_em"] or row["decomp_em"]) for row in rows)
    direct_only = sum(
        int(row["direct_em"] and not row["decomp_em"])
        for row in rows
    )
    decomp_only = sum(
        int(row["decomp_em"] and not row["direct_em"])
        for row in rows
    )
    both_wrong = sum(
        int(not row["direct_em"] and not row["decomp_em"])
        for row in rows
    )
    direct_em = 100.0 * direct / n
    decomp_em = 100.0 * decomp / n
    router_em = 100.0 * router / n
    oracle_em = 100.0 * oracle / n
    best_pure = max(direct_em, decomp_em)
    n_direct = sum(choice == "direct" for choice in choices)
    return {
        "n": n,
        "direct_correct": direct,
        "decomp_correct": decomp,
        "router_correct": router,
        "oracle_correct": oracle,
        "direct_em": direct_em,
        "decomp_em": decomp_em,
        "best_pure": best_pure,
        "router_em": router_em,
        "oracle_em": oracle_em,
        "oracle_vs_best_pure": oracle_em - best_pure,
        "oracle_vs_router": oracle_em - router_em,
        "direct_choices": n_direct,
        "decomp_choices": n - n_direct,
        "direct_only": direct_only,
        "decomp_only": decomp_only,
        "both_wrong": both_wrong,
    }


def verify(stats: dict[str, dict[str, float | int]]) -> None:
    for dataset, expected in EXPECTED.items():
        row = stats[dataset]
        observed = (
            int(row["direct_correct"]),
            int(row["decomp_correct"]),
            int(row["router_correct"]),
            int(row["oracle_correct"]),
            int(row["direct_choices"]),
        )
        if observed != expected:
            raise AssertionError(f"{dataset}: expected {expected}, observed {observed}")


def write_summary(stats: dict[str, dict[str, float | int]]) -> None:
    lines = [
        "# Direct-vs-Decompose Routing Gap (Final all-tl_sum Router)",
        "",
        "Setting: final prompt-fix-A candidates and all-tl_sum verifier replay with direct tau=0.0005 and post-decomposition tau=0.0158.",
        "",
        "| Dataset | Direct EM | Decomp EM | Best pure EM | SEEK final EM | Oracle EM | Oracle - SEEK | Direct choices | Decomp choices | Direct-only | Decomp-only | Both-wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in stats.items():
        lines.append(
            f"| {dataset} | {row['direct_em']:.2f} | {row['decomp_em']:.2f} | "
            f"{row['best_pure']:.2f} | {row['router_em']:.2f} | "
            f"{row['oracle_em']:.2f} | {row['oracle_vs_router']:+.2f} | "
            f"{int(row['direct_choices'])} | {int(row['decomp_choices'])} | "
            f"{int(row['direct_only'])} | {int(row['decomp_only'])} | "
            f"{int(row['both_wrong'])} |"
        )
    lines.extend([
        "",
        "The oracle is a gold-label upper bound, not a deployable method. Its gap over SEEK isolates the remaining direct-vs-decompose routing headroom; both-wrong questions remain outside that bound.",
        "",
    ])
    SUMMARY_OUT.write_text("\n".join(lines), encoding="utf-8")


def make_figure(stats: dict[str, dict[str, float | int]]) -> None:
    datasets = list(stats)
    best = [stats[d]["best_pure"] for d in datasets]
    router = [stats[d]["router_em"] for d in datasets]
    oracle = [stats[d]["oracle_em"] for d in datasets]
    gaps = [stats[d]["oracle_vs_router"] for d in datasets]
    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=170)
    bars_best = ax.bar(x - width, best, width, label="Best pure path", color="#8f969e")
    bars_router = ax.bar(x, router, width, label="SEEK final router", color="#2f7f8f")
    bars_oracle = ax.bar(x + width, oracle, width, label="Oracle selector", color="#d99a22")
    for bars in (bars_best, bars_router, bars_oracle):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    for i, gap in enumerate(gaps):
        top = oracle[i] + 3.8
        ax.plot([x[i], x[i] + width], [top, top], color="#444444", linewidth=1)
        ax.text(
            x[i] + width / 2, top + 0.9, f"+{gap:.1f}",
            ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#444444",
        )
    ax.set_title("Direct-vs-Decompose Routing Gap", fontsize=15, pad=14)
    ax.set_ylabel("Exact Match (EM)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 64)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=3)
    fig.text(
        0.12, 0.02,
        "Oracle uses gold labels to choose between the saved direct and decomposed answers; it measures routing headroom.",
        fontsize=8, color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    stats = {label: summarize(build(dataset)) for label, dataset in DATASETS}
    verify(stats)
    write_summary(stats)
    make_figure(stats)
    print(f"Wrote {OUT}")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
