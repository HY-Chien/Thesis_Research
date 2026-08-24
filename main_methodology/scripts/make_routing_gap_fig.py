#!/usr/bin/env python3
"""Make the accuracy-first direct-vs-decompose routing-gap figure.

Candidates are reconstructed at the thesis headline step-gate threshold
(``tau=0.002305``); routing uses the same final all-Tail-Leakage/type-aware
verifier as the headline analysis.
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

DEFAULT_OUT = ROOT / "results/reproduced/fig_routing_gap_oracle.png"
DEFAULT_SUMMARY = ROOT / "results/reproduced/routing_gap_oracle_accuracy_first.md"
DATASETS = [
    ("HotpotQA", "hotpotqa"),
    ("2Wiki", "2wiki"),
    ("MuSiQue", "musique"),
    ("PopQA", "popqa"),
]
def summarize(rows: list[dict]) -> dict[str, float | int]:
    n = len(rows)
    choices = [row["final_choice"] for row in rows]
    direct = sum(int(row["direct_em"]) for row in rows)
    decomp = sum(int(row["decomp_em"]) for row in rows)
    router = sum(int(row["final_em"]) for row in rows)
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
    direct_only_recovered = sum(
        int(
            row["direct_em"]
            and not row["decomp_em"]
            and row["final_choice"] == "direct"
        )
        for row in rows
    )
    decomp_only_lost = sum(
        int(
            not row["direct_em"]
            and row["decomp_em"]
            and row["final_choice"] == "direct"
        )
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
        "direct_only_recovered": direct_only_recovered,
        "decomp_only_lost": decomp_only_lost,
    }


def verify(stats: dict[str, dict[str, float | int]]) -> None:
    """Check count identities without embedding thesis result values."""
    for dataset, row in stats.items():
        n = int(row["n"])
        if int(row["direct_choices"]) + int(row["decomp_choices"]) != n:
            raise AssertionError(f"{dataset}: routing choices do not sum to n={n}")
        if int(row["oracle_correct"]) + int(row["both_wrong"]) != n:
            raise AssertionError(f"{dataset}: oracle/both-wrong counts do not sum to n={n}")
        router_delta = int(row["router_correct"]) - int(row["decomp_correct"])
        path_delta = (
            int(row["direct_only_recovered"])
            - int(row["decomp_only_lost"])
        )
        if router_delta != path_delta:
            raise AssertionError(
                f"{dataset}: routing delta {router_delta} != path delta {path_delta}"
            )


def write_summary(stats: dict[str, dict[str, float | int]], path: Path) -> None:
    lines = [
        "# Direct-vs-Decompose Routing Gap (Accuracy-First Replay)",
        "",
        "Setting: prompt-fix-A candidates replayed with the Tail-Leakage step gate at tau=0.002305 and the final all-Tail-Leakage verifier (direct tau=0.0005; post-decomposition tau=0.0158).",
        "",
        "| Dataset | Direct EM | Decomp EM | SEEK final EM | Oracle EM | Oracle - SEEK | Direct choices | Direct-only recovered | Decomp-only lost | Direct-only | Decomp-only | Both-wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in stats.items():
        lines.append(
            f"| {dataset} | {row['direct_em']:.2f} | {row['decomp_em']:.2f} | "
            f"{row['router_em']:.2f} | "
            f"{row['oracle_em']:.2f} | {row['oracle_vs_router']:+.2f} | "
            f"{int(row['direct_choices'])} | "
            f"{int(row['direct_only_recovered'])} | "
            f"{int(row['decomp_only_lost'])} | "
            f"{int(row['direct_only'])} | {int(row['decomp_only'])} | "
            f"{int(row['both_wrong'])} |"
        )
    lines.extend([
        "",
        "The oracle is a gold-label upper bound, not a deployable method. Its gap over SEEK isolates the remaining direct-vs-decompose routing headroom; both-wrong questions remain outside that bound.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figure(stats: dict[str, dict[str, float | int]], path: Path) -> None:
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
        "--out", type=Path, default=DEFAULT_OUT,
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
    make_figure(stats, args.out)
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
