#!/usr/bin/env python3
"""Make the direct-vs-decompose routing gap figure.

The figure compares three EM scores for each dataset:

1. Best pure path: max(direct EM, decomposed EM)
2. SEEK verifier: the current rule-based final selection
3. Oracle selector: gold-label upper bound over the two saved answers

Run from Research/main_methodology:
    python scripts/make_routing_gap_fig.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "Thesis" / "figures" / "fig_routing_gap_oracle.png"
SUMMARY_OUT = ROOT / "results" / "routing_gap_oracle.md"

FILES = {
    "HotpotQA": "hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "2Wiki": "2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "MuSiQue": "musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "PopQA": "popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(rows: list[dict]) -> dict[str, float | int]:
    n = len(rows)
    direct = sum(int(row.get("direct_em", 0)) for row in rows)
    decomp = sum(int(row.get("decomp_em", 0)) for row in rows)
    verifier = sum(int(row.get("replay_em", row.get("em", 0))) for row in rows)
    oracle = sum(
        1
        for row in rows
        if int(row.get("direct_em", 0)) or int(row.get("decomp_em", 0))
    )
    both_correct = sum(
        1
        for row in rows
        if int(row.get("direct_em", 0)) and int(row.get("decomp_em", 0))
    )
    direct_only = sum(
        1
        for row in rows
        if int(row.get("direct_em", 0)) and not int(row.get("decomp_em", 0))
    )
    decomp_only = sum(
        1
        for row in rows
        if int(row.get("decomp_em", 0)) and not int(row.get("direct_em", 0))
    )
    both_wrong = n - both_correct - direct_only - decomp_only
    direct_em = 100.0 * direct / n
    decomp_em = 100.0 * decomp / n
    verifier_em = 100.0 * verifier / n
    oracle_em = 100.0 * oracle / n
    best_pure = max(direct_em, decomp_em)
    return {
        "n": n,
        "direct_em": direct_em,
        "decomp_em": decomp_em,
        "best_pure": best_pure,
        "verifier_em": verifier_em,
        "oracle_em": oracle_em,
        "oracle_vs_best_pure": oracle_em - best_pure,
        "oracle_vs_verifier": oracle_em - verifier_em,
        "direct_only": direct_only,
        "decomp_only": decomp_only,
        "both_wrong": both_wrong,
    }


def write_summary(stats: dict[str, dict[str, float | int]]) -> None:
    lines = [
        "# Direct-vs-Decompose Routing Gap",
        "",
        "Setting: current final candidate traces, path-aware retrieval + original-question evidence union + rule-based verifier replay.",
        "",
        "| Dataset | Direct EM | Decomp EM | Best pure EM | SEEK verifier EM | Oracle EM | Oracle - verifier | direct-only | decomp-only | both-wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in stats.items():
        lines.append(
            f"| {dataset} | {row['direct_em']:.2f} | {row['decomp_em']:.2f} | "
            f"{row['best_pure']:.2f} | {row['verifier_em']:.2f} | "
            f"{row['oracle_em']:.2f} | {row['oracle_vs_verifier']:+.2f} | "
            f"{int(row['direct_only'])} | {int(row['decomp_only'])} | "
            f"{int(row['both_wrong'])} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: the oracle is not a method. It is a gold-label upper bound that quantifies the remaining headroom of the direct-vs-decompose routing decision.",
        ]
    )
    SUMMARY_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figure(stats: dict[str, dict[str, float | int]]) -> None:
    datasets = list(stats)
    best = [stats[d]["best_pure"] for d in datasets]
    verifier = [stats[d]["verifier_em"] for d in datasets]
    oracle = [stats[d]["oracle_em"] for d in datasets]
    gaps = [stats[d]["oracle_vs_verifier"] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=170)
    colors = {
        "best": "#8f969e",
        "verifier": "#2f7f8f",
        "oracle": "#d99a22",
    }

    bars_best = ax.bar(x - width, best, width, label="Best pure path", color=colors["best"])
    bars_verifier = ax.bar(x, verifier, width, label="SEEK verifier", color=colors["verifier"])
    bars_oracle = ax.bar(x + width, oracle, width, label="Oracle selector", color=colors["oracle"])

    for bars in (bars_best, bars_verifier, bars_oracle):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    for i, gap in enumerate(gaps):
        top = oracle[i] + 3.8
        ax.plot([x[i], x[i] + width], [top, top], color="#444444", linewidth=1)
        ax.text(
            x[i] + width / 2,
            top + 0.9,
            f"+{gap:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#444444",
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
        0.12,
        0.02,
        "Oracle uses gold labels to choose between saved direct/decomposed answers; it measures routing headroom, not deployable performance.",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)


def main() -> None:
    stats = {
        dataset: summarize(load_jsonl(ROOT / "results" / filename))
        for dataset, filename in FILES.items()
    }
    write_summary(stats)
    make_figure(stats)
    print(f"Wrote {OUT}")
    print(f"Wrote {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
