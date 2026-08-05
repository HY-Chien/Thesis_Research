#!/usr/bin/env python3
"""Plot direct-path correctness across equal-frequency tl_sum bins.

This is the tl_sum counterpart of the earlier MAX_ENTROPY stacked-bar figure.
It deliberately uses the frozen confidence-battery traces because tl_sum is a
direct greedy-path signal; correctness is the stored alias-aware exact match.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "main_methodology" / "results"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

DATASETS = ("2wiki", "hotpotqa", "musique", "popqa")
N_BINS = 8
EPS = 1e-12


def tlsum(tokens: list[dict]) -> float:
    """Sum per-token top-5 retained-mass surprisal."""
    return sum(
        -math.log(max(EPS, min(1.0, float(token["top5m"]))))
        for token in tokens
    )


def load_all() -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    correct: list[int] = []
    for dataset in DATASETS:
        path = RESULTS_DIR / f"conf_battery_{dataset}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("tokens"):
                    continue
                scores.append(tlsum(row["tokens"]))
                correct.append(int(row["em"]))
    return np.asarray(scores), np.asarray(correct)


def equal_frequency_bins(
    scores: np.ndarray,
    correct: np.ndarray,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, kind="stable")
    edges = np.linspace(0, len(scores), n_bins + 1).astype(int)
    medians: list[float] = []
    accuracies: list[float] = []
    sizes: list[int] = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        indices = order[lower:upper]
        medians.append(float(np.median(scores[indices])))
        accuracies.append(float(100 * correct[indices].mean()))
        sizes.append(len(indices))
    return np.asarray(medians), np.asarray(accuracies), np.asarray(sizes)


def format_median(value: float) -> str:
    if value < 0.1:
        return f"{value:.4f}"
    return f"{value:.3f}"


def plot(scores: np.ndarray, correct: np.ndarray) -> tuple[Path, Path, Path]:
    medians, accuracies, sizes = equal_frequency_bins(scores, correct)
    errors = 100 - accuracies
    rho, p_value = stats.spearmanr(scores, 1 - correct)

    x = np.arange(len(medians))
    width = 0.66
    green = "#27AE60"
    red = "#C0392B"
    dark = "#202124"
    muted = "#5F6368"

    fig, ax = plt.subplots(figsize=(10.4, 6.35), facecolor="white")
    ax.set_facecolor("white")

    ax.bar(
        x,
        accuracies,
        width,
        color=green,
        label="Accuracy (correct %)",
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )
    ax.bar(
        x,
        errors,
        width,
        bottom=accuracies,
        color=red,
        label="Error (incorrect %)",
        edgecolor="white",
        linewidth=1.5,
        zorder=3,
    )

    for i, (accuracy, error) in enumerate(zip(accuracies, errors)):
        if accuracy >= 5:
            ax.text(
                x[i],
                accuracy / 2,
                f"{accuracy:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )
        else:
            ax.annotate(
                f"{accuracy:.0f}%",
                xy=(x[i], accuracy),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=green,
                fontsize=9.5,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": "none"},
                zorder=5,
            )
        if error >= 5:
            ax.text(
                x[i],
                accuracy + error / 2,
                f"{error:.0f}%",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
                zorder=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([format_median(value) for value in medians], fontsize=10.5)
    ax.set_xlabel(
        "Tail Leakage  (low → high)",
        fontsize=12,
        labelpad=10,
    )
    ax.set_ylabel("Proportion (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.tick_params(axis="both", colors=dark)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA0A6")
    ax.spines["bottom"].set_color("#9AA0A6")

    ax.set_title(
        "Higher Tail Leakage Corresponds to More Errors\n"
        f"(direct-path pooled n = {len(scores):,}; equal-frequency bins; "
        f"Spearman ρ = {rho:+.3f})",
        fontsize=14,
        fontweight="bold",
        color=dark,
        pad=13,
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        fontsize=11.5,
        frameon=False,
    )
    fig.text(
        0.5,
        0.015,
        "Frozen direct greedy traces • correct = alias-aware EM • "
        f"{int(sizes.min())} questions per bin • Spearman p = {p_value:.1e}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=muted,
    )

    fig.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.25)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig1_tlsum_stacked_accuracy_error.png"
    pdf = FIG_DIR / "fig1_tlsum_stacked_accuracy_error.pdf"
    svg = FIG_DIR / "fig1_tlsum_stacked_accuracy_error.svg"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    fig.savefig(svg, facecolor="white")
    plt.close(fig)

    print(f"n={len(scores):,}; Spearman(tl_sum,error)={rho:+.6f}, p={p_value:.3e}")
    print("bin\tmedian_tl_sum\taccuracy_pct\terror_pct\tn")
    for i, (median, accuracy, error, size) in enumerate(
        zip(medians, accuracies, errors, sizes), start=1
    ):
        print(f"{i}\t{median:.9f}\t{accuracy:.1f}\t{error:.1f}\t{size}")
    print(f"saved: {png}")
    print(f"saved: {pdf}")
    print(f"saved: {svg}")
    return png, pdf, svg


if __name__ == "__main__":
    plot(*load_all())
