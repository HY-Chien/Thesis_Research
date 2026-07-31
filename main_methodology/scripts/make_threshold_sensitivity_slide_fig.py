#!/usr/bin/env python3
"""Create a slide-ready trend chart from the four threshold-table settings."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "tlsum_fixA_step_gate_final.json"
OUT = ROOT.parent / "Thesis" / "figures" / "fig_threshold_sensitivity_slide.png"
DATASETS = ("hotpotqa", "2wiki", "musique", "popqa")

# Ordered by observed retrieval rate so the accuracy--cost trend reads left to right.
SETTINGS = (
    ("tau_0.1165", 0.116500),
    ("tau_0.04417", 0.044170),
    ("deployed", 0.008461),
    ("tau_0.002305", 0.002305),
)


def load_values() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))["results"]
    rates: list[float] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []

    for policy, _ in SETTINGS:
        rows = [results[dataset][policy] for dataset in DATASETS]
        rates.append(
            100.0
            * sum(row["retrieved_steps"] for row in rows)
            / sum(row["total_steps"] for row in rows)
        )
        em_scores.append(float(np.mean([row["em"] for row in rows])))
        f1_scores.append(float(np.mean([row["f1"] for row in rows])))

    rates_array = np.asarray(rates)
    em_array = np.asarray(em_scores)
    f1_array = np.asarray(f1_scores)

    # Keep the chart tied to the audited values shown in the presentation table.
    if not (
        np.allclose(rates_array, [64.40, 75.51, 87.82, 93.84], atol=0.005)
        and np.allclose(em_array, [39.42, 40.60, 41.62, 41.88], atol=0.005)
        and np.allclose(f1_array, [46.62, 47.72, 48.66, 48.90], atol=0.005)
    ):
        raise RuntimeError("Figure values do not match the audited threshold table")

    return rates_array, em_array, f1_array


def main() -> None:
    rates, em, f1 = load_values()
    thresholds = [threshold for _, threshold in SETTINGS]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#263746",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
        }
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.625), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfcfe")

    ax.plot(
        rates,
        f1,
        color="#16836f",
        marker="s",
        linewidth=3.0,
        markersize=7.5,
        label="Average F1",
        zorder=3,
    )
    ax.plot(
        rates,
        em,
        color="#176b9b",
        marker="o",
        linewidth=3.0,
        markersize=7.5,
        label="Average EM",
        zorder=3,
    )

    # Emphasize the accuracy-first setting used by the final system.
    final_index = len(rates) - 1
    ax.axvspan(rates[final_index] - 1.15, rates[final_index] + 1.15, color="#d8eacb", alpha=0.65)
    ax.scatter(
        [rates[final_index], rates[final_index]],
        [em[final_index], f1[final_index]],
        s=150,
        facecolors="white",
        edgecolors="#4b7f28",
        linewidths=2.5,
        zorder=5,
    )

    for x_value, score in zip(rates, f1):
        ax.annotate(
            f"{score:.2f}",
            (x_value, score),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#126956",
        )
    for x_value, score in zip(rates, em):
        ax.annotate(
            f"{score:.2f}",
            (x_value, score),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="#14577d",
        )

    tick_labels = [
        f"{rate:.2f}%\n$\\tau$={threshold:.6f}"
        for rate, threshold in zip(rates, thresholds)
    ]
    tick_labels[-1] += "\n(final)"
    ax.set_xticks(rates)
    ax.set_xticklabels(tick_labels, fontsize=9.5)
    ax.set_xlabel("Average retrieval rate  →  more retrieval", fontsize=11.5, labelpad=10)
    ax.set_ylabel("Average score", fontsize=11.5)
    ax.set_title("Threshold Sensitivity: Accuracy–Retrieval Trade-off", fontsize=17, pad=15)
    ax.text(
        0.5,
        1.01,
        "Lower thresholds retrieve more often and preserve EM/F1",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#5a6b7b",
    )

    ax.set_xlim(61.5, 97.0)
    ax.set_ylim(37.8, 50.4)
    ax.grid(axis="y", linestyle="--", linewidth=0.9, alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#aab4bd")
    ax.spines["bottom"].set_color("#aab4bd")
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=10.5)

    ax.annotate(
        "Best average EM/F1",
        xy=(rates[final_index], f1[final_index]),
        xytext=(-95, 39),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#365f1d",
        arrowprops={"arrowstyle": "-|>", "color": "#4b7f28", "lw": 1.8},
    )

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
