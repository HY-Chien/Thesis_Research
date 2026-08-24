#!/usr/bin/env python3
"""Plot the final step-gate accuracy--retrieval trade-off for the thesis.

Values are aggregated directly from the machine-readable fixed-chain replay in
``results/reproduced/tlsum_fixA_step_gate_final.json``. The x-axis uses the observed
pooled step retrieval rate so the efficiency/accuracy trade-off is visible
without a second y-axis.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/reproduced/tlsum_fixA_step_gate_final.json"
DEFAULT_OUT = ROOT / "results/reproduced/fig_threshold_tradeoff.png"
DATASETS = ("hotpotqa", "2wiki", "musique", "popqa")
OPERATING_POINTS = (
    ("tau_0.1165", 0.116500),
    ("tau_0.07467", 0.074670),
    ("tau_0.04417", 0.044170),
    ("tau_0.02216", 0.022160),
    ("deployed", 0.008461),
    ("tau_0.002305", 0.002305),
    ("always", 0.0),
)


def load_values(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = payload.get("protocol") or {}
    if protocol.get("step_gate_signal") != "tl_sum":
        raise RuntimeError(f"{path} is not a Tail-Leakage step-gate replay")
    results = payload["results"]
    retrieval_rates: list[float] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    for policy, _ in OPERATING_POINTS:
        rows = [results[dataset][policy] for dataset in DATASETS]
        retrieved = sum(row["retrieved_steps"] for row in rows)
        total = sum(row["total_steps"] for row in rows)
        retrieval_rates.append(100.0 * retrieved / total)
        em_scores.append(float(np.mean([row["em"] for row in rows])))
        f1_scores.append(float(np.mean([row["f1"] for row in rows])))

    return (
        np.asarray(retrieval_rates),
        np.asarray(em_scores),
        np.asarray(f1_scores),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="machine-readable fixed-chain replay JSON",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="output PNG (default: results/reproduced)",
    )
    args = parser.parse_args()

    rates, em, f1 = load_values(args.input)
    thresholds = [threshold for _, threshold in OPERATING_POINTS]
    headline_index = 5

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=170)
    ax.axvspan(
        rates[headline_index] - 0.65,
        rates[headline_index] + 0.65,
        color="#f2c14e",
        alpha=0.18,
        zorder=0,
    )
    ax.plot(
        rates,
        em,
        color="#2f78b7",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Macro EM",
    )
    ax.plot(
        rates,
        f1,
        color="#26866a",
        marker="s",
        linewidth=2.2,
        markersize=6,
        label="Macro F1",
    )
    ax.scatter(
        [rates[headline_index], rates[headline_index]],
        [em[headline_index], f1[headline_index]],
        s=115,
        facecolors="white",
        edgecolors="#b77900",
        linewidths=2.2,
        zorder=5,
        label="Accuracy-first headline",
    )

    for x_value, score in zip(rates, em):
        ax.annotate(
            f"{score:.2f}",
            (x_value, score),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#245a86",
        )
    for x_value, score in zip(rates, f1):
        ax.annotate(
            f"{score:.2f}",
            (x_value, score),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f684f",
        )

    tick_labels = [
        f"{rate:.2f}\n$\\tau$={threshold:.6f}"
        for rate, threshold in zip(rates, thresholds)
    ]
    tick_labels[headline_index] += "\n(headline)"
    tick_labels[-1] = "100.00\n$\\tau$=0\n(all retrieve)"
    ax.set_xticks(rates)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_xlabel("Pooled step retrieval rate (%)\nThreshold shown below each operating point")
    ax.set_ylabel("Macro score (%)")
    ax.set_title("Accuracy–Retrieval Trade-off of the Step-level Confidence Gate", fontsize=14, pad=13)
    ax.set_xlim(rates[0] - 2.0, rates[-1] + 2.0)
    ax.set_ylim(38.6, 50.1)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=3, fontsize=9)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
