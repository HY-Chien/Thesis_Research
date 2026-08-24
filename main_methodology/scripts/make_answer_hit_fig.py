#!/usr/bin/env python3
"""Make the four-dataset open-domain answer-containing retrieval figure.

Values are read from the report produced by
``recompute_answer_containing_recall_4ds.py``. By default that report and the
new figure are both placed under ``results/reproduced``.
The metric checks whether top-k retrieved passages contain the gold answer or an
alias under the final open-domain wiki retrieval setting.

Run from ``Research/main_methodology``:
    python scripts/recompute_answer_containing_recall_4ds.py
    python scripts/make_answer_hit_fig.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results/reproduced/answer_containing_recall_4ds_open_domain.md"
)
DEFAULT_OUT = ROOT / "results/reproduced/fig_answer_hit_retrieval.png"
ROW_RE = re.compile(
    r"^\| (?P<dataset>2Wiki|HotpotQA|MuSiQue|PopQA) \| (?P<n>\d+) \| "
    r"(?P<orig>[0-9.]+) \| \*\*(?P<decomp>[0-9.]+)\*\* \| "
    r"\*\*(?P<union>[0-9.]+)\*\* \|"
)


def load_values(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    datasets: list[str] = []
    original: list[float] = []
    decomposed: list[float] = []
    union: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if not match:
            continue
        datasets.append(match.group("dataset"))
        original.append(float(match.group("orig")))
        decomposed.append(float(match.group("decomp")))
        union.append(float(match.group("union")))
    if datasets != ["2Wiki", "HotpotQA", "MuSiQue", "PopQA"]:
        raise RuntimeError(f"Unexpected or missing rows in {path}: {datasets}")
    return datasets, np.array(original), np.array(decomposed), np.array(union)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="answer-containing recall Markdown report",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help="output PNG (default: results/reproduced)",
    )
    args = parser.parse_args()

    datasets, original, decomposed, union = load_values(args.input)
    gains = decomposed - original
    x = np.arange(len(datasets))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=170)
    bars_a = ax.bar(x - width / 2, original, width, label="Original query", color="#aeb7c5")
    bars_b = ax.bar(x + width / 2, decomposed, width, label="Decomposed query", color="#2f78b7")

    for bars in (bars_a, bars_b):
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

    for i, gain in enumerate(gains):
        ax.text(
            x[i],
            decomposed[i] + 4.2,
            f"{gain:+.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#1c4f7a",
        )

    ax.set_title("Open-domain Answer-containing Retrieval Hit@5", fontsize=15, pad=14)
    ax.set_ylabel("Hit rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 82)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncols=2)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
