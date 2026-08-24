#!/usr/bin/env python3
"""Analyze direct-path tl_sum by accuracy-first correctness quadrant.

Candidates are reconstructed from the tracked prompt-fix-A traces at the
thesis headline step-gate threshold (``tau=0.002305``). Final routing uses the
same all-Tail-Leakage/type-aware verifier as the headline analysis. No legacy
confidence field is used.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import mean, median, pstdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_confidence_battery import auroc  # noqa: E402
from accuracy_first_replay import (  # noqa: E402
    DEFAULT_SYNTH_PATHS,
    load_all_accuracy_first_rows,
)

RESULTS = ROOT / "results"
DEFAULT_FIG = RESULTS / "reproduced/fig_quadrant_confidence.png"
DEFAULT_REPORT = RESULTS / "reproduced/quadrant_confidence_analysis.md"
DATASETS = [
    ("HotpotQA", "hotpotqa"),
    ("2Wiki", "2wiki"),
    ("MuSiQue", "musique"),
    ("PopQA", "popqa"),
]
QUADRANTS = ("both_correct", "direct_only", "decomp_only", "both_wrong")
QUADRANT_COLORS = ["#2f7f8f", "#244a8f", "#0b8f6a", "#c05a47"]


def quadrant(row: dict) -> str:
    direct_ok = bool(row["direct_em"])
    decomp_ok = bool(row["decomp_em"])
    if direct_ok and decomp_ok:
        return "both_correct"
    if direct_ok:
        return "direct_only"
    if decomp_ok:
        return "decomp_only"
    return "both_wrong"


def validate_final_rows(dataset: str, rows: list[dict]) -> list[dict]:
    for row in rows:
        value = row.get("direct_tl")
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"{dataset}/{row['id']}: missing direct tl_sum")
        if row.get("final_choice") not in {"direct", "decomp"}:
            raise RuntimeError(f"{dataset}/{row['id']}: invalid final choice")
    return rows


def bucket_rows(rows: list[dict]) -> dict[str, list[dict]]:
    buckets = {name: [] for name in QUADRANTS}
    for row in rows:
        buckets[quadrant(row)].append(row)
    return buckets


def values(rows: list[dict]) -> np.ndarray:
    return np.asarray([float(row["direct_tl"]) for row in rows], dtype=float)


def cohens_d(wrong: np.ndarray, correct: np.ndarray) -> float:
    if len(wrong) < 2 or len(correct) < 2:
        return float("nan")
    wrong_var = pstdev(wrong) ** 2
    correct_var = pstdev(correct) ** 2
    pooled = math.sqrt(
        ((len(wrong) - 1) * wrong_var + (len(correct) - 1) * correct_var)
        / (len(wrong) + len(correct) - 2)
    )
    return (mean(wrong) - mean(correct)) / pooled if pooled else float("nan")


def separation(correct: np.ndarray, wrong: np.ndarray) -> dict[str, float | int]:
    scores = np.concatenate([correct, wrong])
    labels = np.concatenate([
        np.zeros(len(correct), dtype=int),
        np.ones(len(wrong), dtype=int),
    ])
    correct_median = float(np.median(correct))
    confidently_wrong = 100.0 * float(np.mean(wrong <= correct_median))
    return {
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "mean_correct": float(np.mean(correct)),
        "mean_wrong": float(np.mean(wrong)),
        "median_correct": correct_median,
        "median_wrong": float(np.median(wrong)),
        "auroc": float(auroc(scores, labels)),
        "cohens_d": float(cohens_d(wrong, correct)),
        "confidently_wrong": confidently_wrong,
    }


def compute_stats(
    data: dict[str, dict[str, list[dict]]],
) -> tuple[dict[str, dict[str, float | int]], dict[str, float | int]]:
    by_dataset = {}
    pooled_correct: list[float] = []
    pooled_wrong: list[float] = []
    for dataset, buckets in data.items():
        correct = values(buckets["both_correct"])
        wrong = values(buckets["both_wrong"])
        by_dataset[dataset] = separation(correct, wrong)
        pooled_correct.extend(correct.tolist())
        pooled_wrong.extend(wrong.tolist())
    pooled = separation(
        np.asarray(pooled_correct, dtype=float),
        np.asarray(pooled_wrong, dtype=float),
    )
    return by_dataset, pooled


def verify(
    data: dict[str, dict[str, list[dict]]],
    stats: dict[str, dict[str, float | int]],
    pooled: dict[str, float | int],
) -> None:
    for dataset, buckets in data.items():
        observed = sum(len(buckets[name]) for name in QUADRANTS)
        if observed <= 0:
            raise AssertionError(f"{dataset}: no replay rows")
    to_check = {**stats, "Pooled": pooled}
    for dataset, row in to_check.items():
        required = ("mean_correct", "mean_wrong", "auroc", "confidently_wrong")
        if not all(math.isfinite(float(row[name])) for name in required):
            raise AssertionError(f"{dataset}: non-finite separation statistic")


def render_quadrant_table(
    data: dict[str, dict[str, list[dict]]],
) -> str:
    lines = [
        "| Dataset | Quadrant | n | Mean tl_sum | Median tl_sum | Final router picks direct |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset, buckets in data.items():
        for name in QUADRANTS:
            rows = buckets[name]
            signal = values(rows)
            direct_rate = 100.0 * sum(
                row["final_choice"] == "direct" for row in rows
            ) / len(rows)
            lines.append(
                f"| {dataset} | {name} | {len(rows)} | {np.mean(signal):.5f} | "
                f"{np.median(signal):.6f} | {direct_rate:.1f}% |"
            )
    return "\n".join(lines)


def render_separation_table(
    stats: dict[str, dict[str, float | int]],
    pooled: dict[str, float | int],
) -> str:
    lines = [
        "| Dataset | Both-correct n | Both-wrong n | Mean both-correct | Mean both-wrong | Median both-correct | Median both-wrong | AUROC | Confidently wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, row in [*stats.items(), ("Pooled", pooled)]:
        lines.append(
            f"| {dataset} | {row['n_correct']} | {row['n_wrong']} | "
            f"{row['mean_correct']:.5f} | {row['mean_wrong']:.5f} | "
            f"{row['median_correct']:.6f} | {row['median_wrong']:.6f} | "
            f"{row['auroc']:.3f} | {row['confidently_wrong']:.1f}% |"
        )
    return "\n".join(lines)


def render_disagreement_table(
    data: dict[str, dict[str, list[dict]]],
) -> str:
    lines = [
        "| Dataset | Quadrant | n | Final router picks correct path |",
        "|---|---|---:|---:|",
    ]
    for dataset, buckets in data.items():
        for name, correct_choice in (("direct_only", "direct"), ("decomp_only", "decomp")):
            rows = buckets[name]
            hit_rate = 100.0 * sum(
                row["final_choice"] == correct_choice for row in rows
            ) / len(rows)
            lines.append(f"| {dataset} | {name} | {len(rows)} | {hit_rate:.1f}% |")
    return "\n".join(lines)


def write_report(
    data: dict[str, dict[str, list[dict]]],
    stats: dict[str, dict[str, float | int]],
    pooled: dict[str, float | int],
    path: Path,
) -> None:
    lines = [
        "# Direct-path tl_sum by Correctness Quadrant",
        "",
        "Setting: prompt-fix-A candidates replayed with the Tail-Leakage step gate at tau=0.002305, path-aware retrieval, original-question evidence union, and the final all-Tail-Leakage verifier (direct tau=0.0005; post-decomposition tau=0.0158).",
        "",
        "The signal is direct-path tail-leakage surprisal (`direct_tl`); higher values indicate greater uncertainty.",
        "",
        "## 1. All correctness quadrants",
        "",
        render_quadrant_table(data),
        "",
        "## 2. Both-correct versus both-wrong",
        "",
        "AUROC ranks both-wrong above both-correct. Confidently wrong is the share of both-wrong examples at or below the both-correct median tl_sum.",
        "",
        render_separation_table(stats, pooled),
        "",
        "## 3. Final routing inside disagreement quadrants",
        "",
        render_disagreement_table(data),
        "",
        "## Interpretation",
        "",
        "Direct-path tl_sum separates both-wrong from both-correct examples on every dataset "
        f"(pooled AUROC {pooled['auroc']:.3f}), but the distributions overlap. Using the "
        "pooled both-correct median still leaves "
        f"{pooled['confidently_wrong']:.1f}% of both-wrong examples confidently wrong. "
        "Confidence is therefore informative at the population level without being a "
        "reliable per-question correctness test.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def make_fig(data: dict[str, dict[str, list[dict]]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), dpi=170, sharey=True)
    labels = ["Both\ncorrect", "Direct\nonly", "Decomp\nonly", "Both\nwrong"]
    for ax, (dataset, buckets) in zip(axes.flat, data.items()):
        series = [values(buckets[name]) for name in QUADRANTS]
        boxplot = ax.boxplot(
            series, widths=0.55, showfliers=False, patch_artist=True,
            medianprops={"color": "#222222", "linewidth": 1.3},
        )
        for patch, color in zip(boxplot["boxes"], QUADRANT_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        ax.set_yscale("symlog", linthresh=0.001, linscale=0.7)
        ax.set_title(dataset, fontsize=12)
        ax.set_xticks(range(1, len(QUADRANTS) + 1))
        ax.set_xticklabels(labels, fontsize=9)
        for index, name in enumerate(QUADRANTS, 1):
            ax.text(
                index, 0.02, f"n={len(buckets[name])}",
                transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color="#666666",
            )
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0, 0].set_ylabel("Tail-leakage surprisal\n(direct answer)")
    axes[1, 0].set_ylabel("Tail-leakage surprisal\n(direct answer)")
    fig.suptitle("Tail-Leakage Surprisal by Correctness Quadrant", fontsize=14)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
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
        "--report-out", type=Path, default=DEFAULT_REPORT,
        help="output Markdown report (default: results/reproduced)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_FIG,
        help="output PNG (default: results/reproduced)",
    )
    args = parser.parse_args()

    loaded = load_all_accuracy_first_rows(args.synth)
    all_rows = {
        label: validate_final_rows(dataset, loaded[dataset])
        for label, dataset in DATASETS
    }
    data = {label: bucket_rows(rows) for label, rows in all_rows.items()}
    stats, pooled = compute_stats(data)
    verify(data, stats, pooled)
    write_report(data, stats, pooled, args.report_out)
    make_fig(data, args.out)
    for dataset, buckets in data.items():
        print(dataset, {name: len(buckets[name]) for name in QUADRANTS})
    print(f"Wrote {args.report_out}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
