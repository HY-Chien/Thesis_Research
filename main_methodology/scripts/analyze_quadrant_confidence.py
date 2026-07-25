#!/usr/bin/env python3
"""Analyze direct-path tl_sum by direct/decompose correctness quadrant.

All candidate answers come from the final prompt-fix-A traces loaded with
``replay_verifier_tlsum_fixA.build``. Final routing choices are replayed with
``decide(..., 'both', 0.0005, 0.0158)``. No legacy confidence field is used.
"""
from __future__ import annotations

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
from replay_verifier_tlsum_fixA import build, decide  # noqa: E402

RESULTS = ROOT / "results"
THESIS_FIG = ROOT.parent / "Thesis" / "figures" / "fig_quadrant_confidence.png"
REPORT = RESULTS / "quadrant_confidence_analysis.md"
DATASETS = [
    ("HotpotQA", "hotpotqa"),
    ("2Wiki", "2wiki"),
    ("MuSiQue", "musique"),
    ("PopQA", "popqa"),
]
QUADRANTS = ("both_correct", "direct_only", "decomp_only", "both_wrong")
DIRECT_TAU = 0.0005
POST_DECOMP_TAU = 0.0158
EXPECTED_COUNTS = {
    "HotpotQA": (225, 78, 132, 565),
    "2Wiki": (140, 40, 350, 470),
    "MuSiQue": (76, 26, 191, 707),
    "PopQA": (345, 33, 193, 429),
}
EXPECTED_SEPARATION = {
    # mean correct, mean wrong, AUROC, confidently-wrong percentage
    "HotpotQA": (0.01277764593698903, 0.13038910097819073, 0.7770619469026548, 15.575221238938052),
    "2Wiki": (0.03485271656359562, 0.20102858360310327, 0.7474316109422492, 21.06382978723404),
    "MuSiQue": (0.04983399811698286, 0.250543257081529, 0.7632323382714211, 12.446958981612447),
    "PopQA": (0.009770083001594857, 0.1025779216833659, 0.8452214452214453, 4.895104895104895),
    "Pooled": (0.018972535810644835, 0.17931519461305448, 0.8158498036223502, 9.304467987102717),
}
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


def load_final_rows(dataset: str) -> list[dict]:
    rows = build(dataset)
    for row in rows:
        value = row.get("direct_tl")
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"{dataset}/{row['id']}: missing direct tl_sum")
        row["final_choice"] = decide(
            row, "both", DIRECT_TAU, POST_DECOMP_TAU,
        )
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
    for dataset, expected in EXPECTED_COUNTS.items():
        observed = tuple(len(data[dataset][name]) for name in QUADRANTS)
        if observed != expected:
            raise AssertionError(f"{dataset}: expected {expected}, observed {observed}")
    to_check = {**stats, "Pooled": pooled}
    for dataset, expected in EXPECTED_SEPARATION.items():
        row = to_check[dataset]
        observed = (
            row["mean_correct"], row["mean_wrong"],
            row["auroc"], row["confidently_wrong"],
        )
        if not all(np.isclose(a, b, rtol=0.0, atol=1e-12) for a, b in zip(observed, expected)):
            raise AssertionError(f"{dataset}: expected {expected}, observed {observed}")


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
) -> None:
    lines = [
        "# Direct-path tl_sum by Correctness Quadrant",
        "",
        "Setting: final prompt-fix-A candidates with the tl_sum step gate, path-aware retrieval, original-question evidence union, and the final all-tl_sum verifier (direct tau=0.0005; post-decomposition tau=0.0158).",
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
        "Direct-path tl_sum separates both-wrong from both-correct examples on every dataset (pooled AUROC 0.816), but the distributions overlap. Using the pooled both-correct median still leaves 9.3% of both-wrong examples confidently wrong. Confidence is therefore informative at the population level without being a reliable per-question correctness test.",
        "",
    ]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def make_fig(data: dict[str, dict[str, list[dict]]]) -> None:
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
    THESIS_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(THESIS_FIG, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    all_rows = {
        label: load_final_rows(dataset)
        for label, dataset in DATASETS
    }
    data = {label: bucket_rows(rows) for label, rows in all_rows.items()}
    stats, pooled = compute_stats(data)
    verify(data, stats, pooled)
    write_report(data, stats, pooled)
    make_fig(data)
    for dataset, buckets in data.items():
        print(dataset, {name: len(buckets[name]) for name in QUADRANTS})
    print(f"Wrote {REPORT}")
    print(f"Wrote {THESIS_FIG}")


if __name__ == "__main__":
    main()
