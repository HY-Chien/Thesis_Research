#!/usr/bin/env python3
"""Direct-path tl_sum sensitivity to candidate-set size k.

Uses the frozen 4 x 1,000 confidence-battery traces.  Each score is
sum_t -log(m_{k,t}), with k in {2, 5, 10}.  The script reports dataset and
macro AUROC plus paired, within-dataset bootstrap comparisons against k=5.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
DISPLAY = {
    "2wiki": "2Wiki",
    "hotpotqa": "HotpotQA",
    "musique": "MuSiQue",
    "popqa": "PopQA",
}
KS = [2, 5, 10]
EPS = 1e-12
BOOTSTRAPS = 1000
SEED = 13


def auroc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUROC, with label 1 meaning wrong answer."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        rank = (i + j) / 2 + 1
        for position in range(i, j + 1):
            ranks[order[position]] = rank
        i = j + 1
    n_wrong = sum(labels)
    n_right = len(labels) - n_wrong
    mean_wrong_rank = (
        sum(rank for rank, label in zip(ranks, labels) if label) / n_wrong
    )
    return (mean_wrong_rank - (n_wrong + 1) / 2) / n_right


def token_mass(token: dict, k: int) -> float:
    if k == 2:
        return max(EPS, token["top1p"] + token["top2p"])
    if k == 5:
        return max(EPS, token["top5m"])
    if k == 10:
        return max(EPS, token["top10m"])
    raise ValueError(k)


def quantile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = index - lower
    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def main() -> None:
    labels: dict[str, list[int]] = {}
    scores: dict[str, dict[int, list[float]]] = {}
    for dataset in DATASETS:
        path = RESULTS / f"conf_battery_{dataset}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        labels[dataset] = [1 - int(row["em"]) for row in rows]
        scores[dataset] = {k: [] for k in KS}
        for row in rows:
            for k in KS:
                value = sum(
                    -math.log(token_mass(token, k))
                    for token in row["tokens"]
                )
                scores[dataset][k].append(value)

    observed: dict[int, list[float]] = {}
    for k in KS:
        observed[k] = [
            auroc(scores[dataset][k], labels[dataset])
            for dataset in DATASETS
        ]

    rng = random.Random(SEED)
    comparisons: dict[int, tuple[float, float, float, float]] = {}
    for candidate_k in [2, 10]:
        differences: list[float] = []
        for _ in range(BOOTSTRAPS):
            candidate_values: list[float] = []
            base_values: list[float] = []
            for dataset in DATASETS:
                size = len(labels[dataset])
                indices = [rng.randrange(size) for _ in range(size)]
                sampled_labels = [labels[dataset][i] for i in indices]
                candidate_values.append(
                    auroc(
                        [scores[dataset][candidate_k][i] for i in indices],
                        sampled_labels,
                    )
                )
                base_values.append(
                    auroc(
                        [scores[dataset][5][i] for i in indices],
                        sampled_labels,
                    )
                )
            differences.append(
                sum(candidate_values) / len(candidate_values)
                - sum(base_values) / len(base_values)
            )
        differences.sort()
        observed_difference = (
            sum(observed[candidate_k]) / len(DATASETS)
            - sum(observed[5]) / len(DATASETS)
        )
        lower = quantile(differences, 0.025)
        upper = quantile(differences, 0.975)
        left = sum(value <= 0 for value in differences) / BOOTSTRAPS
        right = sum(value >= 0 for value in differences) / BOOTSTRAPS
        p_value = min(1.0, 2 * min(left, right))
        comparisons[candidate_k] = (
            observed_difference,
            lower,
            upper,
            p_value,
        )

    lines = [
        "# tl_sum candidate-set sensitivity",
        "",
        "Direct-path frozen confidence traces, n=1,000 x 4; "
        "score = sum_t -log(m_{k,t}); AUROC predicts wrong answers.",
        "",
        "| k | " + " | ".join(DISPLAY[d] for d in DATASETS) + " | Macro |",
        "|---:|" + "---:|" * (len(DATASETS) + 1),
    ]
    for k in KS:
        values = observed[k]
        lines.append(
            f"| {k} | "
            + " | ".join(f"{value:.6f}" for value in values)
            + f" | {sum(values) / len(values):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Paired within-dataset bootstrap, B={BOOTSTRAPS}, seed={SEED}:",
            "",
            "| Comparison | Macro delta | 95% CI | two-sided p |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate_k in [2, 10]:
        delta, lower, upper, p_value = comparisons[candidate_k]
        lines.append(
            f"| k={candidate_k} - k=5 | {delta:+.6f} | "
            f"[{lower:+.6f}, {upper:+.6f}] | {p_value:.3f} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: k=2 is too narrow on these traces. k=5 and k=10 "
            "are on the same performance plateau; k=5 is retained as the "
            "smaller tested cutoff. This is an empirical hyperparameter, "
            "not a value implied by the entropy identity.",
            "",
        ]
    )
    output = RESULTS / "tlsum_topk_sensitivity.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
