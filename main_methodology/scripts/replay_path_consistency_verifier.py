#!/usr/bin/env python3
"""Offline replay of path-consistency rules for final answer aggregation.

The script reads run_seek_dual/type-aware replay traces and tests whether the
agreement between direct and decomposed answers can improve the final
direct-vs-decompose verifier. It is label-based offline analysis only: no LLM
or retriever is called.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(pred: str | None, gold: str | None) -> float:
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if not n_common:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def direct_pred(row: dict[str, Any]) -> str:
    return row.get("direct_pred") or (row.get("direct") or {}).get("pred") or ""


def decomp_pred(row: dict[str, Any]) -> str:
    return row.get("decomp_pred") or (row.get("decomp") or {}).get("pred") or ""


def path_similarity(row: dict[str, Any]) -> float:
    return token_f1(direct_pred(row), decomp_pred(row))


def current_choice(row: dict[str, Any]) -> str:
    return row.get("replay_choice") or row.get("choice") or "decomp"


def choose_pred(row: dict[str, Any], choice: str) -> str:
    return direct_pred(row) if choice == "direct" else decomp_pred(row)


def choose_em(row: dict[str, Any], choice: str) -> int:
    key = "direct_em" if choice == "direct" else "decomp_em"
    return int(row.get(key) or 0)


def baseline_em(row: dict[str, Any]) -> int:
    if "replay_em" in row:
        return int(row.get("replay_em") or 0)
    return int(row.get("em") or 0)


@dataclass(frozen=True)
class EvalResult:
    em: float
    f1: float
    direct_rate: float
    changed: int
    rescued: int
    lost: int


def evaluate(rows: list[dict[str, Any]], chooser: Callable[[dict[str, Any]], str]) -> EvalResult:
    em_sum = 0
    f1_sum = 0.0
    direct_count = 0
    changed = 0
    rescued = 0
    lost = 0
    for row in rows:
        choice = chooser(row)
        em = choose_em(row, choice)
        base = baseline_em(row)
        em_sum += em
        f1_sum += token_f1(choose_pred(row, choice), row.get("gold") or "")
        direct_count += choice == "direct"
        changed += choice != current_choice(row)
        rescued += int(not base and em)
        lost += int(base and not em)

    n = len(rows)
    return EvalResult(
        em=100 * em_sum / n,
        f1=100 * f1_sum / n,
        direct_rate=100 * direct_count / n,
        changed=changed,
        rescued=rescued,
        lost=lost,
    )


def current_policy(row: dict[str, Any]) -> str:
    return current_choice(row)


def low_similarity_to_decomp(threshold: float) -> Callable[[dict[str, Any]], str]:
    def choose(row: dict[str, Any]) -> str:
        if path_similarity(row) < threshold:
            return "decomp"
        return current_choice(row)

    return choose


def partial_similarity_to_decomp(threshold: float) -> Callable[[dict[str, Any]], str]:
    """Prefer decompose only when answers overlap but are not exact.

    Low-overlap disagreement is too ambiguous: in HotpotQA it often contains
    direct-only cases that the type-aware verifier correctly rescues.
    """

    def choose(row: dict[str, Any]) -> str:
        sim = path_similarity(row)
        if threshold <= sim < 0.999999:
            return "decomp"
        return current_choice(row)

    return choose


def all_disagreement_to_decomp(row: dict[str, Any]) -> str:
    if path_similarity(row) < 0.999999:
        return "decomp"
    return current_choice(row)


def render_table(
    labels: list[str],
    rows_by_label: dict[str, list[dict[str, Any]]],
    policies: list[tuple[str, Callable[[dict[str, Any]], str]]],
) -> str:
    lines = [
        "| Policy | Dataset | EM | F1 | Direct % | Changed | Rescue/Lost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    macro: dict[str, list[EvalResult]] = {name: [] for name, _ in policies}

    for policy_name, chooser in policies:
        for label in labels:
            result = evaluate(rows_by_label[label], chooser)
            macro[policy_name].append(result)
            lines.append(
                f"| {policy_name} | {label} | {result.em:.2f} | {result.f1:.2f} | "
                f"{result.direct_rate:.1f} | {result.changed} | "
                f"{result.rescued}/{result.lost} |"
            )

    lines.extend(["", "| Policy | Macro EM | Macro F1 | Macro Direct % |", "|---|---:|---:|---:|"])
    for policy_name, _ in policies:
        results = macro[policy_name]
        lines.append(
            f"| {policy_name} | "
            f"{sum(r.em for r in results) / len(results):.2f} | "
            f"{sum(r.f1 for r in results) / len(results):.2f} | "
            f"{sum(r.direct_rate for r in results) / len(results):.1f} |"
        )
    return "\n".join(lines)


def render_bucket_summary(labels: list[str], rows_by_label: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| Dataset | Bucket | N | Current EM | Direct EM | Decomp EM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        buckets: dict[str, list[dict[str, Any]]] = {
            "exact": [],
            "partial>=0.5": [],
            "diff<0.5": [],
        }
        for row in rows_by_label[label]:
            sim = path_similarity(row)
            if sim >= 0.999999:
                buckets["exact"].append(row)
            elif sim >= 0.5:
                buckets["partial>=0.5"].append(row)
            else:
                buckets["diff<0.5"].append(row)
        for bucket_name, rows in buckets.items():
            if not rows:
                continue
            n = len(rows)
            cur = sum(baseline_em(row) for row in rows)
            direct = sum(int(row.get("direct_em") or 0) for row in rows)
            decomp = sum(int(row.get("decomp_em") or 0) for row in rows)
            lines.append(
                f"| {label} | {bucket_name} | {n} | "
                f"{100 * cur / n:.2f} | {100 * direct / n:.2f} | {100 * decomp / n:.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.labels) != len(args.paths):
        raise SystemExit("--labels must have the same length as paths")

    rows_by_label = {
        label: load_jsonl(path)
        for label, path in zip(args.labels, args.paths)
    }
    policies: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("current", current_policy),
        ("pc_low<0.5_to_decomp", low_similarity_to_decomp(0.5)),
        ("pc_low<0.75_to_decomp", low_similarity_to_decomp(0.75)),
        ("pc_partial>=0.25_to_decomp", partial_similarity_to_decomp(0.25)),
        ("pc_partial>=0.5_to_decomp", partial_similarity_to_decomp(0.5)),
        ("pc_any_disagree_to_decomp", all_disagreement_to_decomp),
    ]

    text = "\n\n".join(
        [
            "# Path-Consistency Final Aggregation Replay",
            "This replay uses saved direct/decompose outputs only; it does not rerun the LLM.",
            "## Bucket Summary",
            render_bucket_summary(args.labels, rows_by_label),
            "## Policy Replay",
            render_table(args.labels, rows_by_label, policies),
        ]
    )
    text += "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
