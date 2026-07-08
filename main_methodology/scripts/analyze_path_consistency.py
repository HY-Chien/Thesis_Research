#!/usr/bin/env python3
"""Analyze direct/decomposition answer consistency as a QA confidence signal.

This is an offline diagnostic over run_seek_dual traces. It does not rerun the
model. The main question is whether agreement between the direct-path answer
and the decomposition-path answer predicts correctness strongly enough to be
useful as a confidence or escalation signal.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")

    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        rank_sum += sum(label for _, label in pairs[i:j]) * avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def pct(count: float, total: int) -> float:
    return 100 * count / total if total else 0.0


def replay_em(row: dict[str, Any]) -> int:
    if "replay_em" in row:
        return int(row.get("replay_em") or 0)
    return int(row.get("em") or 0)


def direct_pred(row: dict[str, Any]) -> str:
    return row.get("direct_pred") or (row.get("direct") or {}).get("pred") or ""


def decomp_pred(row: dict[str, Any]) -> str:
    return row.get("decomp_pred") or (row.get("decomp") or {}).get("pred") or ""


def oracle_em(row: dict[str, Any]) -> int:
    return int((row.get("direct_em") or 0) or (row.get("decomp_em") or 0))


def bucket_name(similarity: float, exact: bool) -> str:
    if exact:
        return "exact"
    if similarity >= 0.5:
        return "partial>=0.5"
    return "diff<0.5"


def summarize_rows(label: str, rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    similarities: list[float] = []
    current_labels: list[int] = []
    oracle_labels: list[int] = []
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        direct = direct_pred(row)
        decomp = decomp_pred(row)
        sim = token_f1(direct, decomp)
        exact = bool(normalize(direct) and normalize(direct) == normalize(decomp))
        similarities.append(sim)
        current_labels.append(replay_em(row))
        oracle_labels.append(oracle_em(row))
        buckets[bucket_name(sim, exact)].append(row)

    direct_em = sum(int(row.get("direct_em") or 0) for row in rows)
    decomp_em = sum(int(row.get("decomp_em") or 0) for row in rows)
    current_em = sum(current_labels)
    oracle = sum(oracle_labels)

    lines = [
        f"## {label} (N={n})",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Current/replay EM | {pct(current_em, n):.2f} |",
        f"| Direct EM | {pct(direct_em, n):.2f} |",
        f"| Decompose EM | {pct(decomp_em, n):.2f} |",
        f"| Oracle EM | {pct(oracle, n):.2f} |",
        f"| Similarity -> current-correct AUROC | {auroc(similarities, current_labels):.3f} |",
        f"| Similarity -> oracle-correct AUROC | {auroc(similarities, oracle_labels):.3f} |",
        "",
        "| Consistency bucket | N | Share | Current EM | Direct EM | Decomp EM | Oracle EM | Both correct | Both wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in ("exact", "partial>=0.5", "diff<0.5"):
        bucket = buckets.get(name, [])
        m = len(bucket)
        if not m:
            continue
        cur = sum(replay_em(row) for row in bucket)
        dire = sum(int(row.get("direct_em") or 0) for row in bucket)
        dec = sum(int(row.get("decomp_em") or 0) for row in bucket)
        ora = sum(oracle_em(row) for row in bucket)
        both_correct = sum(
            int((row.get("direct_em") or 0) and (row.get("decomp_em") or 0))
            for row in bucket
        )
        both_wrong = sum(
            int(not (row.get("direct_em") or 0) and not (row.get("decomp_em") or 0))
            for row in bucket
        )
        lines.append(
            f"| {name} | {m} | {pct(m, n):.1f} | {pct(cur, m):.2f} | "
            f"{pct(dire, m):.2f} | {pct(dec, m):.2f} | {pct(ora, m):.2f} | "
            f"{pct(both_correct, m):.2f} | {pct(both_wrong, m):.2f} |"
        )

    lines.extend(
        [
            "",
            "| Escalation set | N | Current EM | Oracle EM | Oracle-current gap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for threshold in (0.25, 0.5, 0.75, 0.999999):
        low = [row for row, sim in zip(rows, similarities) if sim < threshold]
        if not low:
            continue
        cur = sum(replay_em(row) for row in low)
        ora = sum(oracle_em(row) for row in low)
        lines.append(
            f"| sim < {threshold:g} | {len(low)} | {pct(cur, len(low)):.2f} | "
            f"{pct(ora, len(low)):.2f} | {pct(ora - cur, len(low)):+.2f} |"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    labels = args.labels or [path.stem for path in args.paths]
    if len(labels) != len(args.paths):
        raise SystemExit("--labels must have the same length as paths")

    sections: list[str] = []
    all_rows: list[dict[str, Any]] = []
    for label, path in zip(labels, args.paths):
        rows = load_jsonl(path)
        sections.append(summarize_rows(label, rows))
        all_rows.extend(rows)

    if len(args.paths) > 1:
        sections.append(summarize_rows("ALL", all_rows))

    text = "\n\n".join(sections) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
