#!/usr/bin/env python3
"""Offline replay for DASC-lite as a direct/decompose verifier feature.

The replay does not rerun the LLM. It reads saved dual-path SEEK outputs and
tests whether a derivation-aware role signal can improve final answer choice.

DASC-lite role risk:
  role_risk = max_F1(direct_pred, intermediate step answers)
              - max_F1(direct_pred, final/decomp answers)

This score uses no gold label at inference time. Gold is used here only to
evaluate replayed choices.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ARTICLE_RE = re.compile(r"\b(a|an|the)\b")

DEFAULT_INPUTS = {
    "2Wiki": "main_methodology/results/2wiki_1000_verify_v2_pathaware_directctx.jsonl",
    "HotpotQA": "main_methodology/results/hotpotqa_1000_verify_v2_pathaware_directctx.jsonl",
    "MuSiQue": "main_methodology/results/musique_1000_verify_v2_pathaware_directctx.jsonl",
    "PopQA": "main_methodology/results/popqa_1000_verify_v2_pathaware_directctx.jsonl",
}


def normalize(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(pred: Any, gold: Any) -> float:
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


def current_choice(row: dict[str, Any]) -> str:
    return row.get("replay_choice") or row.get("choice") or "decomp"


def current_em(row: dict[str, Any]) -> int:
    if "replay_em" in row:
        return int(row.get("replay_em") or 0)
    return int(row.get("em") or 0)


def choose_pred(row: dict[str, Any], choice: str) -> str:
    return direct_pred(row) if choice == "direct" else decomp_pred(row)


def choose_em(row: dict[str, Any], choice: str) -> int:
    key = "direct_em" if choice == "direct" else "decomp_em"
    return int(row.get(key) or 0)


def direct_entropy(row: dict[str, Any]) -> float:
    value = (row.get("direct") or {}).get("skip_probe_ent")
    return float(value) if isinstance(value, (int, float)) and value >= 0 else math.inf


def step_answer_candidates(step: dict[str, Any]) -> list[str]:
    vals = []
    for key in ("tentative", "final_answer", "answer", "pred"):
        val = step.get(key)
        if isinstance(val, str) and normalize(val):
            vals.append(val)
    return vals


def dasc_role_features(row: dict[str, Any]) -> dict[str, Any]:
    pred = direct_pred(row)
    steps = (row.get("decomp") or {}).get("steps") or []
    if not normalize(pred) or len(steps) < 2:
        return {
            "available": False,
            "intermediate_sim": 0.0,
            "final_sim": 0.0,
            "role_risk": 0.0,
            "intermediate_match": "",
            "final_match": "",
        }

    intermediate = []
    for step in steps[:-1]:
        intermediate.extend(step_answer_candidates(step))

    final = []
    if normalize(decomp_pred(row)):
        final.append(decomp_pred(row))
    final.extend(step_answer_candidates(steps[-1]))

    def best(cands: list[str]) -> tuple[float, str]:
        best_score = 0.0
        best_text = ""
        for cand in cands:
            score = token_f1(pred, cand)
            if score > best_score:
                best_score = score
                best_text = cand
        return best_score, best_text

    inter_sim, inter_match = best(intermediate)
    final_sim, final_match = best(final)
    return {
        "available": True,
        "intermediate_sim": inter_sim,
        "final_sim": final_sim,
        "role_risk": inter_sim - final_sim,
        "intermediate_match": inter_match,
        "final_match": final_match,
    }


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
        base = current_em(row)
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


def dasc_guard(ent_tau: float, high_role: float) -> Callable[[dict[str, Any]], str]:
    """Only veto current direct choices that look like intermediate answers."""

    def choose(row: dict[str, Any]) -> str:
        if current_choice(row) != "direct":
            return current_choice(row)
        feat = dasc_role_features(row)
        if (
            feat["available"]
            and direct_entropy(row) <= ent_tau
            and feat["role_risk"] >= high_role
            and normalize(decomp_pred(row))
        ):
            return "decomp"
        return current_choice(row)

    return choose


def dasc_auto(
    ent_tau: float,
    low_role: float,
    high_role: float,
    only_auto_fallback: bool = True,
) -> Callable[[dict[str, Any]], str]:
    """Symmetric guard for final verifier replay.

    High role risk means the direct answer aligns with an intermediate answer,
    so prefer decompose. Low role risk plus low direct entropy means the direct
    answer does not look like intermediate leakage, so direct can be trusted in
    auto-fallback cases.
    """

    def choose(row: dict[str, Any]) -> str:
        feat = dasc_role_features(row)
        if not feat["available"] or direct_entropy(row) > ent_tau:
            return current_choice(row)

        if feat["role_risk"] >= high_role and normalize(decomp_pred(row)):
            return "decomp"

        can_rescue_direct = (
            current_choice(row) == "decomp"
            and (not only_auto_fallback or row.get("reason") == "auto_fallback_decomp")
        )
        if can_rescue_direct and feat["role_risk"] <= low_role and normalize(direct_pred(row)):
            return "direct"

        return current_choice(row)

    return choose


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


def render_grid(
    labels: list[str],
    rows_by_label: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> str:
    all_rows = [row for label in labels for row in rows_by_label[label]]
    candidates = []
    for ent_tau in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]:
        for low_role in [-0.25, 0.0, 0.25, 0.5]:
            for high_role in [0.5, 0.75, 0.9, 1.0]:
                if low_role > high_role:
                    continue
                chooser = dasc_auto(ent_tau, low_role, high_role)
                result = evaluate(all_rows, chooser)
                candidates.append((ent_tau, low_role, high_role, result))
    candidates.sort(key=lambda item: (item[3].em, item[3].f1), reverse=True)

    lines = [
        "| Rank | ent_tau | low_role | high_role | EM | F1 | Direct % | Changed | Rescue/Lost |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, (ent_tau, low_role, high_role, result) in enumerate(candidates[:top_k], 1):
        lines.append(
            f"| {rank} | {ent_tau:.2f} | {low_role:.2f} | {high_role:.2f} | "
            f"{result.em:.2f} | {result.f1:.2f} | {result.direct_rate:.1f} | "
            f"{result.changed} | {result.rescued}/{result.lost} |"
        )
    return "\n".join(lines)


def render_feature_summary(labels: list[str], rows_by_label: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| Dataset | DASC available | current direct | high-role direct | high-role direct wrong |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in labels:
        rows = rows_by_label[label]
        available = [row for row in rows if dasc_role_features(row)["available"]]
        direct_rows = [row for row in rows if current_choice(row) == "direct"]
        high_direct = [
            row
            for row in direct_rows
            if dasc_role_features(row)["available"]
            and direct_entropy(row) <= 0.3
            and dasc_role_features(row)["role_risk"] >= 0.75
        ]
        lines.append(
            f"| {label} | {len(available)} | {len(direct_rows)} | "
            f"{len(high_direct)} | {sum(1 - current_em(row) for row in high_direct)} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("main_methodology/results/dasc_verifier_replay.md"))
    parser.add_argument("--grid-top-k", type=int, default=15)
    parser.add_argument("--input", action="append", default=[], metavar="LABEL:PATH")
    args = parser.parse_args()

    inputs = DEFAULT_INPUTS.copy()
    for spec in args.input:
        label, path = spec.split(":", 1)
        inputs[label] = path

    labels = list(inputs)
    rows_by_label = {label: load_jsonl(Path(path)) for label, path in inputs.items()}

    policies: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("current", current_policy),
        ("dasc_guard_e0.3_r0.75", dasc_guard(0.3, 0.75)),
        ("dasc_guard_e0.5_r0.75", dasc_guard(0.5, 0.75)),
        ("dasc_auto_e0.3_low0_high0.75", dasc_auto(0.3, 0.0, 0.75)),
        ("dasc_auto_e0.5_low0_high0.75", dasc_auto(0.5, 0.0, 0.75)),
        ("dasc_auto_e0.8_low0.25_high0.75", dasc_auto(0.8, 0.25, 0.75)),
    ]

    text = "\n\n".join(
        [
            "# DASC-lite Verifier Replay",
            "This replay uses saved direct/decomposition outputs only; it does not rerun the LLM.",
            "DASC-lite uses no gold labels for making a choice. Gold labels are used only for evaluation.",
            "## Feature Coverage",
            render_feature_summary(labels, rows_by_label),
            "## Fixed Policies",
            render_table(labels, rows_by_label, policies),
            "## Exploratory Full-Dev Grid",
            "The grid is tuned on the same dev traces, so it should be read as an exploratory upper bound rather than a final test result.",
            render_grid(labels, rows_by_label, args.grid_top_k),
        ]
    )
    text += "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
