#!/usr/bin/env python3
"""Offline replay for type-aware direct/decompose verifier rules.

This script does not rerun the LLM. It reads run_seek_dual JSONL output,
replays alternative verifier choices from the saved direct/decompose answers,
and reports the resulting EM/F1. The main target is HotpotQA cases where the
default auto fallback over-selects decomposition although the direct path is
correct.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any


ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
YESNO_RE = re.compile(r"^\s*(yes|no)\s*\.?\s*$", re.I)
DATE_RE = re.compile(
    r"^\s*\d{3,4}\s*$|^\s*\d{1,2}\s+\w+\s+\d{4}\s*$|"
    r"^\s*(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b.*\d{4}",
    re.I,
)
NUMBER_RE = re.compile(
    r"^\s*[+-]?\d[\d,]*(?:\.\d+)?(?:\s*(%|percent|million|billion))?\s*$",
    re.I,
)


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
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


def answer_kind(answer: str) -> str:
    if YESNO_RE.match(answer or ""):
        return "yesno"
    if DATE_RE.match(answer or ""):
        return "date"
    if NUMBER_RE.match(answer or ""):
        return "number"
    return "text"


def question_kind(question: str) -> str:
    q = (question or "").lower()
    if re.match(r"\s*(are|is|was|were|do|does|did|can|could|has|have|had)\b", q):
        return "yesno"
    if re.search(r"\b(what year|which year|what date|date of birth|date of death|when)\b", q):
        return "date"
    if re.search(
        r"\b(how many|how much|what number|what population|population of|"
        r"what.*index|index of|what was the roud)\b",
        q,
    ):
        return "number"
    if re.search(
        r"^\s*(who|whose)\b|\b(screenwriter|singer|politician|actor|actress|"
        r"director|author|writer|composer|artist|player|coach|father|mother|"
        r"grandfather|grandmother)\b",
        q,
    ):
        return "person"
    if re.search(r"\b(nationality|country|language|genre)\b", q):
        return "attribute"
    if re.search(r"\b(which|what)\b.*\b(film|movie|album|song|book|novel|series|team|act|performance act)\b", q):
        return "entity"
    return "open"


def answer_type_mismatch(question: str, answer: str) -> bool:
    q_kind = question_kind(question)
    a_kind = answer_kind(answer)
    if q_kind == "yesno":
        return a_kind != "yesno"
    if q_kind == "date":
        return a_kind != "date"
    if q_kind == "number":
        return a_kind not in {"number", "date"}
    if q_kind in {"person", "attribute", "entity", "open"}:
        return a_kind in {"number", "date", "yesno"}
    return False


def direct_entropy(row: dict[str, Any]) -> float:
    value = (row.get("direct") or {}).get("skip_probe_ent")
    return float(value) if isinstance(value, (int, float)) else 999.0


def max_decomp_post_entropy(row: dict[str, Any]) -> float:
    vals = []
    for step in (row.get("decomp") or {}).get("steps", []):
        value = step.get("post_signal_value")
        if isinstance(value, (int, float)) and value >= 0:
            vals.append(float(value))
    return max(vals) if vals else -1.0


def answer_in_question(answer: str, question: str) -> bool:
    ans = normalize(answer)
    q = normalize(question)
    return bool(ans and len(ans) > 2 and ans in q)


def decomp_final_equals_intermediate(row: dict[str, Any]) -> bool:
    final = normalize(row.get("decomp_pred") or "")
    steps = (row.get("decomp") or {}).get("steps") or []
    if not final or len(steps) < 2:
        return False
    for step in steps[:-1]:
        ans = normalize(step.get("final_answer") or step.get("tentative") or "")
        if ans and ans == final:
            return True
    return False


def decomp_has_entity_fragment_step(row: dict[str, Any]) -> bool:
    for step in (row.get("decomp") or {}).get("steps", []):
        ans = normalize(step.get("final_answer") or "")
        sub_q = normalize(step.get("sub_q") or step.get("raw_sub_q") or "")
        if ans and len(ans) > 2 and ans in sub_q and ans != sub_q and answer_kind(ans) == "text":
            return True
    return False


def choose_typeaware(
    row: dict[str, Any],
    *,
    direct_entropy_tau: float,
    decomp_post_tau: float,
    margin: float,
) -> tuple[str, str]:
    original_choice = row.get("choice") or "decomp"
    if row.get("reason") != "auto_fallback_decomp":
        return original_choice, "keep_non_auto"

    question = row.get("question") or ""
    direct_pred = row.get("direct_pred") or ""
    decomp_pred = row.get("decomp_pred") or ""

    direct_score = 0.0
    decomp_score = 0.0
    if direct_entropy(row) < direct_entropy_tau:
        direct_score += 1.0
    if max_decomp_post_entropy(row) > decomp_post_tau:
        decomp_score -= 1.0
    if answer_type_mismatch(question, decomp_pred) and not answer_type_mismatch(question, direct_pred):
        direct_score += 2.0
    if answer_type_mismatch(question, direct_pred) and not answer_type_mismatch(question, decomp_pred):
        decomp_score += 2.0
    if decomp_has_entity_fragment_step(row):
        decomp_score -= 1.5
    if decomp_final_equals_intermediate(row):
        decomp_score -= 1.5
    if answer_in_question(decomp_pred, question) and question_kind(question) != "entity":
        decomp_score -= 1.0
    if answer_in_question(direct_pred, question) and question_kind(question) != "entity":
        direct_score -= 1.0

    if direct_score >= decomp_score + margin:
        return "direct", f"typeaware_direct({direct_score:.1f},{decomp_score:.1f})"
    return original_choice, f"typeaware_keep({direct_score:.1f},{decomp_score:.1f})"


def evaluate(rows: list[dict[str, Any]], choices: list[tuple[str, str]]) -> dict[str, Any]:
    em_sum = 0.0
    f1_sum = 0.0
    direct_count = 0
    changed = 0
    rescued = 0
    lost = 0
    reasons: Counter[str] = Counter()
    missed: Counter[tuple[str, str]] = Counter()

    for row, (choice, reason) in zip(rows, choices):
        pred = (row.get("direct_pred") if choice == "direct" else row.get("decomp_pred")) or ""
        em = int(row.get("direct_em", 0)) if choice == "direct" else int(row.get("decomp_em", 0))
        em_sum += em
        f1_sum += token_f1(pred, row.get("gold") or "")
        direct_count += choice == "direct"
        changed += choice != row.get("choice")
        rescued += (not row.get("em") and em)
        lost += (row.get("em") and not em)
        reasons[reason] += 1
        if (row.get("direct_em") or row.get("decomp_em")) and not em:
            bucket = (
                "both_correct" if row.get("direct_em") and row.get("decomp_em")
                else "direct_only" if row.get("direct_em")
                else "decomp_only"
            )
            missed[(bucket, choice)] += 1

    n = len(rows)
    return {
        "em": 100 * em_sum / n,
        "f1": 100 * f1_sum / n,
        "direct": direct_count,
        "changed": changed,
        "rescued": rescued,
        "lost": lost,
        "reasons": reasons,
        "missed": missed,
    }


def write_replay(path: Path, rows: list[dict[str, Any]], choices: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row, (choice, reason) in zip(rows, choices):
            pred = (row.get("direct_pred") if choice == "direct" else row.get("decomp_pred")) or ""
            out = dict(row)
            out["replay_choice"] = choice
            out["replay_reason"] = reason
            out["replay_pred"] = pred
            out["replay_em"] = int(row.get("direct_em", 0)) if choice == "direct" else int(row.get("decomp_em", 0))
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--direct-entropy-tau", type=float, default=0.3)
    ap.add_argument("--decomp-post-tau", type=float, default=0.8)
    ap.add_argument("--margin", type=float, default=2.5)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    base_choices = [(row.get("choice") or "decomp", row.get("reason") or "base") for row in rows]
    base = evaluate(rows, base_choices)
    print(f"input: {args.input} ({len(rows)} rows)")
    print(f"base:  EM={base['em']:.1f} F1={base['f1']:.1f} direct={base['direct']} missed={dict(base['missed'])}")

    if args.grid:
        candidates: list[tuple[str, dict[str, Any]]] = []
        for ent_tau in [0.3, 0.5, 0.7, 0.9]:
            for post_tau in [0.8, 1.0, 1.2, 1.5, 2.0]:
                for margin in [1.5, 2.0, 2.5]:
                    choices = [
                        choose_typeaware(
                            row,
                            direct_entropy_tau=ent_tau,
                            decomp_post_tau=post_tau,
                            margin=margin,
                        )
                        for row in rows
                    ]
                    label = f"ent={ent_tau} post={post_tau} margin={margin}"
                    candidates.append((label, evaluate(rows, choices)))
        for label, result in sorted(candidates, key=lambda item: item[1]["em"], reverse=True)[:20]:
            print(
                f"{label:28s} EM={result['em']:5.1f} F1={result['f1']:5.1f} "
                f"direct={result['direct']:3d} changed={result['changed']:3d} "
                f"+{result['rescued']}/-{result['lost']} missed={dict(result['missed'])}"
            )
        return

    choices = [
        choose_typeaware(
            row,
            direct_entropy_tau=args.direct_entropy_tau,
            decomp_post_tau=args.decomp_post_tau,
            margin=args.margin,
        )
        for row in rows
    ]
    result = evaluate(rows, choices)
    print(
        f"replay: EM={result['em']:.1f} F1={result['f1']:.1f} "
        f"direct={result['direct']} changed={result['changed']} "
        f"+{result['rescued']}/-{result['lost']}"
    )
    print("missed:", dict(result["missed"]))
    print("reasons:", ", ".join(f"{k}={v}" for k, v in result["reasons"].most_common()))
    if args.out:
        write_replay(args.out, rows, choices)
        print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
