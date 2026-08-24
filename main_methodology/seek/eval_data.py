"""Compatibility helpers for SEEK evaluation JSONL records."""

from __future__ import annotations

from typing import Any


def normalize_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a legacy-compatible SEEK row without mutating ``row``.

    SEEK's original inputs use ``id`` plus a scalar ``answer`` and optional
    ``answer_aliases``. The compact public evaluation files follow the
    DeepRAG audit schema: ``qid`` plus ``answer`` as either a string or a list.
    This helper accepts both representations.
    """

    normalized = dict(row)

    if not isinstance(normalized.get("question"), str):
        raise ValueError("evaluation row requires a string 'question'")

    if normalized.get("id") is None:
        if normalized.get("qid") is None:
            raise ValueError("evaluation row requires 'id' or 'qid'")
        normalized["id"] = normalized["qid"]

    raw_answer = normalized.get("answer")
    if isinstance(raw_answer, list):
        answers = raw_answer
    elif raw_answer is None:
        raise ValueError("evaluation row requires a non-empty 'answer'")
    else:
        answers = [raw_answer]

    raw_aliases = normalized.get("answer_aliases") or []
    if not isinstance(raw_aliases, list):
        raise ValueError("'answer_aliases' must be a list when present")

    candidates = [str(value) for value in [*answers, *raw_aliases] if value is not None]
    candidates = [value for value in candidates if value]
    if not candidates:
        raise ValueError("evaluation row requires at least one non-empty answer")

    deduplicated: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduplicated.append(candidate)
            seen.add(candidate)

    normalized["answer"] = deduplicated[0]
    normalized["answer_aliases"] = deduplicated[1:]
    return normalized
