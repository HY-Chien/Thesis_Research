"""Prompt builder + output parser for OSS sub-question decomposition.

This is the redesigned prompt (Option A): the model is asked to emit, per
sub-question, not only the question text and dependency edges but also the
ANSWER and the supporting PARAGRAPH index — closing the gap noted in the
oss-120b calibration report (generated_decomposition lacked paragraph_support_idx).

Works for any dataset whose rows carry a MuSiQue-style `paragraphs` list:
each paragraph is {idx, title, paragraph_text, is_supporting}.
"""
from __future__ import annotations

import json
import re

SYSTEM_MSG = (
    "You are an expert at decomposing complex multi-hop questions into atomic "
    "sub-questions and grounding each answer in supporting evidence."
)

_USER_TEMPLATE = """Decompose the QUESTION below into an ordered list of atomic sub-questions.

Rules:
- Each sub-question asks exactly ONE fact and is self-contained except for #-references.
- Reference an earlier sub-question's answer with its id token (e.g. #1) inside the text.
- For each sub-question provide these fields:
  - "id": "#1", "#2", ... numbered in order.
  - "sub_question": the atomic question text.
  - "depends_on": list of earlier ids it references, e.g. ["#1"]. Empty list if none.
  - "answer": the answer to the sub-question, taken from the PARAGRAPHS.
  - "paragraph_support_idx": the integer idx of the ONE paragraph that supports the
    answer, or null if no paragraph supports it.

Output ONLY a JSON object, no prose, in exactly this schema:
{{"decomposition": [{{"id": "#1", "sub_question": "...", "depends_on": [], "answer": "...", "paragraph_support_idx": 0}}]}}

QUESTION:
{question}

PARAGRAPHS:
{paragraphs}
"""


def build_prompt(row: dict) -> tuple[str, str]:
    """Return (system_msg, user_msg) for one dataset row."""
    para_lines = []
    for p in row["paragraphs"]:
        title = p.get("title", "")
        text = p.get("paragraph_text", "")
        para_lines.append(f"[{p['idx']}] ({title}) {text}")
    user_msg = _USER_TEMPLATE.format(
        question=row["question"],
        paragraphs="\n".join(para_lines),
    )
    return SYSTEM_MSG, user_msg


def _extract_json(text: str) -> dict:
    """Pull the first balanced JSON object containing a 'decomposition' key."""
    # Strip markdown fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start = text.find("{")
    if start == -1:
        raise ValueError("no '{' in output")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON braces")


def parse_output(raw_text: str, n_paragraphs: int) -> tuple[list[dict] | None, str | None]:
    """Parse model output into a normalized generated_decomposition list.

    Returns (decomposition, error). Exactly one is non-None.
    Each normalized step:
        {step_id, id, sub_question, depends_on, answer,
         paragraph_support_idx, source}
    """
    try:
        obj = _extract_json(raw_text)
    except Exception as e:  # noqa: BLE001
        return None, f"json_parse_error: {e}"

    steps = obj.get("decomposition")
    if not isinstance(steps, list) or not steps:
        return None, "missing_or_empty_decomposition"

    norm: list[dict] = []
    for i, s in enumerate(steps, start=1):
        if not isinstance(s, dict):
            return None, f"step {i} not an object"
        sub_q = s.get("sub_question")
        if not sub_q:
            return None, f"step {i} missing sub_question"

        # Normalize depends_on to a list of ints (#1 -> 1)
        deps_raw = s.get("depends_on") or []
        deps: list[int] = []
        for d in deps_raw:
            if isinstance(d, str):
                d = d.lstrip("#")
            try:
                deps.append(int(d))
            except (TypeError, ValueError):
                return None, f"step {i} unparseable dep {d!r}"

        # Validate paragraph_support_idx
        psi = s.get("paragraph_support_idx", None)
        if psi is not None:
            if not isinstance(psi, int) or not (0 <= psi < n_paragraphs):
                return None, f"step {i} paragraph_support_idx out of range: {psi!r}"

        norm.append({
            "step_id": i,
            "id": f"#{i}",
            "sub_question": sub_q,
            "depends_on": deps,
            "answer": s.get("answer"),
            "paragraph_support_idx": psi,
            "source": "oss_generated",
        })
    return norm, None
