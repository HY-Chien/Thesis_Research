"""Training-free verifier for choosing direct vs. decomposed SEEK answers."""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass


_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")
_YESNO_Q_RE = re.compile(
    r"^\s*(are|is|was|were|do|does|did|can|could|would|should|has|have|had)\b",
    re.I,
)
_YESNO_A_RE = re.compile(r"^\s*(yes|no)\s*\.?\s*$", re.I)
_CMP_Q_RE = re.compile(
    r"\b(first|earlier|later|older|younger|longer|earliest|latest|"
    r"more recently|newer|born first|released first|came out first)\b",
    re.I,
)
_VALUE_A_RE = re.compile(
    r"^\s*\d{3,4}\s*$"
    r"|^\s*\d{1,2}\s+\w+\s+\d{4}\s*$"
    r"|^\s*(january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b.*\d{4}",
    re.I,
)
_UNRESOLVED_REF_RE = re.compile(r"(?:^|\s)#\d+\b")


@dataclass
class Verification:
    choice: str
    reason: str
    direct_support: float = 0.0
    decomp_support: float = 0.0


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(pred: str, golds: list[str]) -> float:
    pred_toks = normalize(pred).split()
    best = 0.0
    for gold in golds:
        gold_toks = normalize(gold).split()
        if not pred_toks or not gold_toks:
            best = max(best, float(pred_toks == gold_toks))
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        n_common = sum(common.values())
        if not n_common:
            continue
        precision = n_common / len(pred_toks)
        recall = n_common / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def exact_match(pred: str, golds: list[str]) -> bool:
    pred_norm = normalize(pred)
    return any(pred_norm == normalize(gold) for gold in golds)


def _is_yesno_question(question: str) -> bool:
    return bool(_YESNO_Q_RE.search(question or ""))


def _is_yesno_answer(answer: str) -> bool:
    return bool(_YESNO_A_RE.match(answer or ""))


def _is_comparison_question(question: str) -> bool:
    return bool(_CMP_Q_RE.search(question or ""))


def _is_value_answer(answer: str) -> bool:
    return bool(_VALUE_A_RE.match(answer or ""))


def _steps(record: dict) -> list[dict]:
    return record.get("steps") or []


def _final(record: dict) -> str:
    return record.get("pred") or record.get("final_answer") or ""


def _decomp_failed(record: dict) -> bool:
    return not _final(record) or not _steps(record)


def _looks_broken(record: dict) -> bool:
    pred = _final(record)
    if _UNRESOLVED_REF_RE.search(pred):
        return True
    for step in _steps(record):
        ans = step.get("final_answer") or ""
        sub_q = step.get("sub_q") or ""
        if _UNRESOLVED_REF_RE.search(ans):
            return True
        if normalize(ans) and normalize(ans) == normalize(sub_q):
            return True
    return False


def support_score(answer: str, evidence_text: str) -> float:
    """Cheap lexical support score in [0, 2].

    2.0 means normalized exact phrase appears. Otherwise the score is token
    recall over non-trivial answer tokens. This is intentionally simple and
    deterministic; it is a first-pass verifier, not an NLI model.
    """
    ans = normalize(answer)
    ev = normalize(evidence_text)
    if not ans or not ev:
        return 0.0
    if ans in ev:
        return 2.0
    toks = [t for t in ans.split() if len(t) > 2]
    if not toks:
        toks = ans.split()
    if not toks:
        return 0.0
    ev_toks = set(ev.split())
    return sum(1 for tok in toks if tok in ev_toks) / len(toks)


def chain_support_score(
    record: dict,
    step_evidences: list[str],
    fallback_evidence: str = "",
) -> float:
    """Score whether a decomposed reasoning path is supported step by step.

    Final-answer lexical support can be misleading for multi-hop QA: an entity
    can appear in the retrieved passages without the path being valid. For a
    decomposed answer, reward support for intermediate step answers as well.
    The score stays in the same rough [0, 2] range as support_score.
    """
    final_score = support_score(_final(record), fallback_evidence)
    step_scores = []
    for idx, step in enumerate(_steps(record)):
        answer = step.get("final_answer") or step.get("tentative") or ""
        evidence = step_evidences[idx] if idx < len(step_evidences) else ""
        if not normalize(answer) or normalize(answer) == normalize(step.get("sub_q") or ""):
            continue
        step_scores.append(support_score(answer, evidence))
    if not step_scores:
        return final_score
    mean_score = sum(step_scores) / len(step_scores)
    min_score = min(step_scores)
    chain_score = 0.5 * mean_score + 0.5 * min_score
    return max(final_score, chain_score)


def _direct_entropy(record: dict) -> float | None:
    value = record.get("skip_probe_ent", -1.0)
    return float(value) if isinstance(value, (int, float)) and value >= 0 else None


def _decomp_post_entropy(record: dict) -> float | None:
    vals = []
    for step in _steps(record):
        value = step.get("post_signal_value", -1.0)
        if isinstance(value, (int, float)) and value >= 0:
            vals.append(float(value))
    return max(vals) if vals else None


def _is_uncertain_signal(value: float, tau: float, *, higher_is_uncertain: bool) -> bool:
    return value > tau if higher_is_uncertain else value < tau


def _is_confident_signal(value: float, tau: float, *, higher_is_uncertain: bool) -> bool:
    return value < tau if higher_is_uncertain else value > tau


def _auto_fallback(
    direct: dict,
    decomp: dict,
    direct_support: float,
    decomp_support: float,
    *,
    direct_entropy_tau: float,
    decomp_post_entropy_tau: float,
    higher_is_uncertain: bool = True,
) -> Verification:
    direct_ent = _direct_entropy(direct)
    decomp_post = _decomp_post_entropy(decomp)
    decomp_unstable = (
        _decomp_failed(decomp)
        or _looks_broken(decomp)
        or (decomp_post is not None and _is_uncertain_signal(
            decomp_post, decomp_post_entropy_tau,
            higher_is_uncertain=higher_is_uncertain))
    )
    if (direct_ent is not None and _is_confident_signal(
            direct_ent, direct_entropy_tau,
            higher_is_uncertain=higher_is_uncertain) and decomp_unstable):
        return Verification("direct", "auto_confident_direct_unstable_decomp",
                            direct_support, decomp_support)
    return Verification("decomp", "auto_fallback_decomp",
                        direct_support, decomp_support)


def choose_answer(
    question: str,
    direct: dict,
    decomp: dict,
    *,
    direct_evidence: str = "",
    decomp_evidence: str = "",
    decomp_step_evidences: list[str] | None = None,
    evidence_mode: str = "final",
    support_margin: float = 0.35,
    fallback: str = "auto",
    direct_entropy_tau: float = 0.3,
    decomp_post_entropy_tau: float = 0.8,
    higher_is_uncertain: bool = True,
) -> Verification:
    """Choose between direct and decomposition outputs.

    The rule order is conservative: obvious structural cases first, lexical
    evidence support next, and finally a configurable default policy.
    """
    direct_answer = _final(direct)
    decomp_answer = _final(decomp)
    direct_norm = normalize(direct_answer)
    decomp_norm = normalize(decomp_answer)

    if direct_norm and direct_norm == decomp_norm:
        return Verification("decomp", "same_answer")
    if _decomp_failed(decomp):
        return Verification("direct", "decomp_failed")
    if not direct_norm:
        return Verification("decomp", "direct_empty")

    if _is_yesno_question(question):
        direct_yesno = _is_yesno_answer(direct_answer)
        decomp_yesno = _is_yesno_answer(decomp_answer)
        if direct_yesno and not decomp_yesno:
            return Verification("direct", "yesno_type_direct")
        if decomp_yesno and not direct_yesno:
            return Verification("decomp", "yesno_type_decomp")

    if _is_comparison_question(question):
        direct_value = _is_value_answer(direct_answer)
        decomp_value = _is_value_answer(decomp_answer)
        if direct_value and not decomp_value:
            return Verification("decomp", "comparison_value_direct")
        if decomp_value and not direct_value:
            return Verification("direct", "comparison_value_decomp")

    direct_broken = _looks_broken(direct)
    decomp_broken = _looks_broken(decomp)
    if decomp_broken and not direct_broken:
        return Verification("direct", "decomp_broken")
    if direct_broken and not decomp_broken:
        return Verification("decomp", "direct_broken")

    direct_support = support_score(direct_answer, direct_evidence)
    if evidence_mode == "chain":
        decomp_support = chain_support_score(
            decomp, decomp_step_evidences or [], decomp_evidence
        )
    else:
        decomp_support = support_score(decomp_answer, decomp_evidence)
    if direct_support or decomp_support:
        if direct_support >= decomp_support + support_margin:
            return Verification("direct", "evidence_support_direct",
                                direct_support, decomp_support)
        if decomp_support >= direct_support + support_margin:
            return Verification("decomp", "evidence_support_decomp",
                                direct_support, decomp_support)

    if fallback == "auto":
        return _auto_fallback(
            direct, decomp, direct_support, decomp_support,
            direct_entropy_tau=direct_entropy_tau,
            decomp_post_entropy_tau=decomp_post_entropy_tau,
            higher_is_uncertain=higher_is_uncertain,
        )
    if fallback not in {"direct", "decomp"}:
        fallback = "decomp"
    return Verification(fallback, f"fallback_{fallback}",
                        direct_support, decomp_support)
