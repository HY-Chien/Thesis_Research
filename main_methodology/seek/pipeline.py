"""SEEK end-to-end inference: decompose → per-step probe/gate → synthesize.

Modes:
  B  adapter-toggle    decomposer LoRA on, answer with LoRA disabled (pure base)
  C  adapter-always    decomposer LoRA on for both decompose AND answer
"""
from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .answerer import answer_with_entropy, ANSWER_INSTRUCTION
from .retriever import PerQuestionBM25, format_context

DECOMP_INSTRUCTION = (
    "Decompose the following multi-hop question into an ordered list of "
    "atomic sub-questions. Each sub-question must be answerable in a single "
    "retrieval/reasoning step. When a sub-question depends on the answer of "
    "an earlier one, refer to that step with a #N placeholder (e.g. #1). "
    "Output one sub-question per line, prefixed with #<index>:"
)

STEP_RE = re.compile(r"^#(\d+):\s*(.*)$")
REF_RE = re.compile(r"#(\d+)")

HIGHER_IS_UNCERTAIN_SIGNALS = {"max_ent", "mean_ent", "targ_unc"}
LOWER_IS_UNCERTAIN_SIGNALS = {
    "min_lp", "maqa_margin", "mean_margin", "max_margin", "targ_logit_gap"}
SIGNAL_NAMES = HIGHER_IS_UNCERTAIN_SIGNALS | LOWER_IS_UNCERTAIN_SIGNALS


def _higher_is_uncertain_signal(signal: str) -> bool:
    if signal not in SIGNAL_NAMES:
        raise ValueError(f"unknown confidence signal: {signal}")
    return signal in HIGHER_IS_UNCERTAIN_SIGNALS


@dataclass
class StepTrace:
    step_id: int
    raw_sub_q: str        # with #N placeholders
    sub_q: str            # with #N resolved
    tentative: str
    max_ent: float
    mean_ent: float
    min_lp: float
    maqa_margin: float
    mean_margin: float
    max_margin: float
    targ_logit_gap: float
    targ_unc: float
    signal_value: float   # the gating signal chosen for this run
    retrieved: bool
    final_answer: str
    post_signal_value: float = -1.0  # signal AFTER retrieval (-1 if not retrieved)
    retrieval_signal_drop: float = 0.0  # positive means retrieval made the signal more confident
    used_fallback: bool = False      # retrieval didn't help -> reverted to parametric


@dataclass
class SeekTrace:
    question: str
    decomposition: list[str]
    steps: list[StepTrace] = field(default_factory=list)
    final_answer: str = ""
    skipped_decompose: bool = False
    skip_probe_ent: float = -1.0


def _answer_ctx(model, mode: str):
    """Return a FRESH context manager each call (CMs can't be reused)."""
    if mode == "B" and hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


@torch.no_grad()
def _decompose(model, tok, question: str, max_new_tokens: int = 256) -> str:
    msgs = [{"role": "user", "content": DECOMP_INSTRUCTION + "\n" + question}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def _parse(text: str) -> list[tuple[int, str]]:
    out = []
    for line in text.splitlines():
        m = STEP_RE.match(line.strip())
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def _substitute(sub_q: str, answers: dict[int, str]) -> str:
    return REF_RE.sub(lambda m: answers.get(int(m.group(1)), m.group(0)), sub_q)


def _hit_key(hit: dict) -> tuple[str, str]:
    return (str(hit.get("id") or ""), str(hit.get("title") or ""))


def _merge_hits(primary: list[dict], extra: list[dict] | None) -> list[dict]:
    """Append supplemental retrieval hits without duplicating passages."""
    out = []
    seen = set()
    for hit in list(primary or []) + list(extra or []):
        key = _hit_key(hit)
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
    return out


SYNTHESIZE_INSTRUCTION = (
    "You are given a multi-hop question and the intermediate sub-answers. "
    "Use the sub-answers to derive the final answer.\n"
    "Rules:\n"
    "- If the question compares two entities (e.g. 'which film ... earlier/later/born first', "
    "'which X ... taller/longer'), compare the relevant sub-answers and return the matching "
    "ENTITY from the original question, not the value being compared.\n"
    "- If the question asks yes/no (e.g. 'do X and Y share ...', 'are X and Y from the same ...'), "
    "answer 'Yes' or 'No' based on whether the sub-answers agree.\n"
    "- Otherwise, return the answer phrase from the final sub-answer.\n"
    "- KEEP IT TERSE. Return only the minimal answer span. Do NOT add party/title "
    "prefixes, descriptive modifiers, or alternative options.\n"
    "  e.g. 'politician', not 'Conservative politician'\n"
    "  e.g. 'politician', not 'politician, soldier' (pick the single best match)\n"
    "  e.g. 'tuberculosis', not 'extrapulmonary tuberculosis'\n"
    "- If the question asks 'when'/'what year', return ONLY the year "
    "(e.g. '1999', not 'November 1999'; '1983', not 'March 7, 1983').\n"
    "- Output the answer phrase ONLY — no explanation, no preamble.\n\n"
    "Example 1 (comparison → return entity):\n"
    "Sub-answers:\n"
    "- Film A >> director  ->  Alice Smith\n"
    "- Film B >> director  ->  Bob Jones\n"
    "- Alice Smith >> date of birth  ->  1920\n"
    "- Bob Jones >> date of birth  ->  1945\n"
    "Question: Which film has the director born later, Film A or Film B?\n"
    "Answer: Film B\n\n"
    "Example 2 (yes/no):\n"
    "Sub-answers:\n"
    "- Person X >> country of citizenship  ->  France\n"
    "- Person Y >> country of citizenship  ->  France\n"
    "Question: Are Person X and Person Y from the same country?\n"
    "Answer: Yes\n\n"
    "Example 3 (single fact → strip modifiers):\n"
    "Sub-answers:\n"
    "- George Rankin >> occupation  ->  politician and soldier\n"
    "Question: What is George Rankin's occupation?\n"
    "Answer: politician\n\n"
    "Example 4 ('when' question → year only):\n"
    "Sub-answers:\n"
    "- The album >> release date  ->  March 14, 2000\n"
    "Question: In what year was the album released?\n"
    "Answer: 2000\n"
)


_CMP_Q_RE = re.compile(
    r"\b(first|earlier|later|older|younger|longer|earliest|latest|more recently|newer)\b", re.I)
_VALUE_ANS_RE = re.compile(
    r"^\s*\d{3,4}\s*$"                      # bare year: 1932
    r"|^\s*\d{1,2}\s+\w+\s+\d{4}\s*$"       # 8 December 1941
    r"|^\s*(january|february|march|april|may|june|july|august"
    r"|september|october|november|december)\b.*\d{4}", re.I)
_YEAR_RE = re.compile(r"\b(\d{3,4})\b")
_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH_FIRST_RE = re.compile(
    rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})?(?:st|nd|rd|th)?[,]?\s+(\d{{3,4}})\b",
    re.I,
)
_DAY_FIRST_RE = re.compile(
    rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\.?[,]?\s+(\d{{3,4}})\b",
    re.I,
)
_BAD_CANDIDATE_RE = re.compile(
    r"^\s*(who|what|which|when|where|how|why|is|are|was|were|did|does|do)\b",
    re.I,
)


def _norm_span(text: str | None) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _parse_date_key(text: str | None) -> tuple[int, int, int] | None:
    """Return a comparable (year, month, day) from a short answer span."""
    if not text:
        return None
    m = _MONTH_FIRST_RE.search(text)
    if m:
        month = _MONTHS[m.group(1).lower().rstrip(".")]
        day = int(m.group(2) or 1)
        return int(m.group(3)), month, day
    m = _DAY_FIRST_RE.search(text)
    if m:
        month = _MONTHS[m.group(2).lower().rstrip(".")]
        return int(m.group(3)), month, int(m.group(1))
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(1)), 1, 1
    return None


def _date_compare_direction(question: str) -> str | None:
    """For date comparisons, return 'min' for earlier/older, 'max' for later."""
    q = (question or "").lower()
    # Avoid ambiguous auxiliary-start questions such as "Was A or B born first?"
    # where noisy intermediate dates can overwrite an already-correct synthesis.
    if re.match(r"\s*(is|are|was|were|do|does|did|has|have|had)\b", q):
        return None
    if re.search(r"\b(younger|later|latest|newer|more recently|born later)\b", q):
        return "max"
    if re.search(r"\b(older|earlier|earliest|first|born first|came first)\b", q):
        return "min"
    return None


def _candidate_from_step(step) -> str:
    sub_q = getattr(step, "sub_q", "") or ""
    if ">>" in sub_q:
        cand = sub_q.split(">>", 1)[0].strip()
    else:
        cand = sub_q.strip().rstrip("?")
        cand = re.sub(r"^\s*(?:when|what year)\s+(?:did|was|were|is)\s+",
                      "", cand, flags=re.I)
        cand = re.sub(
            r"\s+\b(?:take place|happen|occur|start|begin|release|"
            r"released|published|founded|established|born|opened|open)\b.*$",
            "",
            cand,
            flags=re.I,
        ).strip()
    cand = cand.strip(" .,:;\"'")
    if not cand or "#" in cand or _BAD_CANDIDATE_RE.search(cand):
        return ""
    return cand


def _comparison_candidate_for_step(step, steps: list[StepTrace], original_q: str) -> str:
    """Return the step candidate only when it appears in the original question.

    This intentionally avoids cross-step remapping such as director -> film.
    That remapping can help 2Wiki-style questions, but it can also overwrite a
    correct synthesis when an intermediate answer is wrong. The conservative
    guard here targets HotpotQA-style comparisons where the compared entities
    themselves are the date-bearing steps.
    """
    del steps  # kept in the signature for future path-aware variants
    cand = _candidate_from_step(step)
    cand_norm = _norm_span(cand)
    if cand_norm and cand_norm in _norm_span(original_q):
        return cand
    return ""


def _rescue_date_comparison_answer(
    original_q: str,
    answer: str,
    steps: list[StepTrace],
) -> str:
    """Deterministically fix date-comparison synthesis mistakes.

    If the model has the needed step values but picks the wrong entity (e.g.
    Annie Morton instead of Terry Richardson for an older/younger question),
    compare the parsed dates and return the candidate attached to the winning
    step. Only fires when at least two step answers contain parseable dates.
    """
    direction = _date_compare_direction(original_q)
    if direction is None:
        return answer

    comparable = []
    for step in steps:
        cand = _comparison_candidate_for_step(step, steps, original_q)
        date_key = _parse_date_key(getattr(step, "final_answer", ""))
        if cand and date_key is not None:
            comparable.append((date_key, cand))
    if len(comparable) < 2:
        return answer

    comparable.sort(key=lambda x: x[0])
    winner = comparable[0][1] if direction == "min" else comparable[-1][1]
    if not winner or _norm_span(winner) == _norm_span(answer):
        return answer
    return winner


def _rescue_value_answer(original_q: str, answer: str, steps: list[StepTrace]) -> str:
    """Comparison/date synthesis guards.

    First map a returned comparison value (e.g. '1932') back to the entity that
    produced it. Then fix cases where the model had the date values but selected
    the wrong entity for older/younger/earlier/later questions.
    """
    rescued = answer
    if _CMP_Q_RE.search(original_q) and _VALUE_ANS_RE.match(answer or ""):
        hits = [s for s in steps if answer.lower() in (s.final_answer or "").lower()]
        if len(hits) == 1 and ">>" in hits[0].sub_q:
            rescued = hits[0].sub_q.split(">>")[0].strip()
    return _rescue_date_comparison_answer(original_q, rescued, steps)


@torch.no_grad()
def _synthesize(model, tok, original_q: str,
                steps: list[StepTrace], max_new_tokens: int = 48) -> str:
    """Final pass — give the model the original question + all sub-answers."""
    sub = "\n".join(f"- {s.sub_q}  ->  {s.final_answer}" for s in steps)
    body = f"Sub-answers:\n{sub}\n\nQuestion: {original_q}\nAnswer:"
    msgs = [{"role": "user", "content": f"{SYNTHESIZE_INSTRUCTION}\n{body}"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def run_seek(model, tok, row: dict, *, tau: float = 0.8,
             top_k: int = 5, mode: str = "B",
             signal: str = "max_ent",
             targ_prefix_k: int = 8,
             targ_beta: float = 1.0,
             skip_decompose_tau: float | None = None,
             post_retrieval_tau: float | None = None,
             post_retrieval_drop_tau: float | None = None,
             open_retriever=None,
             title_mode: str = "pathaware",
             direct_context_union: bool = False) -> SeekTrace:
    """Run SEEK on one dev row.

    Gate rule:
      max_ent / mean_ent / targ_unc retrieve when value >= tau (high = uncertain)
      min_lp / maqa_margin / mean_margin / max_margin / targ_logit_gap
        retrieve when value <= tau (low = uncertain)

    If skip_decompose_tau is set:
      Probe the original question with retrieved context first. If the answer's
      signal indicates high confidence (e.g. max_ent < skip_decompose_tau),
      return that answer directly without decomposing. This rescues single-hop
      datasets (e.g. PopQA) where the multi-hop decomposer over-decomposes.

    If direct_context_union is true:
      Each retrieved decomposition step receives its sub-question top-k plus
      the original-question top-k as supplemental context. This targets
      HotpotQA cases where the decomposed query drops disambiguating clues.
    """
    assert mode in ("B", "C")
    assert title_mode in ("off", "global", "pathaware")
    higher_is_uncertain = _higher_is_uncertain_signal(signal)
    # title-field matching per query type (ablation control):
    #   off       -> never match title
    #   global    -> always match title (DeepRAG / IRCoT default)
    #   pathaware -> match title only for short decompose sub-queries, NOT for
    #                the long original question (avoids stray title hits)
    title_direct = (title_mode == "global")          # original-question retrieval
    title_subq = (title_mode != "off")               # decompose sub-q retrieval

    question = row["question"]
    # open_retriever (shared open-domain ES index, DeepRAG setting) takes
    # precedence; otherwise fall back to the per-question paragraph pool.
    if open_retriever is not None:
        retriever = open_retriever
    else:
        paragraphs = row.get("paragraphs") or []
        retriever = PerQuestionBM25(paragraphs) if paragraphs else None

    if skip_decompose_tau is not None:
        # single-step / direct path retrieves with the long ORIGINAL question.
        ctx = format_context(retriever.topk(question, k=top_k, title_field=title_direct)) if retriever else None
        with _answer_ctx(model, mode):
            probe = answer_with_entropy(
                model, tok, question, context=ctx,
                targ_prefix_k=targ_prefix_k, targ_beta=targ_beta)
        sig = probe.signal(signal)
        confident = (sig < skip_decompose_tau) if higher_is_uncertain else (sig > skip_decompose_tau)
        if confident:
            return SeekTrace(question=question, decomposition=[],
                             final_answer=probe.text,
                             skipped_decompose=True, skip_probe_ent=sig)

    direct_context_hits = None
    if direct_context_union and retriever:
        # Supplemental original-question evidence keeps HotpotQA-specific
        # disambiguating clues that short sub-questions can drop.
        direct_context_hits = retriever.topk(
            question, k=top_k, title_field=title_direct)

    decomp_text = _decompose(model, tok, question)
    parsed = _parse(decomp_text)
    trace = SeekTrace(question=question,
                      decomposition=[f"#{i}: {q}" for i, q in parsed])
    if not parsed:
        return trace

    answers: dict[int, str] = {}
    for sid, raw_sub_q in parsed:
        resolved = _substitute(raw_sub_q, answers)
        with _answer_ctx(model, mode):
            res = answer_with_entropy(
                model, tok, resolved,
                targ_prefix_k=targ_prefix_k, targ_beta=targ_beta)
        sig = res.signal(signal)
        retrieved = (sig >= tau) if higher_is_uncertain else (sig <= tau)
        final_ans = res.text
        post_sig = -1.0
        signal_drop = 0.0
        used_fallback = False
        if retrieved and retriever:
            # decompose sub-query ≈ an entity/title.
            subq_hits = retriever.topk(resolved, k=top_k, title_field=title_subq)
            hits = _merge_hits(subq_hits, direct_context_hits)
            ctx = format_context(hits)
            with _answer_ctx(model, mode):
                post = answer_with_entropy(
                    model, tok, resolved, context=ctx,
                    targ_prefix_k=targ_prefix_k, targ_beta=targ_beta)
            post_sig = post.signal(signal)
            signal_drop = sig - post_sig if higher_is_uncertain else post_sig - sig
            if post_retrieval_drop_tau is not None:
                # Evidence-conditioned uncertainty reduction: accept retrieved
                # context only if it makes the answer sufficiently more certain.
                helped = signal_drop >= post_retrieval_drop_tau
                if helped:
                    final_ans = post.text
                else:
                    final_ans = res.text
                    used_fallback = True
            elif post_retrieval_tau is not None:
                # Second probe: did the retrieved context actually resolve the
                # uncertainty? If the post-retrieval answer is STILL uncertain
                # (signal past the threshold the wrong way), the retrieval likely
                # missed the gold passage — revert to the model's parametric
                # (pre-retrieval) answer rather than trust unhelpful context.
                helped = ((post_sig < post_retrieval_tau) if higher_is_uncertain
                          else (post_sig > post_retrieval_tau))
                if helped:
                    final_ans = post.text
                else:
                    final_ans = res.text
                    used_fallback = True
            else:
                final_ans = post.text
        answers[sid] = final_ans
        trace.steps.append(StepTrace(
            step_id=sid, raw_sub_q=raw_sub_q, sub_q=resolved,
            tentative=res.text, max_ent=res.max_ent, mean_ent=res.mean_ent,
            min_lp=res.min_lp, maqa_margin=res.maqa_margin,
            mean_margin=res.mean_margin, max_margin=res.max_margin,
            targ_logit_gap=res.targ_logit_gap, targ_unc=res.targ_unc,
            signal_value=sig, retrieved=retrieved, final_answer=final_ans,
            post_signal_value=post_sig, retrieval_signal_drop=signal_drop,
            used_fallback=used_fallback))

    # Final synthesis runs with the decomposer LoRA DISABLED: the adapter is
    # trained to emit decompositions, so with it on the synthesis step echoes
    # sub-answers / returns the compared value (a date) instead of selecting the
    # asked-for entity. Pure base Llama follows the synthesis instruction far
    # better (2Wiki comparison/bridge_comparison EM 37.7 -> 62.1).
    syn_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with syn_ctx:
        trace.final_answer = _rescue_value_answer(
            question, _synthesize(model, tok, question, trace.steps), trace.steps)
    return trace
