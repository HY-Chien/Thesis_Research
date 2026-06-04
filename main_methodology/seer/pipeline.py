"""SEER end-to-end inference: decompose → per-step probe/gate → synthesize.

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


@dataclass
class StepTrace:
    step_id: int
    raw_sub_q: str        # with #N placeholders
    sub_q: str            # with #N resolved
    tentative: str
    max_ent: float
    mean_ent: float
    min_lp: float
    signal_value: float   # the gating signal chosen for this run
    retrieved: bool
    final_answer: str
    post_signal_value: float = -1.0  # signal AFTER retrieval (-1 if not retrieved)
    used_fallback: bool = False      # retrieval didn't help -> reverted to parametric


@dataclass
class SeerTrace:
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


def run_seer(model, tok, row: dict, *, tau: float = 0.8,
             top_k: int = 5, mode: str = "B",
             signal: str = "max_ent",
             skip_decompose_tau: float | None = None,
             post_retrieval_tau: float | None = None,
             open_retriever=None) -> SeerTrace:
    """Run SEER on one dev row.

    Gate rule:
      signal == "max_ent" or "mean_ent" — retrieve when value >= tau (high = uncertain)
      signal == "min_lp"               — retrieve when value <= tau (low  = uncertain)

    If skip_decompose_tau is set:
      Probe the original question with retrieved context first. If the answer's
      signal indicates high confidence (e.g. max_ent < skip_decompose_tau),
      return that answer directly without decomposing. This rescues single-hop
      datasets (e.g. PopQA) where the multi-hop decomposer over-decomposes.
    """
    assert mode in ("B", "C")
    assert signal in ("max_ent", "mean_ent", "min_lp")
    higher_is_uncertain = signal in ("max_ent", "mean_ent")

    question = row["question"]
    # open_retriever (shared open-domain ES index, DeepRAG setting) takes
    # precedence; otherwise fall back to the per-question paragraph pool.
    if open_retriever is not None:
        retriever = open_retriever
    else:
        paragraphs = row.get("paragraphs") or []
        retriever = PerQuestionBM25(paragraphs) if paragraphs else None

    if skip_decompose_tau is not None:
        ctx = format_context(retriever.topk(question, k=top_k)) if retriever else None
        with _answer_ctx(model, mode):
            probe = answer_with_entropy(model, tok, question, context=ctx)
        sig = probe.signal(signal)
        confident = (sig < skip_decompose_tau) if higher_is_uncertain else (sig > skip_decompose_tau)
        if confident:
            return SeerTrace(question=question, decomposition=[],
                             final_answer=probe.text,
                             skipped_decompose=True, skip_probe_ent=sig)

    decomp_text = _decompose(model, tok, question)
    parsed = _parse(decomp_text)
    trace = SeerTrace(question=question,
                      decomposition=[f"#{i}: {q}" for i, q in parsed])
    if not parsed:
        return trace

    answers: dict[int, str] = {}
    for sid, raw_sub_q in parsed:
        resolved = _substitute(raw_sub_q, answers)
        with _answer_ctx(model, mode):
            res = answer_with_entropy(model, tok, resolved)
        sig = res.signal(signal)
        retrieved = (sig >= tau) if higher_is_uncertain else (sig <= tau)
        final_ans = res.text
        post_sig = -1.0
        used_fallback = False
        if retrieved and retriever:
            ctx = format_context(retriever.topk(resolved, k=top_k))
            with _answer_ctx(model, mode):
                post = answer_with_entropy(model, tok, resolved, context=ctx)
            post_sig = post.signal(signal)
            if post_retrieval_tau is not None:
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
            min_lp=res.min_lp, signal_value=sig,
            retrieved=retrieved, final_answer=final_ans,
            post_signal_value=post_sig, used_fallback=used_fallback))

    # Final synthesis runs with the decomposer LoRA DISABLED: the adapter is
    # trained to emit decompositions, so with it on the synthesis step echoes
    # sub-answers / returns the compared value (a date) instead of selecting the
    # asked-for entity. Pure base Llama follows the synthesis instruction far
    # better (2Wiki comparison/bridge_comparison EM 37.7 -> 62.1).
    syn_ctx = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with syn_ctx:
        trace.final_answer = _synthesize(model, tok, question, trace.steps)
    return trace
