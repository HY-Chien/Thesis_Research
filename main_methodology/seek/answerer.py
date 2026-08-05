"""Answer generation + per-token confidence signals.

The model is prompted to emit the answer phrase ONLY (no preamble), so the
generated span IS the answer-entity span and we probe confidence over those
generated tokens. Returns several signals to allow gate ablation:

  max_ent       max per-token entropy across the answer span -- high = uncertain
  mean_ent      mean per-token entropy
  tail_leakage  sum of -log(top-5 probability mass) -- high = uncertain
  min_lp        minimum log-prob of the chosen tokens -- low = uncertain
  maqa_margin   first-token top-1/top-2 probability margin -- low = uncertain
  mean_margin   prefix-mean top-1/top-2 probability margin -- low = uncertain
  max_margin    prefix-max top-1/top-2 probability margin -- low = uncertain
  targ_unc      prefix-mean exp(-raw_logit_gap / beta) -- high = uncertain
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

ANSWER_INSTRUCTION = (
    "Answer the question with the answer phrase only. "
    "No explanation, no preamble, no full sentence — just the answer."
)


@dataclass
class AnswerResult:
    text: str
    max_ent: float
    mean_ent: float
    tail_leakage: float
    min_lp: float
    maqa_margin: float
    mean_margin: float
    max_margin: float
    targ_logit_gap: float
    targ_unc: float

    def signal(self, name: str) -> float:
        """Return the confidence signal selected for the gate."""
        return {"max_ent": self.max_ent,
                "mean_ent": self.mean_ent,
                "tail_leakage": self.tail_leakage,
                "tl_sum": self.tail_leakage,
                "min_lp": self.min_lp,
                "maqa_margin": self.maqa_margin,
                "mean_margin": self.mean_margin,
                "max_margin": self.max_margin,
                "targ_logit_gap": self.targ_logit_gap,
                "targ_unc": self.targ_unc}[name]


def _prompt(tok, sub_question: str, context: str | None,
            orig_question: str | None = None) -> str:
    body = sub_question
    if context:
        body = f"Context:\n{context}\n\nQuestion: {sub_question}"
    if orig_question and orig_question.strip() != sub_question.strip():
        # Sub-questions produced by substitution can drop disambiguating
        # qualifiers from the original question (e.g. the state in
        # "Sullivan County, New Hampshire"); prepend it so the answer span
        # is grounded in the right entity.
        body = f"Original question: {orig_question}\n\n{body}"
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def answer_with_entropy(model, tok, sub_question: str,
                        context: str | None = None,
                        max_new_tokens: int = 48,
                        targ_prefix_k: int = 8,
                        targ_beta: float = 1.0,
                        orig_question: str | None = None) -> AnswerResult:
    if targ_beta <= 0:
        raise ValueError("targ_beta must be positive")
    prompt = _prompt(tok, sub_question, context, orig_question=orig_question)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.eos_token_id,
    )
    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    eos = tok.eos_token_id
    keep = [i for i, t in enumerate(gen_ids.tolist()) if t != eos]
    if not keep:
        return AnswerResult(
            text=text,
            max_ent=0.0,
            mean_ent=0.0,
            tail_leakage=0.0,
            min_lp=0.0,
            maqa_margin=0.0,
            mean_margin=0.0,
            max_margin=0.0,
            targ_logit_gap=0.0,
            targ_unc=1.0,
        )

    ents, leakages, lps = [], [], []
    prob_margins, logit_gaps, targ_uncs = [], [], []
    maqa_margin = 0.0
    prefix_keep = set(keep[:max(1, targ_prefix_k)])
    for i in keep:
        logits = out.scores[i][0].float()
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        ents.append(-(p * log_p).sum().item())
        top5_mass = torch.topk(p, k=5).values.sum().item()
        leakages.append(-math.log(max(top5_mass, 1e-30)))
        lps.append(log_p[gen_ids[i]].item())
        top_logits = torch.topk(logits, k=2).values
        logit_gap = (top_logits[0] - top_logits[1]).item()
        top_probs = torch.topk(p, k=2).values
        prob_margin = (top_probs[0] - top_probs[1]).item()
        if i == keep[0]:
            maqa_margin = prob_margin
        if i in prefix_keep:
            prob_margins.append(prob_margin)
            logit_gaps.append(logit_gap)
            targ_uncs.append(math.exp(-logit_gap / targ_beta))
    return AnswerResult(
        text=text,
        max_ent=max(ents),
        mean_ent=sum(ents) / len(ents),
        tail_leakage=sum(leakages),
        min_lp=min(lps),
        maqa_margin=maqa_margin,
        mean_margin=sum(prob_margins) / len(prob_margins),
        max_margin=max(prob_margins),
        targ_logit_gap=sum(logit_gaps) / len(logit_gaps),
        targ_unc=sum(targ_uncs) / len(targ_uncs),
    )
