"""Answer generation + per-token confidence signals.

The model is prompted to emit the answer phrase ONLY (no preamble), so the
generated span IS the answer-entity span and we probe confidence over those
generated tokens. Returns three signals to allow gate ablation:

  max_ent   max per-token entropy across the answer span — high = uncertain
  mean_ent  mean per-token entropy
  min_lp    minimum log-prob of the chosen tokens — low = uncertain
"""
from __future__ import annotations

from dataclasses import dataclass

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
    min_lp: float

    def signal(self, name: str) -> float:
        """Return the confidence signal selected for the gate."""
        return {"max_ent": self.max_ent,
                "mean_ent": self.mean_ent,
                "min_lp": self.min_lp}[name]


def _prompt(tok, sub_question: str, context: str | None) -> str:
    body = sub_question
    if context:
        body = f"Context:\n{context}\n\nQuestion: {sub_question}"
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def answer_with_entropy(model, tok, sub_question: str,
                        context: str | None = None,
                        max_new_tokens: int = 48) -> AnswerResult:
    prompt = _prompt(tok, sub_question, context)
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
        return AnswerResult(text, 0.0, 0.0, 0.0)

    ents, lps = [], []
    for i in keep:
        log_p = F.log_softmax(out.scores[i][0], dim=-1)
        p = log_p.exp()
        ents.append(-(p * log_p).sum().item())
        lps.append(log_p[gen_ids[i]].item())
    return AnswerResult(
        text=text,
        max_ent=max(ents),
        mean_ent=sum(ents) / len(ents),
        min_lp=min(lps),
    )
