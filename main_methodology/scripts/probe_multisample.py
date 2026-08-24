"""M2b: multi-sample direct-answer probe on the ORIGINAL question.

For each dev row we retrieve top-k context (same as the always-skip / M1 setting),
then sample the direct answer N times in a single batched generate call.
We DO NOT change the committed answer or EM (that stays the greedy M1 answer);
we only dump richer ROUTING SIGNALS that point-1 showed are strong:

  n_uniq            number of distinct normalized sampled answers   (point-1 AUROC 0.78)
  disc_entropy      Shannon entropy of the sampled-answer distribution
  sc_agree          majority-vote agreement fraction (max count / N)
  mean_max_ent      mean over samples of max per-token entropy (~ fe_max_entropy)
  mean_mean_ent     mean over samples of mean per-token entropy
  mean_min_lp       mean over samples of min token log-prob

Usage:
    python probe_multisample.py --base <base-model> --adapter <ckpt> --dev <dev.jsonl> \
        --top-k 5 --n-sample 15 --temperature 0.7 --out <jsonl> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.answerer import ANSWER_INSTRUCTION  # noqa: E402
from seek.retriever import PerQuestionBM25, format_context  # noqa: E402

BASE = "meta-llama/Meta-Llama-3-8B-Instruct"


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _prompt(tok, question: str, context: str | None) -> str:
    body = question
    if context:
        body = f"Context:\n{context}\n\nQuestion: {question}"
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def multisample(model, tok, question, context, n_sample, temperature, max_new_tokens=48):
    prompt = _prompt(tok, question, context)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=n_sample,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tok.eos_token_id,
    )
    plen = enc["input_ids"].shape[1]
    seqs = out.sequences[:, plen:]                  # [N, T]
    # out.scores: tuple length T, each [N, V]
    eos = tok.eos_token_id
    texts, maxents, meanents, minlps = [], [], [], []
    for s in range(n_sample):
        gen = seqs[s]
        texts.append(tok.decode(gen, skip_special_tokens=True).strip())
        keep = [t for t, tid in enumerate(gen.tolist()) if tid != eos]
        if not keep:
            maxents.append(0.0); meanents.append(0.0); minlps.append(0.0); continue
        ents, lps = [], []
        for t in keep:
            log_p = F.log_softmax(out.scores[t][s], dim=-1)
            p = log_p.exp()
            # torch.special.entr(p) = -p*log(p) with entr(0)=0 -> safe vs top_p -inf masking
            ents.append(torch.special.entr(p).sum().item())
            lps.append(log_p[gen[t]].item())
        maxents.append(max(ents)); meanents.append(sum(ents) / len(ents)); minlps.append(min(lps))

    norm = [_norm(t) for t in texts]
    cnt = Counter(norm)
    N = len(norm)
    disc_ent = -sum((c / N) * math.log(c / N) for c in cnt.values())
    sc_agree = max(cnt.values()) / N
    maj = cnt.most_common(1)[0][0]
    return {
        "samples": texts,
        "n_uniq": len(cnt),
        "disc_entropy": disc_ent,
        "sc_agree": sc_agree,
        "majority": maj,
        "mean_max_ent": sum(maxents) / N,
        "mean_mean_ent": sum(meanents) / N,
        "mean_min_lp": sum(minlps) / N,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-sample", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.dev)]
    if args.limit:
        rows = rows[:args.limit]

    out_f = open(args.out, "w", encoding="utf-8")
    for i, r in enumerate(rows, 1):
        paras = r.get("paragraphs") or []
        retr = PerQuestionBM25(paras) if paras else None
        ctx = format_context(retr.topk(r["question"], k=args.top_k)) if retr else None
        feat = multisample(model, tok, r["question"], ctx,
                           args.n_sample, args.temperature)
        feat["id"] = r.get("id")
        feat["gold"] = r["answer"]
        out_f.write(json.dumps(feat, ensure_ascii=False) + "\n")
        if i % 100 == 0:
            out_f.flush()
            print(f"  [{i}/{len(rows)}] n_uniq={feat['n_uniq']} sc={feat['sc_agree']:.2f}",
                  flush=True)
    out_f.close()
    print(f"DONE {args.out}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
