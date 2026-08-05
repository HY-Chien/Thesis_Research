"""Confidence-signal battery probe on the deployed DIRECT path.

Replicates the canonical dual-run direct path exactly (wiki ES BM25 top-8,
title_field off for the long original question, adapter mode C, greedy,
max_new_tokens=48) and dumps RICH raw probability data per question so that
~30 confidence signals can be computed offline without re-running the model:

  1. greedy regeneration with output_scores -> per-token full-vocab stats
     (entropy, Renyi-2 / Renyi-0.5 entropy, temperature-scaled entropy,
      chosen-token logprob, top-1/2 probs, top-5/10 mass, logit gap)
  2. N sampled answers (temp 0.7, top_p 0.95) with per-sample seq logprob
     -> semantic entropy / self-consistency signals
  3. teacher-forced scores of every unique candidate answer under the same
     prompt -> answer-level posterior (softmax over candidates)
  4. teacher-forced score of the committed answer WITHOUT context
     -> evidence-support gain (delta_ctx)
  5. P(True) self-verification with and without context (Kadavath-style)

Labels: EM recomputed alias-aware on the regenerated greedy answer; the saved
trace's direct_pred / direct_em / skip_probe_ent are stored alongside for a
alignment sanity check.

Usage:
    python scripts/probe_confidence_battery.py \
        --dev datasets/2wiki/raw/dev_first1000.jsonl \
        --trace results/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl \
        --out results/conf_battery_2wiki.jsonl [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.answerer import ANSWER_INSTRUCTION  # noqa: E402
from seek.retriever import OpenDomainBM25, format_context  # noqa: E402
from seek.verifier import exact_match, token_f1  # noqa: E402

BASE = os.environ.get(
    "SEEK_BASE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
)


def _prompt(tok, question: str, context: str | None) -> str:
    body = question
    if context:
        body = f"Context:\n{context}\n\nQuestion: {question}"
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _token_stats(logits: torch.Tensor, chosen_id: int) -> dict:
    """Full-vocab distribution statistics for one generation step."""
    logits = logits.float()
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    ent = torch.special.entr(p).sum().item()
    # Renyi entropies: H_a = log(sum p^a) / (1 - a)
    renyi2 = -torch.log((p * p).sum().clamp_min(1e-30)).item()
    renyi05 = (2.0 * torch.log(p.sqrt().sum().clamp_min(1e-30))).item()
    # temperature-scaled entropy (T=2 flattens -> probes tail competition)
    log_p2 = F.log_softmax(logits / 2.0, dim=-1)
    ent_t2 = torch.special.entr(log_p2.exp()).sum().item()
    topv, topi = torch.topk(p, k=10)
    top_logits = torch.topk(logits, k=2).values
    return {
        "lp": log_p[chosen_id].item(),
        "ent": ent,
        "renyi2": renyi2,
        "renyi05": renyi05,
        "ent_t2": ent_t2,
        "top1p": topv[0].item(),
        "top2p": topv[1].item(),
        "top5m": topv[:5].sum().item(),
        "top10m": topv.sum().item(),
        "lgap": (top_logits[0] - top_logits[1]).item(),
        "top_ids": topi.tolist(),
    }


@torch.inference_mode()
def greedy_probe(model, tok, prompt: str, max_new_tokens: int = 48) -> dict:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        return_dict_in_generate=True, output_scores=True,
        pad_token_id=tok.eos_token_id)
    gen_ids = out.sequences[0, enc["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    eos = tok.eos_token_id
    toks = []
    for i, tid in enumerate(gen_ids.tolist()):
        if tid == eos:
            continue
        st = _token_stats(out.scores[i][0], tid)
        st["text"] = tok.decode([tid])
        toks.append(st)
    return {"pred": text, "tokens": toks, "prompt_len": int(enc["input_ids"].shape[1])}


@torch.inference_mode()
def sample_probe(model, tok, prompt: str, n: int, temperature: float = 0.7,
                 max_new_tokens: int = 48) -> list[dict]:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=0.95, num_return_sequences=n,
        return_dict_in_generate=True, output_scores=True,
        pad_token_id=tok.eos_token_id)
    plen = enc["input_ids"].shape[1]
    seqs = out.sequences[:, plen:]
    eos = tok.eos_token_id
    samples = []
    for s in range(n):
        gen = seqs[s]
        keep = [t for t, tid in enumerate(gen.tolist()) if tid != eos]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        if not keep:
            samples.append({"text": text, "mean_lp": 0.0, "max_ent": 0.0})
            continue
        lps, ents = [], []
        for t in keep:
            log_p = F.log_softmax(out.scores[t][s].float(), dim=-1)
            lps.append(log_p[gen[t]].item())
            ents.append(torch.special.entr(log_p.exp()).sum().item())
        samples.append({"text": text,
                        "mean_lp": sum(lps) / len(lps),
                        "max_ent": max(ents)})
    return samples


@torch.inference_mode()
def teacher_force_score(model, tok, prompt: str, answers: list[str],
                        chunk: int = 6) -> list[float]:
    """Mean per-token logprob of each answer string under the prompt."""
    if not answers:
        return []
    enc = tok(prompt, return_tensors="pt")
    p_ids = enc["input_ids"][0]
    plen = p_ids.shape[0]
    seqs = []
    for a in answers:
        a_ids = tok(a, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if a_ids.numel() == 0:
            a_ids = torch.tensor([tok.eos_token_id])
        seqs.append(torch.cat([p_ids, a_ids]))
    pad = tok.eos_token_id
    out_scores: list[float] = []
    for c0 in range(0, len(seqs), chunk):
        batch = seqs[c0:c0 + chunk]
        maxlen = max(s.shape[0] for s in batch)
        input_ids = torch.full((len(batch), maxlen), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for j, s in enumerate(batch):
            input_ids[j, :s.shape[0]] = s
            attn[j, :s.shape[0]] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        log_p = F.log_softmax(logits, dim=-1)
        for j, s in enumerate(batch):
            alen = s.shape[0] - plen
            # logits at position t predict token t+1
            pos = torch.arange(plen - 1, plen - 1 + alen)
            tgt = s[plen:].to(model.device)
            lp = log_p[j, pos, :].gather(1, tgt.unsqueeze(1)).squeeze(1)
            out_scores.append(lp.mean().item())
    return out_scores


@torch.inference_mode()
def p_true_probe(model, tok, question: str, answer: str,
                 context: str | None) -> float:
    """Renormalised P(True) over {True, False} first tokens (Kadavath-style)."""
    body = f"Question: {question}\nProposed answer: {answer}\n"
    if context:
        body = f"Context:\n{context}\n\n" + body
    body += "Is the proposed answer correct? Reply with exactly one word: True or False."
    msgs = [{"role": "user", "content": body}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0, -1].float()
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()

    def mass(words):
        total = 0.0
        for w in words:
            ids = tok(w, add_special_tokens=False)["input_ids"]
            if ids:
                total += p[ids[0]].item()
        return total

    pt = mass(["True", " True", "true", " true"])
    pf = mass(["False", " False", "false", " false"])
    return pt / (pt + pf) if (pt + pf) > 0 else 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--trace", required=True,
                    help="canonical dual-run trace jsonl (for stored direct_pred/em sanity)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default="checkpoints/llama3-8b-decomposer-musique-2wiki-popqa")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--corpus", default="wiki")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--n-sample", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already present in --out and append")
    args = ap.parse_args()

    retriever = OpenDomainBM25(corpus_name=args.corpus)

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    rows = [
        json.loads(line)
        for line in open(args.dev, encoding="utf-8")
        if line.strip()
    ]
    if args.limit:
        rows = rows[:args.limit]
    trace = {}
    for line in open(args.trace, encoding="utf-8"):
        r = json.loads(line)
        trace[str(r.get("id"))] = r

    done = set()
    if args.resume and Path(args.out).exists():
        for line in open(args.out, encoding="utf-8"):
            try:
                done.add(str(json.loads(line).get("id")))
            except json.JSONDecodeError:
                pass
    out_f = open(args.out, "a" if args.resume else "w", encoding="utf-8")

    t_start = time.time()
    n_done = n_match = 0
    for i, row in enumerate(rows, 1):
        rid = str(row.get("id"))
        if rid in done:
            continue
        question = row["question"]
        golds = [row["answer"]] + (row.get("answer_aliases") or [])

        # --- deployed direct-path retrieval: original question, title off ---
        hits = retriever.topk(question, k=args.top_k, title_field=False)
        ctx = format_context(hits) if hits else None
        prompt_ctx = _prompt(tok, question, ctx)
        prompt_noctx = _prompt(tok, question, None)

        g = greedy_probe(model, tok, prompt_ctx)
        pred = g["pred"]
        em = int(exact_match(pred, golds))
        f1 = token_f1(pred, golds)

        samples = sample_probe(model, tok, prompt_ctx, args.n_sample, args.temperature)

        # unique candidates for the answer-level posterior (committed first)
        cand_texts: list[str] = []
        for t in [pred] + [s["text"] for s in samples]:
            t = (t or "").strip()
            if t and t not in cand_texts:
                cand_texts.append(t)
        cand_lps = teacher_force_score(model, tok, prompt_ctx, cand_texts)
        noctx_lp = teacher_force_score(model, tok, prompt_noctx, [pred])[0] if pred else 0.0

        ptrue_ctx = p_true_probe(model, tok, question, pred, ctx) if pred else 0.5
        ptrue_noctx = p_true_probe(model, tok, question, pred, None) if pred else 0.5

        tr = trace.get(rid, {})
        stored_pred = tr.get("direct_pred")
        stored_ent = (tr.get("direct") or {}).get("skip_probe_ent")
        n_done += 1
        n_match += int(stored_pred is not None and pred == stored_pred)

        rec = {
            "id": rid,
            "question": question,
            "gold": row["answer"],
            "aliases": row.get("answer_aliases") or [],
            "pred": pred,
            "em": em,
            "f1": round(f1, 4),
            "stored_pred": stored_pred,
            "stored_direct_em": tr.get("direct_em"),
            "stored_ent": stored_ent,
            "ctx_titles": [h.get("title") for h in hits],
            "tokens": g["tokens"],
            "samples": samples,
            "cand_texts": cand_texts,
            "cand_mean_lps": [round(x, 6) for x in cand_lps],
            "noctx_mean_lp": round(noctx_lp, 6),
            "p_true_ctx": round(ptrue_ctx, 6),
            "p_true_noctx": round(ptrue_noctx, 6),
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 20 == 0 or i == len(rows):
            out_f.flush()
            dt = time.time() - t_start
            print(f"  [{i}/{len(rows)}] em={em} pred_match_stored={n_match}/{n_done} "
                  f"({dt/max(n_done,1):.2f}s/q)", flush=True)
    out_f.close()
    print(f"DONE {args.out} rows={n_done} stored-pred match={n_match}/{n_done}", flush=True)


if __name__ == "__main__":
    main()
