"""
Main experiment: Self-Certainty gated RAG vs. CoT baseline vs. always-retrieve.

Protocol (strict DRAGIN / DeepRAG comparison):
  * Model:           meta-llama/Meta-Llama-3-8B-Instruct  (matches DeepRAG)
  * Corpus:          Wikipedia psgs_w100, BM25 via Elasticsearch index `wiki`
  * Top-k:           3 passages per retrieval
  * Generation:      temperature=0, max_new_tokens=100 (DRAGIN default)
  * Fewshot prompt:  8-shot DRAGIN-style CoT ("Question: ... Answer: ... So the answer is X.")
  * Evaluation:      DRAGIN's normalize_answer + EM/F1 + "the answer is" extraction
  * Data:            top 1000 from dev split (matches DeepRAG scripts/inference/run.sh)

We generate once per (question, method=no_rag | single_rag); the SC-gated RAG
variant is composed post-hoc at multiple thresholds — it takes the no-RAG
output when SC ≥ τ and the single-RAG output otherwise.

Output per dataset:
  results/<dataset>_<model-tag>.jsonl       (per-example records)
  results/<dataset>_<model-tag>_summary.json (aggregate metrics)

Usage:
  python run_experiment.py --dataset hotpotqa --max_questions 1000
  python run_experiment.py --dataset 2wikimultihopqa --max_questions 50  # smoke test
"""

import argparse
import json
import logging
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from dataset_eval import (DATASET_CLASSES, build_fewshot_prompt, extract_answer,
                          load_dataset, score)
from retriever_bm25 import BM25Retriever


# --- CLI ------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=list(DATASET_CLASSES))
parser.add_argument("--max_questions", type=int, default=1000)
parser.add_argument("--model_id", default="meta-llama/Meta-Llama-3-8B-Instruct")
parser.add_argument("--hf_cache", default="/home/S113065528/Adaptive-RAG/hf_cache")
parser.add_argument("--topk", type=int, default=3)
parser.add_argument("--max_new_tokens", type=int, default=100)
parser.add_argument("--n_shots", type=int, default=8)
parser.add_argument("--es_index", default="wiki")
parser.add_argument("--device", default="cuda")
parser.add_argument("--output_dir",
                    default="/home/S113065528/Research/deeprag_compare/results")
parser.add_argument("--data_dir",
                    default="/home/S113065528/Research/deeprag_compare/data/top1000")
parser.add_argument("--log_dir",
                    default="/home/S113065528/Research/deeprag_compare/logs")
# Dynamic sentence-level RAG settings
parser.add_argument("--dyn_sc_threshold", type=float, default=None,
                    help="Absolute SC trigger threshold (None = disable absolute check)")
parser.add_argument("--dyn_drop_threshold", type=float, default=1.5,
                    help="Trigger RAG when sentence SC drops by this much vs prev 2")
parser.add_argument("--dyn_trigger_mode", default="either",
                    choices=["absolute", "drop", "either", "both"])
parser.add_argument("--dyn_max_retrievals", type=int, default=2,
                    help="Max retrieval events per question for dynamic RAG")
parser.add_argument("--dyn_min_sent_tokens", type=int, default=4,
                    help="Min tokens to count as a valid sentence boundary")
parser.add_argument("--run_dynamic", action="store_true",
                    help="Enable dynamic sentence-level RAG (token-by-token, slow; off by default)")
args = parser.parse_args()


MODEL_TAG = args.model_id.split("/")[-1].lower().replace(".", "")
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.log_dir, exist_ok=True)

LOG_FILE = os.path.join(args.log_dir, f"{args.dataset}_{MODEL_TAG}.log")
RESULTS_FILE = os.path.join(args.output_dir, f"{args.dataset}_{MODEL_TAG}.jsonl")
SUMMARY_FILE = os.path.join(args.output_dir, f"{args.dataset}_{MODEL_TAG}_summary.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)


# --- Model loading --------------------------------------------------------

def load_model(model_id, cache_dir, device):
    """Try local cache first (no auth needed); fall back to HF download."""
    token = os.environ.get("HF_TOKEN")
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        logging.info(f"Local cache miss for {model_id}, downloading from HF (needs HF_TOKEN)")
        tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, token=token)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    try:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device,
            torch_dtype=torch.bfloat16, local_files_only=True)
    except Exception:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, device_map=device,
            torch_dtype=torch.bfloat16, token=token)
    m.eval()
    return tok, m


# --- Core generation ------------------------------------------------------

@torch.no_grad()
def generate_and_score(model, tok, prompt: str, max_new: int):
    """Greedy generation + self-certainty + perplexity of the generated tokens."""
    prompt_ids = tok(prompt, return_tensors="pt",
                     add_special_tokens=False).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    out_ids = model.generate(
        prompt_ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    ans_ids = out_ids[0][prompt_len:]
    text = tok.decode(ans_ids, skip_special_tokens=True).strip()
    n_tok = len(ans_ids)
    if n_tok == 0:
        return {"text": "", "sc": 0.0, "ppl": float("inf"), "n_tokens": 0}

    full_ids = out_ids[:, :prompt_len + n_tok]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    # logits over the ANSWER positions (shifted by 1)
    ans_logits = logits[:, prompt_len - 1:prompt_len + n_tok - 1, :]\
        .squeeze(0).float()
    log_probs = F.log_softmax(ans_logits, dim=-1)

    ans_token_ids = ans_ids.to(model.device)
    token_lp = log_probs.gather(1, ans_token_ids.unsqueeze(1)).squeeze(1)
    avg_nll = -token_lp.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    V = float(log_probs.size(-1))
    lp_sum = log_probs.sum(dim=-1)
    # SC = -1/(nV) · Σ_t Σ_v log p - log V
    sc = (-lp_sum.mean().item() / V) - math.log(V)

    return {"text": text, "sc": sc, "ppl": ppl, "n_tokens": n_tok}


@torch.no_grad()
def regenerate_answer(model, tok, prompt: str, cot_text: str, max_new: int = 20) -> str:
    """Force-complete the answer when the primary generation didn't emit
    'the answer is'. Matches DeepRAG evaluate_ans.py's regenerate_answer:
    append ' So the answer is ' and let the model finish the phrase.
    """
    for cut in ("Question:", "\n\n", "Note:"):
        pos = cot_text.find(cut)
        if pos > 0:
            cot_text = cot_text[:pos]
    cot_text = cot_text.rstrip()
    cont_prompt = prompt + cot_text + " So the answer is "
    ids = tok(cont_prompt, return_tensors="pt",
              add_special_tokens=False).input_ids.to(model.device)
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    # Reconstruct the DRAGIN-parseable string
    return cot_text + " So the answer is " + gen.strip()


# --- Dynamic sentence-level SC-triggered RAG ------------------------------

import re as _re
_SENT_END_RE = _re.compile(r'[.!?]\s*$|\n\s*$')


@torch.no_grad()
def generate_dynamic_rag(model, tok, retriever, ds_cls, fewshot_prefix: str,
                          question: str, max_new: int, topk: int,
                          sc_threshold, drop_threshold, trigger_mode,
                          max_retrievals: int, min_sent_tokens: int):
    """
    Token-by-token greedy generation with sentence-level SC-triggered retrieval.
    Mirrors exp10_sentence_rag.py but uses DRAGIN fewshot prompt and BM25Retriever.
    """
    V = float(model.config.vocab_size)
    log_V = math.log(V)
    device = model.device

    def _make_prompt(docs_txt=None):
        if docs_txt:
            return build_single_rag_prompt(ds_cls, fewshot_prefix, question, docs_txt)
        return build_no_rag_prompt(ds_cls, fewshot_prefix, question)

    # Bootstrap: feed prompt, get KV cache + last logits
    prompt_ids = tok(_make_prompt(), return_tensors="pt",
                     add_special_tokens=False).input_ids.to(device)
    out = model(prompt_ids, use_cache=True)
    past_kv = out.past_key_values
    last_logits = out.logits[0, -1, :].float()

    generated_ids = []
    sc_per_token = []
    sentence_scs = []
    trigger_events = []
    current_sent_start = 0
    n_retrievals = 0

    def _pick(logits):
        lp = F.log_softmax(logits, dim=-1)
        sc_tok = (-lp.sum() / V - log_V).item()
        return int(logits.argmax()), sc_tok

    def _is_sent_end(recent: str) -> bool:
        return bool(_SENT_END_RE.search(recent))

    def _should_trigger(sent_sc, history) -> bool:
        abs_fire = sc_threshold is not None and sent_sc < sc_threshold
        drop_fire = False
        if drop_threshold is not None and len(history) >= 2:
            drop_fire = (float(np.mean(history[-2:])) - sent_sc) >= drop_threshold
        if trigger_mode == "absolute": return abs_fire
        if trigger_mode == "drop":    return drop_fire
        if trigger_mode == "both":    return abs_fire and drop_fire
        return abs_fire or drop_fire  # "either" (default)

    step = 0
    while step < max_new:
        next_id, tok_sc = _pick(last_logits)
        if next_id == tok.eos_token_id:
            break

        generated_ids.append(next_id)
        sc_per_token.append(tok_sc)
        step += 1

        look_back = min(10, len(generated_ids))
        recent_text = tok.decode(generated_ids[-look_back:], skip_special_tokens=True)
        n_in_sent = step - current_sent_start

        if _is_sent_end(recent_text) and n_in_sent >= min_sent_tokens:
            sent_scs = sc_per_token[current_sent_start:]
            if sent_scs:
                mean_sc = float(np.mean(sent_scs))
                sentence_scs.append(mean_sc)

                if n_retrievals < max_retrievals and _should_trigger(mean_sc, sentence_scs[:-1]):
                    partial_cot = tok.decode(generated_ids, skip_special_tokens=True)
                    query = question + " " + partial_cot[-300:]
                    docs = retriever.search(query, topk=topk)
                    if docs:
                        docs_txt = BM25Retriever.format_context(docs)
                        new_prompt_ids = tok(_make_prompt(docs_txt), return_tensors="pt",
                                             add_special_tokens=False).input_ids.to(device)
                        partial_ids = torch.tensor([generated_ids], device=device)
                        full_ids = torch.cat([new_prompt_ids, partial_ids], dim=1)
                        out2 = model(full_ids, use_cache=True)
                        past_kv = out2.past_key_values
                        last_logits = out2.logits[0, -1, :].float()
                        n_retrievals += 1
                        trigger_events.append({
                            "step": step, "sent_sc": mean_sc,
                            "titles": [d["title"] for d in docs],
                        })
                        current_sent_start = step
                        continue

            current_sent_start = step

        new_input = torch.tensor([[next_id]], device=device)
        out = model(new_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[0, -1, :].float()

    text = tok.decode(generated_ids, skip_special_tokens=True).strip()
    overall_sc = float(np.mean(sc_per_token)) if sc_per_token else 0.0
    return {
        "text": text,
        "sc": overall_sc,
        "n_tokens": len(generated_ids),
        "n_retrievals": n_retrievals,
        "sentence_scs": sentence_scs,
        "trigger_events": trigger_events,
    }


# --- Prompt builders ------------------------------------------------------

def build_no_rag_prompt(ds_cls, fewshot_prefix: str, question: str) -> str:
    return fewshot_prefix + ds_cls.test_input(question) + " "


def build_single_rag_prompt(ds_cls, fewshot_prefix: str, question: str, docs_text: str) -> str:
    # Insert retrieved context just before the question block.
    return (
        fewshot_prefix
        + "Context (retrieved from Wikipedia):\n" + docs_text + "\n\n"
        + ds_cls.test_input(question) + " "
    )


# --- Main loop ------------------------------------------------------------

def main():
    ds_cls = DATASET_CLASSES[args.dataset]
    data = load_dataset(args.dataset, args.data_dir)
    data = data[: args.max_questions]
    fewshot = build_fewshot_prompt(ds_cls, args.n_shots)

    logging.info(f"Dataset={args.dataset}  N={len(data)}  model={args.model_id}")
    logging.info(f"Fewshot tokens (approx chars)={len(fewshot)}  topk={args.topk}")

    logging.info("Loading model...")
    tok, model = load_model(args.model_id, args.hf_cache, args.device)

    retriever = BM25Retriever(index=args.es_index)

    # Warm up retriever once
    _ = retriever.search("what is the capital of France?", topk=2)

    records = []
    t0 = time.time()
    for ex in tqdm(data, desc=args.dataset):
        golds = ds_cls.golds(ex)

        # 1) CoT no-RAG generation + SC + PPL
        p_nr = build_no_rag_prompt(ds_cls, fewshot, ex["question"])
        g_nr = generate_and_score(model, tok, p_nr, args.max_new_tokens)
        nr_text = g_nr["text"]
        if "the answer is" not in nr_text:
            nr_text = regenerate_answer(model, tok, p_nr, g_nr["text"])
        pred_nr = extract_answer(nr_text)
        sc_nr = score(pred_nr, golds)

        # 2) Single-retrieval RAG (always retrieve)
        docs = retriever.search(ex["question"], topk=args.topk)
        docs_text = BM25Retriever.format_context(docs)
        p_rag = build_single_rag_prompt(ds_cls, fewshot, ex["question"], docs_text)
        g_rag = generate_and_score(model, tok, p_rag, args.max_new_tokens)
        rag_text = g_rag["text"]
        if "the answer is" not in rag_text:
            rag_text = regenerate_answer(model, tok, p_rag, g_rag["text"])
        pred_rag = extract_answer(rag_text)
        sc_rag = score(pred_rag, golds)

        # 3) Dynamic sentence-level SC-triggered RAG (only with --run_dynamic)
        if not args.run_dynamic:
            g_dyn = {"text": "", "sc": 0.0, "n_tokens": 0,
                     "n_retrievals": 0, "sentence_scs": [], "trigger_events": []}
            dyn_text = nr_text   # fall back to no-RAG result
            pred_dyn = pred_nr
            sc_dyn = sc_nr
        else:
            g_dyn = generate_dynamic_rag(
                model, tok, retriever, ds_cls, fewshot, ex["question"],
                max_new=args.max_new_tokens, topk=args.topk,
                sc_threshold=args.dyn_sc_threshold,
                drop_threshold=args.dyn_drop_threshold,
                trigger_mode=args.dyn_trigger_mode,
                max_retrievals=args.dyn_max_retrievals,
                min_sent_tokens=args.dyn_min_sent_tokens,
            )
            dyn_text = g_dyn["text"]
            if "the answer is" not in dyn_text:
                p_dyn_fallback = build_no_rag_prompt(ds_cls, fewshot, ex["question"])
                dyn_text = regenerate_answer(model, tok, p_dyn_fallback, g_dyn["text"])
            pred_dyn = extract_answer(dyn_text)
            sc_dyn = score(pred_dyn, golds)

        rec = {
            "qid": ex["qid"],
            "question": ex["question"],
            "golds": golds,
            # no-RAG branch
            "nr_text": nr_text, "nr_pred": pred_nr,
            "nr_sc": g_nr["sc"], "nr_ppl": g_nr["ppl"], "nr_n_tok": g_nr["n_tokens"],
            "nr_em": sc_nr["em"], "nr_f1": sc_nr["f1"],
            # single-RAG branch
            "rag_text": rag_text, "rag_pred": pred_rag,
            "rag_sc": g_rag["sc"], "rag_ppl": g_rag["ppl"], "rag_n_tok": g_rag["n_tokens"],
            "rag_em": sc_rag["em"], "rag_f1": sc_rag["f1"],
            # dynamic sentence-level RAG branch
            "dyn_text": dyn_text, "dyn_pred": pred_dyn,
            "dyn_sc": g_dyn["sc"], "dyn_n_tok": g_dyn["n_tokens"],
            "dyn_n_retrievals": g_dyn["n_retrievals"],
            "dyn_sentence_scs": g_dyn["sentence_scs"],
            "dyn_em": sc_dyn["em"], "dyn_f1": sc_dyn["f1"],
            # retrieval
            "retrieved_titles": [d["title"] for d in docs],
        }
        records.append(rec)

    elapsed = time.time() - t0
    logging.info(f"Generation done. Elapsed={elapsed/60:.1f} min")

    # Save per-example records
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logging.info(f"Wrote {RESULTS_FILE}")

    # Aggregate summary (including post-hoc SC-gated thresholds)
    summary = summarize(records, args.dataset)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Wrote {SUMMARY_FILE}")

    # Pretty print main numbers
    logging.info("\n=== Summary ===")
    for k, v in summary["methods"].items():
        logging.info(f"  {k:<18s}  EM={v['em']*100:5.2f}  F1={v['f1']*100:5.2f}  "
                     f"RCount={v['retrieve_rate']:.2f}")


def summarize(records, dataset_tag):
    n = len(records)
    def agg(field_em, field_f1, r_rate):
        em = sum(r[field_em] for r in records) / n
        f1 = sum(r[field_f1] for r in records) / n
        return {"em": em, "f1": f1, "retrieve_rate": r_rate}

    # Dynamic RAG retrieve rate = fraction of questions with ≥1 retrieval
    dyn_retrieve_rate = sum(1 for r in records if r.get("dyn_n_retrievals", 0) > 0) / n
    dyn_mean_retrievals = sum(r.get("dyn_n_retrievals", 0) for r in records) / n

    # Baselines
    methods = {
        "no_rag":         agg("nr_em", "nr_f1", 0.0),
        "single_rag":     agg("rag_em", "rag_f1", 1.0),
        "dynamic_rag":    {**agg("dyn_em", "dyn_f1", dyn_retrieve_rate),
                           "mean_retrievals": dyn_mean_retrievals},
    }

    # SC-gated: if SC_nr >= τ, take no-RAG pred; else take RAG pred.
    # Sweep τ at percentiles so the retrieve rate is controllable.
    import numpy as np
    sc_arr = sorted(r["nr_sc"] for r in records)
    for target_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # retrieve_rate = fraction with SC < τ, so τ = the (rate)-quantile of SC
        tau = sc_arr[max(0, int(target_rate * n) - 1)]
        em_sum = f1_sum = r_count = 0
        for r in records:
            if r["nr_sc"] >= tau:
                em_sum += r["nr_em"]; f1_sum += r["nr_f1"]
            else:
                em_sum += r["rag_em"]; f1_sum += r["rag_f1"]
                r_count += 1
        methods[f"sc_gated_rate{int(target_rate*100):02d}"] = {
            "em": em_sum / n, "f1": f1_sum / n,
            "retrieve_rate": r_count / n,
            "sc_threshold": tau,
        }

    return {
        "dataset": dataset_tag,
        "model": args.model_id,
        "n_questions": n,
        "max_new_tokens": args.max_new_tokens,
        "topk": args.topk,
        "n_shots": args.n_shots,
        "methods": methods,
    }


if __name__ == "__main__":
    main()
