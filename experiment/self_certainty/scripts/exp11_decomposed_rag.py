"""
Exp11: Decomposed SC-triggered RAG (DeepRAG-style with SC as zero-shot classifier)
====================================================================================
Inspired by DeepRAG (arxiv 2502.01142), which models retrieval-augmented reasoning
as an MDP where a *learned* binary classifier decides whether to retrieve at each
reasoning step.

Here we replace the learned classifier with Self-Certainty (SC) — no fine-tuning
needed. SC = -1/V * Σ log P(v|context) - log(V)  (KL divergence from uniform)

Three-stage pipeline per question:
  Stage 1 — Decompose:
      Prompt LLM to break the question into 1-3 atomic sub-questions.
      Simple (single-hop) questions stay as 1 sub-question.
      Multi-hop questions yield 2-3 sub-questions.

  Stage 2 — Answer each subquery with SC-triggered RAG:
      For each sub-question:
        a. Generate a short parametric answer (~80 tokens), compute mean SC.
        b. If mean SC < threshold → retrieve using the sub-question as query.
        c. If retrieved: regenerate the answer with passages injected.
      Accumulate sub-answers as "known facts" for subsequent sub-questions.

  Stage 3 — Synthesise:
      Build a final CoT prompt using all sub-question answers as context.
      Extract final answer.

Key advantages over Exp10 (sentence-level trigger):
  - Retrieval query = atomic sub-question (targeted, semantically meaningful)
  - Decomposition naturally handles multi-hop structure (important for 2wiki)
  - Trigger aligned with reasoning units, not arbitrary sentence boundaries

Comparison table (all results at 500 questions):
  A. No-RAG CoT         — exp7 baseline
  B. Decomposed SC-RAG  — this experiment (exp11)
  C. Dynamic SC-RAG     — exp10 (sentence-level trigger, for reference)
  D. Static SC-Gated    — exp9 Q1 (whole-question trigger, for reference)

Usage:
  conda activate adaptive_rag
  python exp11_decomposed_rag.py --dataset nq
  python exp11_decomposed_rag.py --dataset 2wiki
  python exp11_decomposed_rag.py --dataset nq --max_questions 100   # quick test
"""

import os
import sys
sys.path.insert(0, "/home/S113065528/Adaptive-RAG/retriever_server")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import logging
import math
import re
import string
import collections
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from elasticsearch import Elasticsearch
from elasticsearch_retriever import ElasticsearchRetriever

# ================= Args =================
parser = argparse.ArgumentParser(description="Exp11: Decomposed SC-triggered RAG")
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique"])
parser.add_argument("--max_questions", type=int, default=500)
parser.add_argument("--sc_threshold", type=float, default=12.0,
                    help="Trigger retrieval when subquery mean SC < this (default: 12.0)")
parser.add_argument("--top_k", type=int, default=3,
                    help="Retrieved passages per triggered subquery (default: 3)")
parser.add_argument("--max_subqueries", type=int, default=3,
                    help="Max subqueries to decompose into (default: 3)")
parser.add_argument("--decompose_max_tokens", type=int, default=150,
                    help="Max tokens for decomposition output (default: 150)")
parser.add_argument("--subquery_max_tokens", type=int, default=80,
                    help="Max tokens per subquery answer (default: 80)")
parser.add_argument("--synthesis_max_tokens", type=int, default=300,
                    help="Max tokens for final synthesis (default: 300)")
args = parser.parse_args()

# ================= Paths =================
DATASET_MAP = {
    "nq":       ("nq",              "wiki"),
    "2wiki":    ("2wikimultihopqa", "2wikimultihopqa"),
    "hotpotqa": ("hotpotqa",        "hotpotqa"),
    "musique":  ("musique",         "musique"),
}
DATASET_NAME, CORPUS_NAME = DATASET_MAP[args.dataset]
DS_TAG = args.dataset

DATA_DIR     = "/home/S113065528/Adaptive-RAG/processed_data"
DATA_FILE    = os.path.join(DATA_DIR, DATASET_NAME, "test_subsampled.jsonl")
BASE_DIR     = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
EXP7_FILE    = os.path.join(BASE_DIR, "results", f"exp7_cot_{DS_TAG}_results.json")
EXP10_FILE   = os.path.join(BASE_DIR, "results", f"exp10_dynamic_{DS_TAG}_results.json")
OUTPUT_FILE  = os.path.join(BASE_DIR, "results", f"exp11_decomposed_{DS_TAG}_results.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "results", f"exp11_decomposed_{DS_TAG}_summary.json")
FIGURE_DIR   = os.path.join(BASE_DIR, "figures")
LOG_FILE     = os.path.join(BASE_DIR, "logs", f"exp11_decomposed_{DS_TAG}.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
              logging.StreamHandler()],
)

BLUE = "#4C72B0"; ORANGE = "#DD8452"; GREEN = "#55A868"; RED = "#C44E52"; PURPLE = "#9B59B6"
WIKI_STYLE_CORPORA = {"wiki"}


# ================= Eval =================
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def compute_f1(pred, gold):
    pt, gt = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(pt) & collections.Counter(gt)
    num = sum(common.values())
    if not pt or not gt: return int(pt == gt)
    if num == 0: return 0
    p, r = num / len(pt), num / len(gt)
    return 2 * p * r / (p + r)

def check_correct(pred, golds):
    pn = normalize_answer(pred)
    for g in golds:
        if pn == normalize_answer(g): return True
        if compute_f1(pred, g) >= 0.5: return True
    return False

FINAL_PATTERNS = [
    r"final answer\s*[:：]\s*(.+?)(?:\n|$)",
    r"\banswer\s*[:：]\s*(.+?)(?:\n|$)",
    r"the answer is\s*(.+?)(?:\.|\n|$)",
]
def extract_final_answer(text):
    text = text.strip()
    for pat in FINAL_PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m: return m[-1].strip().strip(".").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


# ================= Data loading =================
def load_data(filepath, max_n=None):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["question_text"]
            answers = []
            for a in item.get("answers_objects", []):
                answers.extend(a.get("spans", []))
            if q and answers:
                data.append({"id": item.get("question_id", item.get("id", "")),
                             "question": q, "answers": answers})
            if max_n and len(data) >= max_n:
                break
    return data


# ================= Retrieval =================
def retrieve_passages(es_client, corpus_name, query_text, top_k):
    if corpus_name in WIKI_STYLE_CORPORA:
        result = es_client.search(index=corpus_name, body={
            "size": top_k, "_source": ["title", "txt"],
            "query": {"match": {"txt": query_text}},
        })
        return [{"title": h["_source"].get("title", ""),
                 "paragraph_text": h["_source"].get("txt", ""),
                 "score": h["_score"]}
                for h in result["hits"]["hits"]]
    else:
        r = ElasticsearchRetriever(host="localhost", port=9200)
        return r.retrieve_paragraphs(corpus_name=corpus_name, query_text=query_text,
                                     max_hits_count=top_k)


# ================= Prompt builders =================
def build_decompose_prompt(tokenizer, question):
    """Ask LLM to decompose question into atomic sub-questions."""
    msgs = [
        {"role": "system", "content":
         "You decompose questions into atomic sub-questions. "
         "Output ONLY a numbered list (e.g. '1. ...'). "
         "For simple single-fact questions, output exactly 1 sub-question (the original). "
         "For multi-hop questions requiring multiple facts, output 2-3 sub-questions. "
         "Each sub-question must be self-contained and answerable independently."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_subquery_prompt(tokenizer, subquery, context_so_far="", passages=None):
    """Answer a single sub-question, optionally with retrieved context."""
    if passages:
        passage_text = "\n\n".join(
            f"Wikipedia Title: {p.get('title', '').strip()}\n{p.get('paragraph_text', '').strip()}"
            for p in passages
        )
        sys_content = (
            "Answer the question concisely (1-2 sentences) using the Wikipedia passages below.\n\n"
            + passage_text
        )
    else:
        sys_content = "Answer the question concisely in 1-2 sentences."

    if context_so_far:
        sys_content += f"\n\nAlready established:\n{context_so_far}"

    msgs = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": subquery},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def build_synthesis_prompt(tokenizer, question, subqa_context):
    """Synthesise the final answer from accumulated sub-question answers."""
    msgs = [
        {"role": "system", "content":
         "You are given facts gathered by answering sub-questions. "
         "Use them to answer the original question step by step. "
         "End your response with exactly: 'Final Answer: <your answer>'.\n\n"
         f"Facts:\n{subqa_context}"},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ================= Core: greedy generation with SC =================
@torch.no_grad()
def generate_with_sc(model, tokenizer, prompt, max_new_tokens):
    """
    Greedy generation from a prompt. Returns (text, mean_sc, sc_per_token).
    Uses KV cache for efficiency.
    """
    V = model.config.vocab_size
    log_V = math.log(V)
    device = model.device

    input_ids = tokenizer(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids.to(device)
    out = model(input_ids, use_cache=True)
    past_kv = out.past_key_values
    last_logits = out.logits[0, -1, :].float()

    generated_ids = []
    sc_per_token = []

    for _ in range(max_new_tokens):
        log_probs = F.log_softmax(last_logits, dim=-1)
        token_sc = (-log_probs.sum() / V - log_V).item()
        next_id = last_logits.argmax().item()

        if next_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_id)
        sc_per_token.append(token_sc)

        new_input = torch.tensor([[next_id]], device=device)
        out = model(new_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[0, -1, :].float()

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    mean_sc = float(np.mean(sc_per_token)) if sc_per_token else 0.0
    return text, mean_sc, sc_per_token


# ================= Subquery parsing =================
def parse_subqueries(decomp_text, original_question, max_subqueries):
    """
    Parse numbered list from decomposition output.
    Falls back to original question if parsing fails.
    """
    lines = decomp_text.strip().splitlines()
    subqs = []
    for line in lines:
        line = line.strip()
        # Match "1. ...", "1) ...", "- ..."
        m = re.match(r'^[\d]+[.)]\s*(.+)', line)
        if not m:
            m = re.match(r'^[-•]\s*(.+)', line)
        if m:
            q = m.group(1).strip().rstrip(".")
            if len(q) > 8:  # skip very short fragments
                subqs.append(q)

    if not subqs:
        # No numbered list found — fall back to original question
        return [original_question]

    return subqs[:max_subqueries]


# ================= Main solve =================
def solve_with_decomposed_rag(model, tokenizer, es_client, question,
                               sc_threshold, top_k, max_subqueries,
                               decompose_max_tokens, subquery_max_tokens,
                               synthesis_max_tokens):
    """
    Full three-stage pipeline for one question.
    Returns dict with all intermediate outputs.
    """
    # ---- Stage 1: Decompose ----
    decompose_prompt = build_decompose_prompt(tokenizer, question)
    decomp_text, _, _ = generate_with_sc(model, tokenizer, decompose_prompt,
                                          decompose_max_tokens)
    subqueries = parse_subqueries(decomp_text, question, max_subqueries)

    # ---- Stage 2: Answer each subquery ----
    subqa_pairs = []    # list of {"q", "a", "sc", "retrieved", "passages"}
    n_retrievals = 0

    for sq in subqueries:
        # Build context from previous sub-answers
        context_so_far = "\n".join(
            f"Q: {p['q']}\nA: {p['a']}" for p in subqa_pairs
        )

        # First try: parametric answer
        prompt_param = build_subquery_prompt(tokenizer, sq,
                                              context_so_far=context_so_far)
        answer, mean_sc, _ = generate_with_sc(model, tokenizer, prompt_param,
                                               subquery_max_tokens)

        retrieved = False
        passages_used = []

        if mean_sc < sc_threshold:
            # SC too low → retrieve using the sub-question as query
            try:
                passages = retrieve_passages(es_client, CORPUS_NAME, sq, top_k)
            except Exception as e:
                logging.warning(f"  Retrieval failed for subquery '{sq[:50]}': {e}")
                passages = []

            if passages:
                n_retrievals += 1
                retrieved = True
                passages_used = passages
                # Regenerate answer with retrieved context
                prompt_rag = build_subquery_prompt(tokenizer, sq,
                                                    context_so_far=context_so_far,
                                                    passages=passages)
                answer, mean_sc, _ = generate_with_sc(model, tokenizer, prompt_rag,
                                                       subquery_max_tokens)

        subqa_pairs.append({
            "q": sq,
            "a": answer,
            "sc_parametric": mean_sc,   # SC after potential RAG re-generation
            "retrieved": retrieved,
            "passage_titles": [p.get("title", "") for p in passages_used],
        })

    # ---- Stage 3: Synthesise ----
    subqa_context = "\n".join(
        f"Q{i+1}: {p['q']}\nA{i+1}: {p['a']}" for i, p in enumerate(subqa_pairs)
    )
    synthesis_prompt = build_synthesis_prompt(tokenizer, question, subqa_context)
    final_cot, _, _ = generate_with_sc(model, tokenizer, synthesis_prompt,
                                        synthesis_max_tokens)
    final_answer = extract_final_answer(final_cot)

    return {
        "decomp_text":       decomp_text,
        "subqueries":        [p["q"] for p in subqa_pairs],
        "subquery_answers":  [p["a"] for p in subqa_pairs],
        "subquery_scs":      [p["sc_parametric"] for p in subqa_pairs],
        "subquery_retrieved": [p["retrieved"] for p in subqa_pairs],
        "passage_titles":    [p["passage_titles"] for p in subqa_pairs],
        "n_retrievals":      n_retrievals,
        "n_subqueries":      len(subqueries),
        "final_cot":         final_cot,
        "final_answer":      final_answer,
    }


# ================= Main =================
def main():
    # Load exp7 baseline
    baseline_by_id = {}
    if os.path.exists(EXP7_FILE):
        with open(EXP7_FILE, encoding="utf-8") as f:
            for r in json.load(f):
                baseline_by_id[r["id"]] = r
        logging.info(f"Loaded {len(baseline_by_id)} exp7 baseline results")
    else:
        logging.warning(f"Exp7 baseline not found ({EXP7_FILE})")

    # Load exp10 for cross-comparison (optional)
    exp10_by_id = {}
    if os.path.exists(EXP10_FILE):
        with open(EXP10_FILE, encoding="utf-8") as f:
            for r in json.load(f):
                exp10_by_id[r["id"]] = r
        logging.info(f"Loaded {len(exp10_by_id)} exp10 results (for cross-comparison)")

    # Load questions
    data = load_data(DATA_FILE, args.max_questions)
    logging.info(f"Dataset: {args.dataset} | Corpus: {CORPUS_NAME}")
    logging.info(f"Questions: {len(data)}")
    logging.info(f"Config: SC_threshold={args.sc_threshold} | top_k={args.top_k} | "
                 f"max_subqueries={args.max_subqueries}")
    logging.info("=" * 60)

    # Elasticsearch
    es = Elasticsearch(["http://localhost:9200"], request_timeout=30)
    try:
        retrieve_passages(es, CORPUS_NAME, "test", 1)
        logging.info("Retriever OK.")
    except Exception as e:
        logging.error(f"Retriever failed: {e}")
        sys.exit(1)

    # Model
    logging.info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, device_map="cuda",
        torch_dtype=torch.bfloat16, token=HF_TOKEN, local_files_only=True)
    model.eval()
    logging.info("Model loaded.\n")

    # ---- Run ----
    results = []
    t0 = time.time()

    for item in tqdm(data, desc="Decomposed RAG"):
        try:
            out = solve_with_decomposed_rag(
                model, tokenizer, es, item["question"],
                sc_threshold=args.sc_threshold,
                top_k=args.top_k,
                max_subqueries=args.max_subqueries,
                decompose_max_tokens=args.decompose_max_tokens,
                subquery_max_tokens=args.subquery_max_tokens,
                synthesis_max_tokens=args.synthesis_max_tokens,
            )
        except Exception as e:
            logging.warning(f"[{item['id']}] Failed: {e}")
            continue

        is_correct = check_correct(out["final_answer"], item["answers"]) \
                     if out["final_answer"] else False

        b = baseline_by_id.get(item["id"], {})
        e10 = exp10_by_id.get(item["id"], {})

        entry = {
            "id":               item["id"],
            "question":         item["question"],
            "ground_truths":    item["answers"],
            # Baseline
            "baseline_final":   b.get("final_answer", ""),
            "baseline_correct": b.get("is_correct", None),
            "baseline_sc":      b.get("sc", None),
            # Exp10 (cross-reference)
            "exp10_correct":    e10.get("dynamic_correct", None),
            # Exp11
            "decomp_text":      out["decomp_text"],
            "subqueries":       out["subqueries"],
            "subquery_answers": out["subquery_answers"],
            "subquery_scs":     out["subquery_scs"],
            "subquery_retrieved": out["subquery_retrieved"],
            "passage_titles":   out["passage_titles"],
            "n_subqueries":     out["n_subqueries"],
            "n_retrievals":     out["n_retrievals"],
            "final_cot":        out["final_cot"],
            "final_answer":     out["final_answer"],
            "decomposed_correct": is_correct,
        }
        results.append(entry)

        mark  = "O" if is_correct else "X"
        bmark = "O" if b.get("is_correct") else ("X" if b else "?")
        logging.info(
            f"[{item['id']}] base={bmark} dec={mark} | "
            f"subqs={out['n_subqueries']} retrieved={out['n_retrievals']} | "
            f"final={out['final_answer'][:60]}"
        )

    elapsed = time.time() - t0
    logging.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/question)")

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Results saved: {OUTPUT_FILE}")

    # ========== Summary ==========
    logging.info("\n" + "=" * 60)
    logging.info("Results Summary")
    logging.info("=" * 60)

    n = len(results)
    n_correct_dec  = sum(r["decomposed_correct"] for r in results)
    n_correct_base = sum(1 for r in results if r.get("baseline_correct"))
    n_correct_e10  = sum(1 for r in results if r.get("exp10_correct"))
    has_baseline   = all(r.get("baseline_correct") is not None for r in results)
    has_exp10      = any(r.get("exp10_correct") is not None for r in results)

    acc_dec  = n_correct_dec  / n * 100
    acc_base = n_correct_base / n * 100 if has_baseline else None
    acc_e10  = n_correct_e10  / n * 100 if has_exp10  else None

    logging.info(f"\n  A. No-RAG CoT baseline:     {acc_base:.1f}%  (n={n})" if acc_base else "")
    logging.info(f"  B. Decomposed SC-RAG (exp11):{acc_dec:.1f}%")
    if acc_base:
        logging.info(f"     Gain vs baseline:        {acc_dec - acc_base:+.1f} pp")
    if acc_e10:
        logging.info(f"  C. Dynamic SC-RAG (exp10):  {acc_e10:.1f}%  (reference)")
        logging.info(f"     Exp11 vs Exp10:           {acc_dec - acc_e10:+.1f} pp")

    # Subquery stats
    n_subqs_all = [r["n_subqueries"] for r in results]
    logging.info(f"\n  Subquery decomposition:")
    for k in sorted(set(n_subqs_all)):
        cnt = sum(1 for x in n_subqs_all if x == k)
        logging.info(f"    {k} subquery(ies): {cnt} questions ({cnt/n*100:.0f}%)")

    # Retrieval stats
    retrievals = [r["n_retrievals"] for r in results]
    n_triggered = sum(1 for x in retrievals if x > 0)
    logging.info(f"\n  Questions with ≥1 retrieval: {n_triggered}/{n} ({n_triggered/n*100:.0f}%)")
    logging.info(f"  Mean retrievals per question: {np.mean(retrievals):.2f}")

    # Fix/regress
    if has_baseline:
        fixed      = sum(1 for r in results if not r["baseline_correct"] and r["decomposed_correct"])
        regressed  = sum(1 for r in results if r["baseline_correct"] and not r["decomposed_correct"])
        stay_right = sum(1 for r in results if r["baseline_correct"] and r["decomposed_correct"])
        stay_wrong = sum(1 for r in results if not r["baseline_correct"] and not r["decomposed_correct"])
        logging.info(f"\n  Case breakdown:")
        logging.info(f"    Fixed by RAG:  {fixed}")
        logging.info(f"    Stayed right:  {stay_right}")
        logging.info(f"    Regressed:     {regressed}")
        logging.info(f"    Stayed wrong:  {stay_wrong}")

    # Acc by n_subqueries (does decomposition help more on multi-hop?)
    logging.info(f"\n  Accuracy by n_subqueries:")
    for k in sorted(set(n_subqs_all)):
        grp = [r for r in results if r["n_subqueries"] == k]
        if grp:
            g_acc = sum(r["decomposed_correct"] for r in grp) / len(grp) * 100
            b_acc = sum(1 for r in grp if r.get("baseline_correct")) / len(grp) * 100
            logging.info(f"    {k} subq: n={len(grp)}  dec={g_acc:.1f}%  base={b_acc:.1f}%  "
                         f"Δ={g_acc-b_acc:+.1f} pp")

    # Save summary
    summary = {
        "dataset": args.dataset, "mode": "decomposed_sc_rag",
        "n_questions": n,
        "config": {
            "sc_threshold":         args.sc_threshold,
            "top_k":                args.top_k,
            "max_subqueries":       args.max_subqueries,
            "subquery_max_tokens":  args.subquery_max_tokens,
            "synthesis_max_tokens": args.synthesis_max_tokens,
        },
        "accuracy": {
            "decomposed_rag":  round(acc_dec, 2),
            "no_rag_baseline": round(acc_base, 2) if acc_base else None,
            "dynamic_rag_exp10": round(acc_e10, 2) if acc_e10 else None,
            "gain_vs_baseline": round(acc_dec - acc_base, 2) if acc_base else None,
            "gain_vs_exp10":    round(acc_dec - acc_e10, 2) if acc_e10 else None,
        },
        "subquery_stats": {
            "distribution": {str(k): sum(1 for x in n_subqs_all if x == k)
                             for k in sorted(set(n_subqs_all))},
            "mean_subqueries": round(float(np.mean(n_subqs_all)), 2),
        },
        "retrieval_stats": {
            "questions_with_retrieval": n_triggered,
            "fraction_triggered": round(n_triggered / n, 4),
            "mean_retrievals": round(float(np.mean(retrievals)), 3),
        },
    }
    if has_baseline:
        summary["case_breakdown"] = {
            "fixed": fixed, "stayed_right": stay_right,
            "regressed": regressed, "stayed_wrong": stay_wrong,
        }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nSummary saved: {SUMMARY_FILE}")

    # ========== Plots ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Exp11: Decomposed SC-triggered RAG ({DS_TAG.upper()})\n"
        f"SC threshold={args.sc_threshold} | max_subqueries={args.max_subqueries}",
        fontsize=13, fontweight="bold"
    )

    # Panel 1: accuracy comparison (baseline / exp11 / exp10 if available)
    ax = axes[0]
    labels, vals, colors = [], [], []
    if acc_base:
        labels.append("No-RAG\nCoT"); vals.append(acc_base); colors.append(BLUE)
    labels.append("Decomposed\nSC-RAG"); vals.append(acc_dec); colors.append(GREEN)
    if acc_e10:
        labels.append("Dynamic\nSC-RAG"); vals.append(acc_e10); colors.append(ORANGE)

    bars = ax.bar(labels, vals, color=colors, edgecolor="k", linewidth=0.5, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy"); ax.set_ylim(0, 100)

    # Panel 2: subquery count distribution
    ax = axes[1]
    sq_counts = sorted(set(n_subqs_all))
    sq_dist = [sum(1 for x in n_subqs_all if x == k) for k in sq_counts]
    bars = ax.bar(sq_counts, sq_dist, color=PURPLE, edgecolor="k", linewidth=0.5)
    for bar, v in zip(bars, sq_dist):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Number of sub-questions")
    ax.set_ylabel("Count")
    ax.set_title("Subquery Decomposition Distribution")
    ax.set_xticks(sq_counts)

    # Panel 3: Δacc (exp11 vs baseline) per n_subqueries
    ax = axes[2]
    deltas = []
    labels_sq = []
    for k in sq_counts:
        grp = [r for r in results if r["n_subqueries"] == k]
        if len(grp) < 5:
            continue
        g_acc = sum(r["decomposed_correct"] for r in grp) / len(grp) * 100
        b_acc = sum(1 for r in grp if r.get("baseline_correct")) / len(grp) * 100
        deltas.append(g_acc - b_acc)
        labels_sq.append(f"{k} subq\n(n={len(grp)})")

    bar_colors = [GREEN if d >= 0 else RED for d in deltas]
    bars = ax.bar(labels_sq, deltas, color=bar_colors, edgecolor="k", linewidth=0.5)
    for bar, d in zip(bars, deltas):
        ypos = bar.get_height() + 0.2 if d >= 0 else bar.get_height() - 0.8
        ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                f"{d:+.1f} pp", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Δ Accuracy (pp)")
    ax.set_title("Gain vs Baseline by Subquery Count\n(exp11 contribution)")

    fig.tight_layout()
    fig_path = os.path.join(FIGURE_DIR, f"exp11_decomposed_{DS_TAG}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {fig_path}")
    plt.close(fig)

    logging.info("Done.")


if __name__ == "__main__":
    main()
