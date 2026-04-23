"""
Exp10: Dynamic Sentence-Level SC-Triggered RAG (FLARE/DRAGIN-style with SC)
============================================================================
Unlike Exp8/9 which gate RAG at the whole-question level (run once, decide if
entire question needs RAG), this experiment triggers RAG DURING generation.
When a sentence's SC is low (uncertain) or drops sharply, we pause generation,
retrieve, inject context into the prompt, and continue.

Two trigger conditions (configurable):
  1. Absolute threshold: sentence_mean_sc < sc_threshold
  2. Relative drop:     (mean of previous 2 sentences) - current_sent_sc > drop_threshold

Method:
  - Generate token-by-token with KV cache (single-token forward passes reuse cache)
  - Each token's SC is computed from its next-token-prediction logits
  - When we hit a sentence boundary ('.', '!', '?', '\\n'), we compute the mean SC
    over that sentence's tokens and check trigger
  - If triggered: use (question + partial CoT) as retrieval query, rebuild prompt
    with retrieved passages prepended, reset KV cache, continue generating

Comparison:
  A. No-RAG CoT       — from exp7 baseline
  B. Dynamic-RAG      — this experiment
  C. SC-Gated (exp9)  — for reference (Q1 whole-question retrieval)

Usage:
  conda activate adaptive_rag
  python exp10_dynamic_rag.py --dataset nq --sc_threshold 12 --drop_threshold 3
  python exp10_dynamic_rag.py --dataset nq --max_questions 100  # faster test
"""

import os
import sys
sys.path.insert(0, "/home/S113065528/Adaptive-RAG/retriever_server")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique"])
parser.add_argument("--max_questions", type=int, default=500)
parser.add_argument("--top_k", type=int, default=3)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--sc_threshold", type=float, default=12.0,
                    help="Trigger RAG when sentence mean SC < this (absolute)")
parser.add_argument("--drop_threshold", type=float, default=3.0,
                    help="Trigger RAG when current SC drops this much vs prev 2 sentences")
parser.add_argument("--max_retrievals", type=int, default=3,
                    help="Max retrieval events per question (avoid infinite loops)")
parser.add_argument("--min_sent_tokens", type=int, default=3,
                    help="Minimum tokens for a valid sentence (avoid ' .' triggering)")
parser.add_argument("--trigger_mode", type=str, default="either",
                    choices=["absolute", "drop", "either", "both"],
                    help="Which condition triggers RAG")
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

DATA_DIR = "/home/S113065528/Adaptive-RAG/processed_data"
DATA_FILE = os.path.join(DATA_DIR, DATASET_NAME, "test_subsampled.jsonl")
BASE_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
EXP7_FILE    = os.path.join(BASE_DIR, "results", f"exp7_cot_{DS_TAG}_results.json")
OUTPUT_FILE  = os.path.join(BASE_DIR, "results", f"exp10_dynamic_{DS_TAG}_results.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "results", f"exp10_dynamic_{DS_TAG}_summary.json")
FIGURE_DIR   = os.path.join(BASE_DIR, "figures")
LOG_FILE     = os.path.join(BASE_DIR, "logs", f"exp10_dynamic_{DS_TAG}.log")
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

BLUE = "#4C72B0"; ORANGE = "#DD8452"; GREEN = "#55A868"; RED = "#C44E52"
WIKI_STYLE_CORPORA = {"wiki"}


# ================= Answer eval =================
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
    p, r = num/len(pt), num/len(gt)
    return 2*p*r/(p+r)

def check_correct(pred, golds):
    pn = normalize_answer(pred)
    for g in golds:
        if pn == normalize_answer(g): return True
        if compute_f1(pred, g) >= 0.5: return True
    return False

# ================= Final answer extraction =================
FINAL_PATTERNS = [
    r"final answer\s*[:：]\s*(.+?)(?:\n|$)",
    r"\banswer\s*[:：]\s*(.+?)(?:\n|$)",
    r"the answer is\s*(.+?)(?:\.|\n|$)",
]
def extract_final_answer(cot_text):
    text = cot_text.strip()
    for pat in FINAL_PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m: return m[-1].strip().strip(".").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text

# ================= Load data =================
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
            if max_n and len(data) >= max_n: break
    return data

# ================= Prompts =================
def build_cot_prompt(tokenizer, question):
    msgs = [
        {"role": "system", "content":
            "Answer the given question. Always conduct and show your "
            "step-by-step reasoning first, then give the final answer. "
            "End with a line: 'Final Answer: <your answer>'."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def build_cot_rag_prompt(tokenizer, question, passages):
    ctx = "\n\n".join(
        f"Wikipedia Title: {p.get('title','').strip()}\n{p.get('paragraph_text','').strip()}"
        for p in passages)
    msgs = [
        {"role": "system", "content":
            "Answer the given question using the Wikipedia passages below. "
            "Always conduct and show your step-by-step reasoning first, then give the final answer. "
            "End with a line: 'Final Answer: <your answer>'.\n\n" + ctx},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# ================= Retrieval =================
def retrieve_passages(es_client, corpus_name, query_text, top_k):
    if corpus_name in WIKI_STYLE_CORPORA:
        result = es_client.search(index=corpus_name, body={
            "size": top_k, "_source": ["title", "txt"],
            "query": {"match": {"txt": query_text}},
        })
        return [{"title": h["_source"].get("title",""),
                 "paragraph_text": h["_source"].get("txt",""),
                 "score": h["_score"]}
                for h in result["hits"]["hits"]]
    else:
        r = ElasticsearchRetriever(host="localhost", port=9200)
        return r.retrieve_paragraphs(corpus_name=corpus_name, query_text=query_text,
                                      max_hits_count=top_k)

# ================= Trigger logic =================
SENT_END_RE = re.compile(r'[.!?]\s*$|\n\s*$')

def is_sentence_end(recent_text: str) -> bool:
    """True if recent decoded text ends a sentence."""
    return bool(SENT_END_RE.search(recent_text))


def should_trigger(sent_sc, sc_history, mode, sc_threshold, drop_threshold):
    """Decide whether this sentence triggers RAG. Returns (bool, reason)."""
    abs_fire  = sent_sc < sc_threshold
    drop_fire = False
    if len(sc_history) >= 2:
        prev2_mean = float(np.mean(sc_history[-2:]))
        drop_fire  = (prev2_mean - sent_sc) >= drop_threshold

    if mode == "absolute": return (abs_fire,  "absolute" if abs_fire else None)
    if mode == "drop":     return (drop_fire, "drop"     if drop_fire else None)
    if mode == "either":   return (abs_fire or drop_fire,
                                    "absolute" if abs_fire else ("drop" if drop_fire else None))
    if mode == "both":     return (abs_fire and drop_fire,
                                    "both" if (abs_fire and drop_fire) else None)
    return (False, None)


# ================= Core: Dynamic generation =================
@torch.no_grad()
def generate_with_dynamic_rag(model, tokenizer, es_client, question,
                               max_new_tokens, sc_threshold, drop_threshold,
                               trigger_mode, top_k, max_retrievals, min_sent_tokens):
    """
    Token-by-token CoT generation with sentence-level SC-triggered RAG.
    Uses KV cache for efficiency.
    """
    V = model.config.vocab_size
    device = model.device
    log_V = math.log(V)

    # Initial prompt (no RAG)
    prompt = build_cot_prompt(tokenizer, question)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    # Feed entire prompt to build KV cache
    out = model(prompt_ids, use_cache=True)
    past_kv = out.past_key_values
    last_logits = out.logits[0, -1, :].float()

    generated_ids        = []   # only the newly generated tokens (CoT text)
    sc_per_token         = []   # SC for each generated token
    sentence_scs         = []   # one entry per completed sentence
    sentence_texts       = []
    trigger_events       = []   # list of dicts when RAG fired
    current_sent_start   = 0
    n_retrievals         = 0

    # Greedy from prompt's last_logits
    def pick_token(logits):
        log_probs = F.log_softmax(logits, dim=-1)
        # SC for this step: -1/V * sum(log P) - log V
        token_sc = (-log_probs.sum() / V - log_V).item()
        return logits.argmax().item(), token_sc

    # Generation loop
    step = 0
    while step < max_new_tokens:
        next_id, tok_sc = pick_token(last_logits)

        if next_id == tokenizer.eos_token_id:
            break

        generated_ids.append(next_id)
        sc_per_token.append(tok_sc)
        step += 1

        # Check sentence end: decode last few tokens to see text
        # (Just decoding the single new token is unreliable for sentence detection)
        look_back = min(10, len(generated_ids))
        recent_text = tokenizer.decode(generated_ids[-look_back:], skip_special_tokens=True)
        n_in_sent = step - current_sent_start

        sent_ended = is_sentence_end(recent_text) and n_in_sent >= min_sent_tokens

        if sent_ended:
            sent_sc_vals = sc_per_token[current_sent_start:]
            if sent_sc_vals:
                mean_sc = float(np.mean(sent_sc_vals))
                sent_text = tokenizer.decode(
                    generated_ids[current_sent_start:], skip_special_tokens=True).strip()
                sentence_scs.append(mean_sc)
                sentence_texts.append(sent_text)

                # Trigger check (only if budget remains and we have at least 1 prev sentence)
                fire, reason = False, None
                if n_retrievals < max_retrievals and len(sentence_scs) >= 1:
                    fire, reason = should_trigger(mean_sc, sentence_scs[:-1],
                                                   trigger_mode, sc_threshold, drop_threshold)

                if fire:
                    # Build retrieval query: question + last N chars of partial CoT
                    partial_cot = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    query = f"{question} {partial_cot[-300:]}"
                    try:
                        passages = retrieve_passages(es_client, CORPUS_NAME, query, top_k)
                    except Exception as e:
                        logging.warning(f"Retrieval failed: {e}")
                        passages = []

                    if passages:
                        trigger_events.append({
                            "step":         step,
                            "sentence_idx": len(sentence_scs) - 1,
                            "sentence_sc":  mean_sc,
                            "reason":       reason,
                            "titles":       [p.get("title","") for p in passages],
                            "partial_cot_tail": partial_cot[-200:],
                        })
                        n_retrievals += 1

                        # Rebuild prompt with context + already-generated CoT
                        new_prompt = build_cot_rag_prompt(tokenizer, question, passages)
                        new_prompt_ids = tokenizer(
                            new_prompt, return_tensors="pt", add_special_tokens=False
                        ).input_ids.to(device)
                        partial_ids = torch.tensor([generated_ids], device=device)
                        full_ids = torch.cat([new_prompt_ids, partial_ids], dim=1)

                        # Reset KV cache: feed full_ids
                        out = model(full_ids, use_cache=True)
                        past_kv = out.past_key_values
                        last_logits = out.logits[0, -1, :].float()

                        current_sent_start = step
                        continue  # don't re-feed last_id below

            current_sent_start = step

        # Default path: feed just the new token with cached KV
        new_input = torch.tensor([[next_id]], device=device)
        out = model(new_input, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        last_logits = out.logits[0, -1, :].float()

    cot_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    final = extract_final_answer(cot_text)

    return {
        "cot":              cot_text,
        "final_answer":     final,
        "n_tokens":         len(generated_ids),
        "overall_sc":       float(np.mean(sc_per_token)) if sc_per_token else 0.0,
        "sentence_scs":     sentence_scs,
        "sentence_texts":   sentence_texts,
        "n_retrievals":     n_retrievals,
        "trigger_events":   trigger_events,
    }


# ================= Main =================
def main():
    # Load exp7 baseline (for comparison)
    baseline_by_id = {}
    if os.path.exists(EXP7_FILE):
        with open(EXP7_FILE, encoding="utf-8") as f:
            for r in json.load(f):
                baseline_by_id[r["id"]] = r
        logging.info(f"Loaded {len(baseline_by_id)} baseline results from exp7")
    else:
        logging.warning(f"Exp7 baseline not found ({EXP7_FILE}) — comparison disabled")

    # Load questions
    data = load_data(DATA_FILE, args.max_questions)
    logging.info(f"Dataset: {args.dataset} | Corpus: {CORPUS_NAME}")
    logging.info(f"Questions: {len(data)}")
    logging.info(f"Trigger: mode={args.trigger_mode} | SC<{args.sc_threshold} | "
                 f"drop≥{args.drop_threshold} | max_retrievals={args.max_retrievals}")
    logging.info("=" * 60)

    # Elasticsearch
    es = Elasticsearch(["http://localhost:9200"], request_timeout=30)
    try:
        retrieve_passages(es, CORPUS_NAME, "test", 1)
        logging.info("Retriever OK.")
    except Exception as e:
        logging.error(f"Retriever failed: {e}"); sys.exit(1)

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

    # Run
    results = []
    t0 = time.time()
    for item in tqdm(data, desc="Dynamic RAG"):
        try:
            out = generate_with_dynamic_rag(
                model, tokenizer, es, item["question"],
                max_new_tokens=args.max_new_tokens,
                sc_threshold=args.sc_threshold,
                drop_threshold=args.drop_threshold,
                trigger_mode=args.trigger_mode,
                top_k=args.top_k,
                max_retrievals=args.max_retrievals,
                min_sent_tokens=args.min_sent_tokens,
            )
        except Exception as e:
            logging.warning(f"[{item['id']}] generation failed: {e}")
            continue

        is_correct = check_correct(out["final_answer"], item["answers"]) \
                     if out["final_answer"] else False

        baseline_info = baseline_by_id.get(item["id"], {})
        entry = {
            "id":             item["id"],
            "question":       item["question"],
            "ground_truths":  item["answers"],
            "baseline_cot":   baseline_info.get("cot", ""),
            "baseline_final": baseline_info.get("final_answer", ""),
            "baseline_sc":    baseline_info.get("sc", None),
            "baseline_correct": baseline_info.get("is_correct", None),
            "dynamic_cot":    out["cot"],
            "dynamic_final":  out["final_answer"],
            "dynamic_sc":     out["overall_sc"],
            "dynamic_correct": is_correct,
            "n_tokens":       out["n_tokens"],
            "n_retrievals":   out["n_retrievals"],
            "sentence_scs":   out["sentence_scs"],
            "trigger_events": out["trigger_events"],
        }
        results.append(entry)

        mark = "O" if is_correct else "X"
        bmark = "O" if baseline_info.get("is_correct", False) else ("X" if baseline_info else "?")
        logging.info(
            f"[{item['id']}] base={bmark} dyn={mark} | "
            f"retrievals={out['n_retrievals']} | final={out['final_answer'][:60]}"
        )

    elapsed = time.time() - t0
    logging.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/max(len(results),1):.1f}s/question)")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"Results saved: {OUTPUT_FILE}")

    # ========== Summary ==========
    logging.info("\n" + "=" * 60)
    logging.info("Results Summary")
    logging.info("=" * 60)

    n = len(results)
    n_correct_dyn  = sum(r["dynamic_correct"] for r in results)
    n_correct_base = sum(1 for r in results if r.get("baseline_correct"))
    has_baseline   = all(r.get("baseline_correct") is not None for r in results)

    acc_dyn  = n_correct_dyn / n * 100
    logging.info(f"\n  Dynamic-RAG acc: {acc_dyn:.1f}%  (n={n})")
    if has_baseline:
        acc_base = n_correct_base / n * 100
        logging.info(f"  No-RAG baseline: {acc_base:.1f}%")
        logging.info(f"  Gain:            {acc_dyn - acc_base:+.1f} pp")

    # Retrieval statistics
    retrievals    = [r["n_retrievals"] for r in results]
    n_triggered   = sum(1 for x in retrievals if x > 0)
    mean_retrieve = float(np.mean(retrievals))
    logging.info(f"\n  Questions with ≥1 retrieval: {n_triggered}/{n} ({n_triggered/n*100:.0f}%)")
    logging.info(f"  Mean retrievals per question: {mean_retrieve:.2f}")
    logging.info(f"  Retrieval distribution:")
    for k in range(args.max_retrievals + 1):
        cnt = sum(1 for x in retrievals if x == k)
        logging.info(f"    {k}× retrievals: {cnt}")

    # Fix/regress breakdown
    if has_baseline:
        fixed      = sum(1 for r in results if not r["baseline_correct"] and r["dynamic_correct"])
        regressed  = sum(1 for r in results if r["baseline_correct"] and not r["dynamic_correct"])
        stay_right = sum(1 for r in results if r["baseline_correct"] and r["dynamic_correct"])
        stay_wrong = sum(1 for r in results if not r["baseline_correct"] and not r["dynamic_correct"])
        logging.info(f"\n  Case breakdown:")
        logging.info(f"    Fixed by RAG:  {fixed}")
        logging.info(f"    Stayed right:  {stay_right}")
        logging.info(f"    Regressed:     {regressed}")
        logging.info(f"    Stayed wrong:  {stay_wrong}")

    # Retrievals → accuracy
    logging.info(f"\n  Accuracy by n_retrievals:")
    for k in range(args.max_retrievals + 1):
        grp = [r for r in results if r["n_retrievals"] == k]
        if grp:
            g_acc = sum(r["dynamic_correct"] for r in grp) / len(grp) * 100
            logging.info(f"    {k}× retrievals: n={len(grp)}  acc={g_acc:.1f}%")

    # Save summary
    summary = {
        "dataset": args.dataset, "mode": "dynamic_rag_cot",
        "n_questions": n,
        "config": {
            "sc_threshold":   args.sc_threshold,
            "drop_threshold": args.drop_threshold,
            "trigger_mode":   args.trigger_mode,
            "top_k":          args.top_k,
            "max_retrievals": args.max_retrievals,
            "min_sent_tokens": args.min_sent_tokens,
        },
        "accuracy": {
            "dynamic_rag":    round(acc_dyn, 2),
            "no_rag_baseline": round(n_correct_base / n * 100, 2) if has_baseline else None,
            "gain":           round(acc_dyn - n_correct_base / n * 100, 2) if has_baseline else None,
        },
        "retrieval_stats": {
            "questions_with_retrieval": n_triggered,
            "fraction_triggered":       round(n_triggered / n, 4),
            "mean_retrievals":          round(mean_retrieve, 3),
            "distribution": {str(k): sum(1 for x in retrievals if x == k)
                             for k in range(args.max_retrievals + 1)},
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

    # ========== Plot ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Exp10: Dynamic SC-Triggered RAG ({DS_TAG.upper()})\n"
                 f"mode={args.trigger_mode}, SC<{args.sc_threshold}, drop≥{args.drop_threshold}",
                 fontsize=13, fontweight="bold")

    # Panel 1: accuracy comparison
    ax = axes[0]
    conds = ["No-RAG\nCoT"] if has_baseline else []
    vals  = [n_correct_base/n*100] if has_baseline else []
    cols  = [BLUE] if has_baseline else []
    conds.append("Dynamic\nRAG"); vals.append(acc_dyn); cols.append(GREEN)
    bars = ax.bar(conds, vals, color=cols, edgecolor="k", linewidth=0.5, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy"); ax.set_ylim(0, 100)

    # Panel 2: retrieval count distribution
    ax = axes[1]
    dist = [sum(1 for x in retrievals if x == k) for k in range(args.max_retrievals+1)]
    bars = ax.bar(range(args.max_retrievals+1), dist, color=ORANGE, edgecolor="k", linewidth=0.5)
    for bar, v in zip(bars, dist):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Retrievals per question")
    ax.set_ylabel("Count")
    ax.set_title("Retrieval Count Distribution")

    # Panel 3: acc by n_retrievals
    ax = axes[2]
    xs, ys, ns = [], [], []
    for k in range(args.max_retrievals+1):
        grp = [r for r in results if r["n_retrievals"] == k]
        if grp:
            xs.append(k)
            ys.append(sum(r["dynamic_correct"] for r in grp) / len(grp) * 100)
            ns.append(len(grp))
    bars = ax.bar(xs, ys, color=GREEN, edgecolor="k", linewidth=0.5)
    for bar, v, n_ in zip(bars, ys, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.0f}%\n(n={n_})", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Retrievals"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Retrieval Count")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp10_dynamic_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}"); plt.close(fig)

    logging.info("Done.")


if __name__ == "__main__":
    main()
