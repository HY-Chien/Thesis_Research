"""
Exp9: CoT + SC-Gated RAG (End-to-End)
======================================
The CoT version of Exp8. Uses exp7 CoT baseline results (SC scores for 500 questions)
and re-generates Q1 (lowest SC) questions with retrieved context + CoT prompt.

Four-way comparison:
  A. No-RAG CoT baseline  — use answers from exp7 directly
  B. SC-Gated RAG         — Q1 gets retrieved context + CoT re-generation
  C. Always-RAG           — all 500 questions get context + CoT re-generation
  D. Random-Gated RAG     — randomly select same n as Q1, give them RAG
                            simulated from C (1000 trials, no extra GPU)

Thesis payoff:
  - CoT under SC-Gated RAG should show larger gains than direct-answer Exp8
  - Because CoT SC signal is stronger (exp7: AUC 0.735 vs 0.668 for PPL)
  - Key: B > D proves SC selects better questions than random

Usage:
  conda activate adaptive_rag
  python exp9_cot_sc_gated_rag.py --dataset nq
  python exp9_cot_sc_gated_rag.py --dataset 2wiki
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

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from elasticsearch_retriever import ElasticsearchRetriever
from elasticsearch import Elasticsearch

# ================= Args =================
parser = argparse.ArgumentParser(description="Exp9: CoT SC-Gated RAG")
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique"])
parser.add_argument("--top_k", type=int, default=3,
                    help="Retrieved passages per question (default: 3)")
parser.add_argument("--gated_quantile", type=float, default=0.2,
                    help="Bottom SC quantile that triggers RAG (default: 0.2)")
parser.add_argument("--max_new_tokens", type=int, default=512,
                    help="Max new tokens for CoT generation (default: 512)")
parser.add_argument("--n_random_seeds", type=int, default=1000,
                    help="Random trials for Random-Gated baseline (no GPU, default: 1000)")
parser.add_argument("--skip_always_rag", action="store_true",
                    help="Skip Always-RAG (also disables Random-Gated baseline)")
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

BASE_DIR       = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
EXP7_FILE      = os.path.join(BASE_DIR, "results", f"exp7_cot_{DS_TAG}_results.json")
OUTPUT_FILE    = os.path.join(BASE_DIR, "results", f"exp9_cot_{DS_TAG}_results.json")
SUMMARY_FILE   = os.path.join(BASE_DIR, "results", f"exp9_cot_{DS_TAG}_summary.json")
FIGURE_DIR     = os.path.join(BASE_DIR, "figures")
LOG_FILE       = os.path.join(BASE_DIR, "logs", f"exp9_cot_{DS_TAG}.log")
MODEL_ID       = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE       = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
PURPLE = "#8172B3"

WIKI_STYLE_CORPORA = {"wiki"}


# ===================== Answer Evaluation =====================

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    common = collections.Counter(pt) & collections.Counter(gt)
    num = sum(common.values())
    if not pt or not gt:
        return int(pt == gt)
    if num == 0:
        return 0
    p, r = num / len(pt), num / len(gt)
    return 2 * p * r / (p + r)


def check_correct(pred, golds):
    pn = normalize_answer(pred)
    for g in golds:
        if pn == normalize_answer(g):
            return True
        if compute_f1(pred, g) >= 0.5:
            return True
    return False


# ===================== Final Answer Extraction (same as exp7) =====================

FINAL_PATTERNS = [
    r"final answer\s*[:：]\s*(.+?)(?:\n|$)",
    r"\banswer\s*[:：]\s*(.+?)(?:\n|$)",
    r"the answer is\s*(.+?)(?:\.|\n|$)",
    r"so,?\s*(.+?)(?:\.|\n|$)",
]


def extract_final_answer(cot_text):
    text = cot_text.strip()
    for pat in FINAL_PATTERNS:
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return m[-1].strip().strip(".").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text


# ===================== Prompt Builders =====================

def build_cot_prompt(tokenizer, question: str) -> str:
    """No-RAG CoT prompt — same as exp7."""
    messages = [
        {"role": "system", "content":
            "Answer the given question. Always conduct and show your "
            "step-by-step reasoning first, then give the final answer. "
            "End with a line: 'Final Answer: <your answer>'."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


def build_cot_rag_prompt(tokenizer, question: str, passages: list) -> str:
    """CoT prompt with retrieved context injected into system message."""
    if not passages:
        return build_cot_prompt(tokenizer, question)

    context_lines = []
    for p in passages:
        title = p.get("title", "").strip()
        text  = p.get("paragraph_text", "").strip()
        if title:
            context_lines.append(f"Wikipedia Title: {title}\n{text}")
        else:
            context_lines.append(text)
    context_str = "\n\n".join(context_lines)

    messages = [
        {"role": "system", "content":
            "Answer the given question using the Wikipedia passages below as reference. "
            "Always conduct and show your step-by-step reasoning first, then give the final answer. "
            "End with a line: 'Final Answer: <your answer>'.\n\n"
            + context_str},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


# ===================== Generation =====================

@torch.no_grad()
def generate_cot(model, tokenizer, prompt_text: str) -> dict:
    """Greedy CoT generation. Returns cot text, final answer, sc, ppl, n_tokens."""
    V = model.config.vocab_size
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer_ids = output_ids[0][prompt_len:]
    cot_text   = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    n_tok      = len(answer_ids)

    if n_tok == 0:
        return {"cot": "", "final_answer": "", "sc": 0.0, "ppl": float("inf"), "n_tokens": 0}

    full_ids = output_ids[:, :prompt_len + n_tok]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    ans_logits = logits[:, prompt_len - 1: prompt_len + n_tok - 1, :].squeeze(0).float()
    ans_ids    = answer_ids.to(model.device)
    log_probs  = F.log_softmax(ans_logits, dim=-1)

    token_lp = log_probs.gather(1, ans_ids.unsqueeze(1)).squeeze(1)
    avg_nll  = -token_lp.mean().item()
    ppl      = math.exp(min(avg_nll, 20))

    lp_sum = log_probs.sum(dim=-1)
    sc_per = -1.0 / float(V) * lp_sum - math.log(float(V))
    sc     = sc_per.mean().item()

    final = extract_final_answer(cot_text)
    return {"cot": cot_text, "final_answer": final, "sc": sc, "ppl": ppl, "n_tokens": n_tok}


# ===================== Retrieval =====================

def retrieve_passages(es_client, corpus_name: str, query_text: str, top_k: int) -> list:
    if corpus_name in WIKI_STYLE_CORPORA:
        query = {
            "size": top_k,
            "_source": ["title", "txt"],
            "query": {"match": {"txt": query_text}},
        }
        result = es_client.search(index=corpus_name, body=query)
        passages = []
        for hit in result.get("hits", {}).get("hits", []):
            src = hit["_source"]
            passages.append({
                "title":          src.get("title", ""),
                "paragraph_text": src.get("txt", ""),
                "score":          hit["_score"],
            })
        return passages
    else:
        retriever = ElasticsearchRetriever(host="localhost", port=9200)
        return retriever.retrieve_paragraphs(
            corpus_name=corpus_name,
            query_text=query_text,
            max_hits_count=top_k,
        )


# ===================== Random-Gated Simulation =====================

def compute_random_gated_acc(baseline_results, always_rag_results, n_triggered, n_seeds=1000):
    """Simulate Random-Gated from Always-RAG results. No GPU needed."""
    n       = len(baseline_results)
    ar_by_id = {r["id"]: r for r in always_rag_results}
    b_by_id  = {r["id"]: r for r in baseline_results}
    all_ids  = [r["id"] for r in baseline_results]

    rng  = np.random.default_rng(42)
    accs = []
    for _ in range(n_seeds):
        chosen  = set(rng.choice(all_ids, size=n_triggered, replace=False).tolist())
        correct = sum(
            ar_by_id[rid]["is_correct_always_rag"] if rid in chosen
            else b_by_id[rid]["is_correct"]
            for rid in all_ids
        )
        accs.append(correct / n * 100)

    return float(np.mean(accs)), float(np.std(accs)), accs


# ===================== Plotting =====================

def plot_four_way(baseline_results, sc_gated_results, always_rag_results,
                  random_mean=None, random_std=None, random_accs=None, n_quintiles=5):

    sorted_by_sc = sorted(baseline_results, key=lambda r: r["sc"])
    n     = len(sorted_by_sc)
    q_size = n // n_quintiles

    sg_by_id = {r["id"]: r for r in sc_gated_results}
    ar_by_id = {r["id"]: r for r in always_rag_results} if always_rag_results else {}

    b_per_q = []; sg_per_q = []; ar_per_q = []; q_labels = []

    for q in range(n_quintiles):
        s = q * q_size
        e = s + q_size if q < n_quintiles - 1 else n
        grp = sorted_by_sc[s:e]

        b_acc  = sum(r["is_correct"]         for r in grp) / len(grp) * 100
        sg_acc = sum(sg_by_id[r["id"]]["is_correct_sc_gated"]
                     for r in grp if r["id"] in sg_by_id) / len(grp) * 100
        b_per_q.append(b_acc); sg_per_q.append(sg_acc)

        if ar_by_id:
            ar_acc = sum(ar_by_id[r["id"]]["is_correct_always_rag"]
                         for r in grp if r["id"] in ar_by_id) / len(grp) * 100
            ar_per_q.append(ar_acc)

        q_labels.append(f"Q{q+1}\n({sorted_by_sc[s]['sc']:.1f}–{sorted_by_sc[e-1]['sc']:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Exp9: CoT SC-Gated RAG — Four-Way Comparison ({DS_TAG.upper()})\n"
        f"Bottom {int(args.gated_quantile*100)}% SC triggers RAG | top-{args.top_k} passages | CoT",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: quintile bars
    ax    = axes[0]
    x     = np.arange(n_quintiles)
    n_b   = 3 if ar_by_id else 2
    width = 0.7 / n_b
    offs  = np.linspace(-(n_b-1)/2, (n_b-1)/2, n_b) * width

    bars1 = ax.bar(x + offs[0], b_per_q,  width, label="No-RAG CoT",   color=BLUE,   edgecolor="k", lw=0.5)
    bars2 = ax.bar(x + offs[1], sg_per_q, width, label="SC-Gated RAG", color=GREEN,  edgecolor="k", lw=0.5)
    if ar_by_id:
        bars3 = ax.bar(x + offs[2], ar_per_q, width, label="Always-RAG", color=ORANGE, edgecolor="k", lw=0.5)

    for bar, v in zip(bars1, b_per_q):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7, color=BLUE)
    for bar, v in zip(bars2, sg_per_q):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=7, color=GREEN)
    if ar_by_id:
        for bar, v in zip(bars3, ar_per_q):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=7, color=ORANGE)

    ax.set_xticks(x); ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Accuracy by SC Quintile (CoT)")
    ax.set_ylim(0, 100); ax.legend(fontsize=9)

    # Panel 2: overall + random band
    ax = axes[1]
    overall_b  = sum(r["is_correct"]         for r in baseline_results)  / n * 100
    overall_sg = sum(r["is_correct_sc_gated"] for r in sc_gated_results) / n * 100
    conds  = ["No-RAG CoT\n(baseline)", "SC-Gated RAG\n(CoT)"]
    vals   = [overall_b, overall_sg]
    cols   = [BLUE, GREEN]

    if always_rag_results:
        overall_ar = sum(r["is_correct_always_rag"] for r in always_rag_results) / n * 100
        conds.append("Always-RAG\n(CoT)"); vals.append(overall_ar); cols.append(ORANGE)

    bars = ax.bar(conds, vals, color=cols, edgecolor="k", lw=0.5, width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    if random_mean is not None:
        ax.axhline(random_mean, color=RED, ls="--", lw=2,
                   label=f"Random-Gated mean={random_mean:.1f}%")
        ax.axhspan(random_mean - 2*random_std, random_mean + 2*random_std,
                   color=RED, alpha=0.12, label=f"±2σ ({random_std:.2f}%)")
        ax.legend(fontsize=9)

    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Overall Accuracy\n(dashed = random 20% gets RAG, 1000 trials)")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp9_four_way_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}"); plt.close(fig)

    # Histogram of random distribution
    if random_accs is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(random_accs, bins=40, color=RED, alpha=0.6, edgecolor="k", lw=0.3)
        ax2.axvline(random_mean,  color=RED,   lw=2,   ls="--", label=f"Random mean={random_mean:.1f}%")
        ax2.axvline(overall_sg,   color=GREEN, lw=2.5, ls="-",  label=f"SC-Gated={overall_sg:.1f}%")
        ax2.axvline(overall_b,    color=BLUE,  lw=2,   ls=":",  label=f"No-RAG={overall_b:.1f}%")
        pct = float(np.mean(np.array(random_accs) <= overall_sg) * 100)
        ax2.set_xlabel("Accuracy (%)"); ax2.set_ylabel("Count")
        ax2.set_title(
            f"Random-Gated RAG Distribution ({DS_TAG.upper()}, CoT, n={len(random_accs)} trials)\n"
            f"SC-Gated beats {pct:.0f}% of random splits"
        )
        ax2.legend(fontsize=9)
        fig2.tight_layout()
        out2 = os.path.join(FIGURE_DIR, f"exp9_random_dist_{DS_TAG}.png")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        logging.info(f"Saved: {out2}"); plt.close(fig2)


def plot_q1_detail(q1_baseline, q1_sg_items):
    sg_by_id = {r["id"]: r for r in q1_sg_items}

    stayed_correct = fixed = regressed = stayed_wrong = 0
    for r in q1_baseline:
        was = r["is_correct"]
        now = sg_by_id.get(r["id"], {}).get("is_correct_sc_gated", was)
        if was and now:      stayed_correct += 1
        elif not was and now: fixed += 1
        elif was and not now: regressed += 1
        else:                 stayed_wrong += 1

    n = len(q1_baseline)
    acc_before = sum(r["is_correct"] for r in q1_baseline) / n * 100
    acc_after  = (stayed_correct + fixed) / n * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Exp9: Q1 Detail — CoT RAG Impact ({DS_TAG.upper()})", fontsize=13, fontweight="bold")

    ax = axes[0]
    bars = ax.bar(["Before RAG\n(CoT baseline)", "After RAG\n(SC-Gated CoT)"],
                  [acc_before, acc_after], color=[BLUE, GREEN], edgecolor="k", lw=0.5, width=0.4)
    for bar, v in zip(bars, [acc_before, acc_after]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Q1 Accuracy (%)"); ax.set_title(f"Q1 Before vs After RAG (n={n})"); ax.set_ylim(0, 100)

    ax = axes[1]
    sizes = [fixed, stayed_correct, regressed, stayed_wrong]
    labels_pie = [f"Fixed by RAG\n({fixed})", f"Stayed correct\n({stayed_correct})",
                  f"Regressed\n({regressed})", f"Stayed wrong\n({stayed_wrong})"]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_pie, colors=[GREEN, BLUE, RED, ORANGE],
        autopct="%1.0f%%", startangle=90, pctdistance=0.75)
    for at in autotexts: at.set_fontsize(10)
    ax.set_title("Q1 Case Breakdown")

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp9_q1_detail_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}"); plt.close(fig)


# ===================== Main =====================

def main():
    # Load exp7 CoT baseline
    if not os.path.exists(EXP7_FILE):
        logging.error(f"Exp7 CoT results not found: {EXP7_FILE}")
        logging.error("Run exp7_cot_sc_vs_ppl.py first.")
        sys.exit(1)

    with open(EXP7_FILE, encoding="utf-8") as f:
        baseline_results = json.load(f)
    logging.info(f"Loaded {len(baseline_results)} CoT baseline results from {EXP7_FILE}")

    n = len(baseline_results)
    sorted_by_sc = sorted(baseline_results, key=lambda r: r["sc"])
    q1_cutoff    = int(n * args.gated_quantile)
    q1_items     = sorted_by_sc[:q1_cutoff]
    q1_ids       = {r["id"] for r in q1_items}
    sc_threshold = q1_items[-1]["sc"]

    baseline_acc = sum(r["is_correct"] for r in baseline_results) / n * 100
    q1_acc       = sum(r["is_correct"] for r in q1_items) / len(q1_items) * 100

    logging.info(f"\nDataset: {args.dataset} | Corpus: {CORPUS_NAME} | Mode: CoT")
    logging.info(f"Total questions:  {n}")
    logging.info(f"Gated quantile:   bottom {args.gated_quantile*100:.0f}% → {len(q1_items)} questions")
    logging.info(f"SC threshold:     {sc_threshold:.4f}")
    logging.info(f"Baseline acc:     {baseline_acc:.1f}%  (CoT, no RAG)")
    logging.info(f"Q1 baseline acc:  {q1_acc:.1f}%")
    logging.info("=" * 60)

    # Elasticsearch
    logging.info("Connecting to Elasticsearch...")
    es_client = Elasticsearch(["http://localhost:9200"], request_timeout=30)
    try:
        test = retrieve_passages(es_client, CORPUS_NAME, "test query", top_k=1)
        logging.info(f"Retriever OK ({len(test)} result).")
    except Exception as e:
        logging.error(f"Retriever failed: {e}"); sys.exit(1)

    # Model
    logging.info(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, device_map="cuda",
        torch_dtype=torch.bfloat16, token=HF_TOKEN, local_files_only=True)
    model.eval()
    logging.info("Model loaded.\n")

    # ======================================================
    # Condition B: SC-Gated RAG (CoT)
    # ======================================================
    logging.info("=" * 60)
    logging.info("Condition B: SC-Gated RAG (CoT)")
    logging.info("=" * 60)

    sc_gated_results = []
    for item in tqdm(baseline_results, desc="SC-Gated CoT RAG"):
        rid = item["id"]
        if rid in q1_ids:
            try:
                passages = retrieve_passages(es_client, CORPUS_NAME, item["question"], top_k=args.top_k)
            except Exception as e:
                logging.warning(f"Retrieval failed {rid}: {e}")
                passages = []

            prompt = build_cot_rag_prompt(tokenizer, item["question"], passages) if passages \
                     else build_cot_prompt(tokenizer, item["question"])
            gen    = generate_cot(model, tokenizer, prompt)
            is_correct = check_correct(gen["final_answer"], item["ground_truths"]) \
                         if gen["final_answer"] else False

            entry = {
                **item,
                "rag_triggered":       True,
                "cot_sc_gated":        gen["cot"],
                "final_answer_sc_gated": gen["final_answer"],
                "sc_sc_gated":         gen["sc"],
                "n_tokens_sc_gated":   gen["n_tokens"],
                "is_correct_sc_gated": is_correct,
                "passages": [{"title": p.get("title",""), "text": p.get("paragraph_text","")[:300]}
                             for p in passages],
            }
            logging.info(f"[RAG] {rid} | {'O' if item['is_correct'] else 'X'}→{'O' if is_correct else 'X'} "
                         f"SC={item['sc']:.2f} | {gen['final_answer'][:60]}")
        else:
            entry = {
                **item,
                "rag_triggered":         False,
                "cot_sc_gated":          item["cot"],
                "final_answer_sc_gated": item["final_answer"],
                "sc_sc_gated":           item["sc"],
                "n_tokens_sc_gated":     item["n_tokens"],
                "is_correct_sc_gated":   item["is_correct"],
                "passages":              [],
            }
        sc_gated_results.append(entry)

    # ======================================================
    # Condition C: Always-RAG (CoT)
    # ======================================================
    always_rag_results = []
    if not args.skip_always_rag:
        logging.info("\n" + "=" * 60)
        logging.info("Condition C: Always-RAG (CoT)")
        logging.info("=" * 60)

        for item in tqdm(baseline_results, desc="Always-RAG CoT"):
            rid = item["id"]
            try:
                passages = retrieve_passages(es_client, CORPUS_NAME, item["question"], top_k=args.top_k)
            except Exception as e:
                logging.warning(f"Retrieval failed {rid}: {e}")
                passages = []

            prompt = build_cot_rag_prompt(tokenizer, item["question"], passages) if passages \
                     else build_cot_prompt(tokenizer, item["question"])
            gen    = generate_cot(model, tokenizer, prompt)
            is_correct = check_correct(gen["final_answer"], item["ground_truths"]) \
                         if gen["final_answer"] else False

            always_rag_results.append({
                **item,
                "cot_always_rag":        gen["cot"],
                "final_answer_always_rag": gen["final_answer"],
                "is_correct_always_rag": is_correct,
            })

    # ======================================================
    # Summary
    # ======================================================
    logging.info("\n" + "=" * 60)
    logging.info("Results Summary")
    logging.info("=" * 60)

    overall_b  = sum(r["is_correct"]         for r in baseline_results)  / n * 100
    overall_sg = sum(r["is_correct_sc_gated"] for r in sc_gated_results) / n * 100
    gain_sg    = overall_sg - overall_b

    logging.info(f"\n  A. No-RAG CoT baseline: {overall_b:.1f}%  (n={n})")
    logging.info(f"  B. SC-Gated RAG (CoT):  {overall_sg:.1f}%  ({len(q1_ids)}/{n} = {len(q1_ids)/n*100:.0f}% retrieve)")
    logging.info(f"     Gain: {gain_sg:+.1f} pp")

    overall_ar  = None
    random_mean = random_std = random_accs = None

    if always_rag_results:
        overall_ar = sum(r["is_correct_always_rag"] for r in always_rag_results) / n * 100
        gain_ar    = overall_ar - overall_b
        eff        = (gain_sg / gain_ar * 100) if gain_ar > 0 else float("nan")
        logging.info(f"  C. Always-RAG (CoT):    {overall_ar:.1f}%  (all {n} retrieve)")
        logging.info(f"     Gain: {gain_ar:+.1f} pp")
        logging.info(f"  SC-Gated efficiency: {eff:.0f}% of Always-RAG gain at 20% cost")

        logging.info(f"\n  Computing Random-Gated baseline ({args.n_random_seeds} trials)...")
        random_mean, random_std, random_accs = compute_random_gated_acc(
            baseline_results, always_rag_results, len(q1_ids), args.n_random_seeds)
        pct = float(np.mean(np.array(random_accs) <= overall_sg) * 100)
        logging.info(f"  D. Random-Gated RAG:    {random_mean:.1f}% ± {random_std:.2f}%")
        logging.info(f"     Gain: {random_mean-overall_b:+.1f} pp")
        logging.info(f"\n  SC-Gated ({overall_sg:.1f}%) beats {pct:.0f}% of random splits"
                     f"  → {'SC adds value!' if pct > 90 else 'SC advantage not significant'}")

    # Q1 detail
    q1_baseline = [r for r in baseline_results   if r["id"] in q1_ids]
    q1_sg       = [r for r in sc_gated_results   if r["id"] in q1_ids]
    sg_by_id    = {r["id"]: r for r in q1_sg}

    fixed     = sum(1 for r in q1_baseline if not r["is_correct"] and sg_by_id.get(r["id"],{}).get("is_correct_sc_gated", False))
    regressed = sum(1 for r in q1_baseline if r["is_correct"]     and not sg_by_id.get(r["id"],{}).get("is_correct_sc_gated", True))
    q1_after  = sum(r["is_correct_sc_gated"] for r in q1_sg) / max(len(q1_sg), 1) * 100

    logging.info(f"\n  Q1 acc before: {q1_acc:.1f}% → after: {q1_after:.1f}%")
    logging.info(f"  Fixed: {fixed}  Regressed: {regressed}")

    # Per-quintile
    sg_by_id_full = {r["id"]: r for r in sc_gated_results}
    q_size = n // 5
    logging.info(f"\n  Per-quintile (No-RAG CoT → SC-Gated CoT RAG):")
    quintile_data = []
    for q in range(5):
        s, e = q * q_size, (q * q_size + q_size if q < 4 else n)
        grp  = sorted_by_sc[s:e]
        b_a  = sum(r["is_correct"] for r in grp) / len(grp) * 100
        sg_a = sum(sg_by_id_full[r["id"]]["is_correct_sc_gated"] for r in grp) / len(grp) * 100
        trig = sum(1 for r in grp if r["id"] in q1_ids)
        logging.info(f"    Q{q+1}: {b_a:.1f}% → {sg_a:.1f}%  (RAG: {trig}/{len(grp)})")
        quintile_data.append({"quintile": f"Q{q+1}", "n": len(grp),
                               "baseline_acc": round(b_a, 2), "sc_gated_acc": round(sg_a, 2),
                               "rag_triggered": trig})

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"sc_gated": sc_gated_results, "always_rag": always_rag_results},
                  f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved: {OUTPUT_FILE}")

    summary = {
        "dataset": args.dataset, "corpus": CORPUS_NAME, "mode": "cot",
        "n_questions": n, "top_k_passages": args.top_k,
        "gated_quantile": args.gated_quantile, "sc_threshold": round(sc_threshold, 4),
        "n_rag_triggered": len(q1_ids),
        "overall": {
            "no_rag_cot_baseline":  round(overall_b,  2),
            "sc_gated_rag_cot":     round(overall_sg, 2),
            "always_rag_cot":       round(overall_ar, 2) if overall_ar else None,
            "random_gated_mean":    round(random_mean, 2) if random_mean else None,
            "random_gated_std":     round(random_std, 4)  if random_std  else None,
            "gain_sc_gated":        round(gain_sg, 2),
            "gain_always_rag":      round(overall_ar - overall_b, 2) if overall_ar else None,
            "gain_random_gated":    round(random_mean - overall_b, 2) if random_mean else None,
            "sc_beats_random_pct":  round(float(np.mean(np.array(random_accs) <= overall_sg)*100), 1)
                                    if random_accs else None,
        },
        "q1_detail": {"acc_before": round(q1_acc, 2), "acc_after": round(q1_after, 2),
                      "fixed": fixed, "regressed": regressed},
        "quintiles": quintile_data,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Summary saved: {SUMMARY_FILE}")

    # Plots
    plot_four_way(baseline_results, sc_gated_results,
                  always_rag_results if not args.skip_always_rag else None,
                  random_mean=random_mean, random_std=random_std, random_accs=random_accs)
    plot_q1_detail(q1_baseline, q1_sg)
    logging.info("Done.")


if __name__ == "__main__":
    main()
