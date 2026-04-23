"""
Exp8: SC-Gated RAG — Phase 2 (End-to-End Comparison)
=====================================================
Phase 2 of Exp4. Takes the exp4 baseline results (SC scores for 500 NQ questions)
and re-generates Q1 (lowest SC = most uncertain) questions with retrieved context.

Four-way comparison:
  A. No-RAG baseline     — use answers from exp4 directly (no re-generation)
  B. SC-Gated RAG        — Q1 questions get RAG, Q2-Q5 keep baseline answers
  C. Always-RAG          — all 500 questions get RAG re-generation
  D. Random-Gated RAG    — randomly select same number of questions as Q1, give them RAG
                           averaged over n_random_seeds trials (computed from C, no extra GPU)

Key thesis payoff:
  - If B > A: SC correctly identifies when RAG helps → valid gating signal
  - If B > D: SC selects BETTER questions than random → SC adds value beyond budget
  - If B ≈ C: SC-gated RAG is efficient (same gain, only 20% of questions retrieve)
  - The critical comparison is B vs D: does SC beat random selection?

Retrieval: Elasticsearch wiki index (BM25), top-3 paragraphs
Model: meta-llama/Llama-3.1-8B-Instruct (same as exp4)
Corpus: wiki (for NQ)

Usage:
  conda activate adaptive_rag
  cd /home/S113065528/Adaptive-RAG/experiment/self_certainty/scripts
  python exp8_sc_gated_rag_phase2.py --dataset nq
  python exp8_sc_gated_rag_phase2.py --dataset nq --top_k 5 --gated_quantile 0.2
"""

import os
import sys
# Add retriever_server to path for direct import (no need to start FastAPI serve.py)
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

# ================= Parse Arguments =================
parser = argparse.ArgumentParser(description="Exp8: SC-Gated RAG Phase 2")
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique"])
parser.add_argument("--top_k", type=int, default=3,
                    help="Number of retrieved paragraphs to include in context (default: 3)")
parser.add_argument("--gated_quantile", type=float, default=0.2,
                    help="Bottom quantile fraction to trigger RAG (default: 0.2 = bottom 20%% = Q1)")
parser.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max new tokens for generation (default: 64)")
parser.add_argument("--skip_always_rag", action="store_true",
                    help="Skip the Always-RAG condition (also disables Random-Gated baseline)")
parser.add_argument("--n_random_seeds", type=int, default=1000,
                    help="Number of random splits for Random-Gated baseline (default: 1000)")
args = parser.parse_args()

# ================= Dataset Registry =================
DATASET_MAP = {
    "nq":       ("nq",               "wiki"),
    "2wiki":    ("2wikimultihopqa",  "2wikimultihopqa"),
    "hotpotqa": ("hotpotqa",         "hotpotqa"),
    "musique":  ("musique",          "musique"),
}
DATA_DIR = "/home/S113065528/Adaptive-RAG/processed_data"
BASE_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ================= Paths =================
DATASET_NAME, CORPUS_NAME = DATASET_MAP[args.dataset]
DS_TAG = args.dataset
EXP4_RESULTS_FILE = os.path.join(BASE_DIR, "results", f"exp4_{DS_TAG}_results.json")
OUTPUT_FILE       = os.path.join(BASE_DIR, "results", f"exp8_{DS_TAG}_results.json")
SUMMARY_FILE      = os.path.join(BASE_DIR, "results", f"exp8_{DS_TAG}_summary.json")
FIGURE_DIR        = os.path.join(BASE_DIR, "figures")
LOG_FILE          = os.path.join(BASE_DIR, "logs", f"exp8_{DS_TAG}.log")

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


# ===================== Answer Evaluation =====================

def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())


def compute_f1(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num = sum(common.values())
    if not pred_toks or not gold_toks:
        return int(pred_toks == gold_toks)
    if num == 0:
        return 0
    p = num / len(pred_toks)
    r = num / len(gold_toks)
    return 2 * p * r / (p + r)


def check_correct(prediction, ground_truths):
    pred_n = normalize_answer(prediction)
    for gt in ground_truths:
        if pred_n == normalize_answer(gt):
            return True
        if compute_f1(prediction, gt) >= 0.5:
            return True
    return False


# ===================== Prompt Builders =====================

def build_norag_prompt(tokenizer, question: str) -> str:
    """Same as exp4: no context, direct answer."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question with a short phrase. "
                "Do not explain. Output only the answer."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_rag_prompt(tokenizer, question: str, passages: list) -> str:
    """Include retrieved passages in the system context."""
    if not passages:
        return build_norag_prompt(tokenizer, question)

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
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the following Wikipedia passages to answer "
                "the question with a short phrase. Do not explain. Output only the answer.\n\n"
                + context_str
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ===================== Generation =====================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt_text: str) -> str:
    """Greedy generate, return decoded answer text."""
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
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


# ===================== Plotting =====================

def compute_random_gated_acc(baseline_results, always_rag_results, n_triggered, n_seeds=1000):
    """
    Simulate Random-Gated RAG: randomly select n_triggered questions to receive RAG,
    rest keep baseline answer. Average accuracy over n_seeds trials.

    Uses always_rag_results as the source of RAG answers (no extra GPU needed).
    Returns (mean_acc, std_acc, all_accs).
    """
    n = len(baseline_results)
    ar_by_id = {r["id"]: r for r in always_rag_results}
    all_ids  = [r["id"] for r in baseline_results]
    b_by_id  = {r["id"]: r for r in baseline_results}

    rng = np.random.default_rng(42)
    accs = []
    for _ in range(n_seeds):
        chosen = set(rng.choice(all_ids, size=n_triggered, replace=False).tolist())
        correct = 0
        for rid in all_ids:
            if rid in chosen:
                correct += ar_by_id[rid]["is_correct_always_rag"]
            else:
                correct += b_by_id[rid]["is_correct"]
        accs.append(correct / n * 100)

    return float(np.mean(accs)), float(np.std(accs)), accs


def plot_four_way(baseline_results, sc_gated_results, always_rag_results,
                  random_mean=None, random_std=None, random_accs=None, n_quintiles=5):
    """
    Four-way comparison figure:
      Panel 1: per-quintile accuracy (baseline / SC-gated / always-RAG)
      Panel 2: overall accuracy bar chart with random baseline distribution
    """
    sorted_by_sc = sorted(baseline_results, key=lambda r: r["sc"])
    n = len(sorted_by_sc)
    q_size = n // n_quintiles

    sc_gated_by_id   = {r["id"]: r for r in sc_gated_results}
    always_rag_by_id = {r["id"]: r for r in always_rag_results} if always_rag_results else {}

    baseline_acc_per_q   = []
    sc_gated_acc_per_q   = []
    always_rag_acc_per_q = []
    q_labels = []

    for q_idx in range(n_quintiles):
        start = q_idx * q_size
        end   = start + q_size if q_idx < n_quintiles - 1 else n
        group_ids      = [r["id"] for r in sorted_by_sc[start:end]]
        group_baseline = sorted_by_sc[start:end]

        b_acc = sum(1 for r in group_baseline if r["is_correct"]) / len(group_baseline) * 100
        baseline_acc_per_q.append(b_acc)

        sg_group = [sc_gated_by_id[i] for i in group_ids if i in sc_gated_by_id]
        sg_acc   = sum(1 for r in sg_group if r["is_correct_sc_gated"]) / max(len(sg_group), 1) * 100
        sc_gated_acc_per_q.append(sg_acc)

        if always_rag_by_id:
            ar_group = [always_rag_by_id[i] for i in group_ids if i in always_rag_by_id]
            ar_acc   = sum(1 for r in ar_group if r["is_correct_always_rag"]) / max(len(ar_group), 1) * 100
            always_rag_acc_per_q.append(ar_acc)

        sc_lo = sorted_by_sc[start]["sc"]
        sc_hi = sorted_by_sc[end - 1]["sc"]
        q_labels.append(f"Q{q_idx+1}\n(SC {sc_lo:.1f}–{sc_hi:.1f})")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Exp8: SC-Gated RAG — Four-Way Comparison ({DS_TAG.upper()})\n"
        f"Bottom {int(args.gated_quantile*100)}% SC triggers RAG | top-{args.top_k} passages",
        fontsize=13, fontweight="bold",
    )

    # ---------- Panel 1: per-quintile ----------
    ax   = axes[0]
    x    = np.arange(n_quintiles)
    has_ar = bool(always_rag_by_id)
    n_bars = 3 if has_ar else 2
    width  = 0.7 / n_bars

    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    bars1 = ax.bar(x + offsets[0], baseline_acc_per_q,  width, label="No-RAG (baseline)", color=BLUE,   edgecolor="k", linewidth=0.5)
    bars2 = ax.bar(x + offsets[1], sc_gated_acc_per_q,  width, label="SC-Gated RAG",      color=GREEN,  edgecolor="k", linewidth=0.5)
    if has_ar:
        bars3 = ax.bar(x + offsets[2], always_rag_acc_per_q, width, label="Always-RAG",   color=ORANGE, edgecolor="k", linewidth=0.5)

    for bar, val in zip(bars1, baseline_acc_per_q):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color=BLUE)
    for bar, val in zip(bars2, sc_gated_acc_per_q):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color=GREEN)
    if has_ar:
        for bar, val in zip(bars3, always_rag_acc_per_q):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.0f}%", ha="center", va="bottom", fontsize=7, color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(q_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by SC Quintile")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    # ---------- Panel 2: overall bar + random distribution ----------
    ax = axes[1]

    overall_b  = sum(1 for r in baseline_results  if r["is_correct"])          / n * 100
    overall_sg = sum(1 for r in sc_gated_results  if r["is_correct_sc_gated"]) / n * 100
    conditions = ["No-RAG\n(baseline)", "SC-Gated RAG"]
    values     = [overall_b, overall_sg]
    bar_colors = [BLUE, GREEN]

    if always_rag_results:
        overall_ar = sum(1 for r in always_rag_results if r["is_correct_always_rag"]) / n * 100
        conditions.append("Always-RAG")
        values.append(overall_ar)
        bar_colors.append(ORANGE)

    x2   = np.arange(len(conditions))
    bars = ax.bar(x2, values, color=bar_colors, edgecolor="k", linewidth=0.5, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Overlay random baseline as horizontal band (mean ± 2 std)
    if random_mean is not None and random_std is not None:
        ax.axhline(random_mean, color=RED, linestyle="--", linewidth=2,
                   label=f"Random-Gated mean={random_mean:.1f}%")
        ax.axhspan(random_mean - 2 * random_std, random_mean + 2 * random_std,
                   color=RED, alpha=0.12, label=f"±2σ ({random_std:.2f}%)")
        ax.legend(fontsize=9)

    ax.set_xticks(x2)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("Overall Accuracy\n(dashed = random 20% gets RAG, averaged over 1000 trials)")
    ax.set_ylim(0, 100)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp8_four_way_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved figure: {out}")
    plt.close(fig)

    # ---------- Bonus: random distribution histogram ----------
    if random_accs is not None:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(random_accs, bins=40, color=RED, alpha=0.6, edgecolor="k", linewidth=0.3)
        ax2.axvline(random_mean, color=RED,   lw=2,   linestyle="--", label=f"Random mean={random_mean:.1f}%")
        ax2.axvline(overall_sg,  color=GREEN, lw=2.5, linestyle="-",  label=f"SC-Gated={overall_sg:.1f}%")
        ax2.axvline(overall_b,   color=BLUE,  lw=2,   linestyle=":",  label=f"No-RAG={overall_b:.1f}%")
        percentile = float(np.mean(np.array(random_accs) <= overall_sg) * 100)
        ax2.set_xlabel("Accuracy (%)")
        ax2.set_ylabel("Count")
        ax2.set_title(
            f"Random-Gated RAG Distribution (n={len(random_accs)} trials, {DS_TAG.upper()})\n"
            f"SC-Gated beats {percentile:.0f}% of random splits"
        )
        ax2.legend(fontsize=9)
        fig2.tight_layout()
        out2 = os.path.join(FIGURE_DIR, f"exp8_random_dist_{DS_TAG}.png")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        logging.info(f"Saved figure: {out2}")
        plt.close(fig2)


def plot_q1_detail(q1_baseline, q1_rag_items):
    """
    For Q1 only: side-by-side accuracy before vs after RAG.
    Shows case breakdown: stayed wrong / fixed / regressed.
    """
    rag_by_id = {r["id"]: r for r in q1_rag_items}

    stayed_correct = 0
    fixed          = 0  # wrong → correct (RAG helped)
    regressed      = 0  # correct → wrong (RAG hurt)
    stayed_wrong   = 0

    for r in q1_baseline:
        rid = r["id"]
        was_correct = r["is_correct"]
        now_correct = rag_by_id[rid]["is_correct_sc_gated"] if rid in rag_by_id else was_correct

        if was_correct and now_correct:
            stayed_correct += 1
        elif not was_correct and now_correct:
            fixed += 1
        elif was_correct and not now_correct:
            regressed += 1
        else:
            stayed_wrong += 1

    n = len(q1_baseline)
    acc_before = sum(1 for r in q1_baseline if r["is_correct"]) / n * 100
    acc_after  = (stayed_correct + fixed) / n * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Exp8: Q1 Detail — RAG Impact on Most Uncertain Questions ({DS_TAG.upper()})",
                 fontsize=13, fontweight="bold")

    # Before vs after
    ax = axes[0]
    bars = ax.bar(["Before RAG\n(baseline)", "After RAG\n(SC-gated)"],
                  [acc_before, acc_after], color=[BLUE, GREEN], edgecolor="k", linewidth=0.5, width=0.4)
    for bar, val in zip(bars, [acc_before, acc_after]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.set_ylabel("Q1 Accuracy (%)")
    ax.set_title(f"Q1 Accuracy Before vs After RAG (n={n})")
    ax.set_ylim(0, 100)

    # Breakdown
    ax = axes[1]
    labels_pie = [
        f"Fixed by RAG\n({fixed})",
        f"Stayed correct\n({stayed_correct})",
        f"Regressed\n({regressed})",
        f"Stayed wrong\n({stayed_wrong})",
    ]
    sizes  = [fixed, stayed_correct, regressed, stayed_wrong]
    colors_pie = [GREEN, BLUE, RED, ORANGE]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_pie, colors=colors_pie,
        autopct="%1.0f%%", startangle=90, pctdistance=0.75
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Q1 Case Breakdown")

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp8_q1_detail_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved figure: {out}")
    plt.close(fig)


# ===================== Retrieval (wiki-compatible) =====================

# The "wiki" index in this deployment uses a non-standard schema: field "txt"
# instead of "paragraph_text". All other indices (2wikimultihopqa, hotpotqa,
# musique) use the standard schema expected by ElasticsearchRetriever.
# We handle both cases here.

WIKI_STYLE_CORPORA = {"wiki"}  # corpora that use the non-standard "txt" field


def retrieve_passages(es_client, corpus_name: str, query_text: str, top_k: int) -> list:
    """
    Retrieve top-k passages for a query.
    Handles both the standard schema (paragraph_text) and the wiki schema (txt).
    Returns list of dicts with keys: title, paragraph_text.
    """
    if corpus_name in WIKI_STYLE_CORPORA:
        # wiki index: field is "txt", no is_abstract filter
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
        # Standard schema: use ElasticsearchRetriever
        retriever = ElasticsearchRetriever(host="localhost", port=9200)
        results = retriever.retrieve_paragraphs(
            corpus_name=corpus_name,
            query_text=query_text,
            max_hits_count=top_k,
        )
        return results


# ===================== Main =====================

def main():
    # ---------- Load exp4 baseline ----------
    if not os.path.exists(EXP4_RESULTS_FILE):
        logging.error(f"Exp4 results not found: {EXP4_RESULTS_FILE}")
        logging.error("Run exp4_sc_gated_rag.py first to generate Phase 1 baseline.")
        sys.exit(1)

    with open(EXP4_RESULTS_FILE, encoding="utf-8") as f:
        baseline_results = json.load(f)
    logging.info(f"Loaded {len(baseline_results)} baseline results from {EXP4_RESULTS_FILE}")

    # ---------- Identify Q1 (bottom quantile by SC) ----------
    n = len(baseline_results)
    sorted_by_sc = sorted(baseline_results, key=lambda r: r["sc"])
    q1_cutoff    = int(n * args.gated_quantile)
    q1_items     = sorted_by_sc[:q1_cutoff]
    q1_ids       = {r["id"] for r in q1_items}

    baseline_acc = sum(1 for r in baseline_results if r["is_correct"]) / n * 100
    q1_acc       = sum(1 for r in q1_items        if r["is_correct"]) / len(q1_items) * 100
    sc_threshold = q1_items[-1]["sc"]

    logging.info(f"\nDataset: {args.dataset} | Corpus: {CORPUS_NAME}")
    logging.info(f"Total questions: {n}")
    logging.info(f"Gated quantile:  bottom {args.gated_quantile*100:.0f}% → {len(q1_items)} questions will retrieve")
    logging.info(f"SC threshold:    {sc_threshold:.4f} (all SC ≤ this get RAG)")
    logging.info(f"Baseline acc:    {baseline_acc:.1f}%")
    logging.info(f"Q1 baseline acc: {q1_acc:.1f}%")
    logging.info("=" * 60)

    # ---------- Set up retriever ----------
    logging.info("Connecting to Elasticsearch...")
    es_client = Elasticsearch(["http://localhost:9200"], request_timeout=30)
    # Quick connectivity test
    try:
        test_passages = retrieve_passages(es_client, CORPUS_NAME, "test query", top_k=1)
        logging.info(f"Retriever OK. Test query returned {len(test_passages)} passage(s).")
    except Exception as e:
        logging.error(f"Retriever connection failed: {e}")
        sys.exit(1)

    # ---------- Load model ----------
    logging.info(f"\nLoading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
        local_files_only=True,
    )
    model.eval()
    logging.info("Model loaded.\n")

    # ======================================================
    # Condition B: SC-Gated RAG
    #   Q1 → retrieve + re-generate
    #   Q2-Q5 → keep baseline answer
    # ======================================================
    logging.info("=" * 60)
    logging.info("Condition B: SC-Gated RAG")
    logging.info("=" * 60)

    sc_gated_results = []
    baseline_by_id   = {r["id"]: r for r in baseline_results}

    for item in tqdm(baseline_results, desc="SC-Gated RAG"):
        rid = item["id"]
        if rid in q1_ids:
            # Retrieve and re-generate
            try:
                passages = retrieve_passages(es_client, CORPUS_NAME, item["question"], top_k=args.top_k)
            except Exception as e:
                logging.warning(f"Retrieval failed for {rid}: {e}. Using baseline answer.")
                passages = []

            if passages:
                prompt = build_rag_prompt(tokenizer, item["question"], passages)
                rag_answer = generate_answer(model, tokenizer, prompt)
            else:
                rag_answer = item["answer"]  # fallback to baseline

            is_correct_sg = check_correct(rag_answer, item["ground_truths"]) if rag_answer else False

            entry = {
                **item,
                "rag_triggered":      True,
                "answer_sc_gated":    rag_answer,
                "is_correct_sc_gated": is_correct_sg,
                "passages":           [{"title": p.get("title",""), "text": p.get("paragraph_text","")[:300]} for p in passages],
            }
            logging.info(
                f"[RAG] {rid} | was {'O' if item['is_correct'] else 'X'} → now {'O' if is_correct_sg else 'X'} "
                f"| SC={item['sc']:.2f} | A={rag_answer[:60]}"
            )
        else:
            # Keep baseline answer
            entry = {
                **item,
                "rag_triggered":       False,
                "answer_sc_gated":     item["answer"],
                "is_correct_sc_gated": item["is_correct"],
                "passages":            [],
            }

        sc_gated_results.append(entry)

    # ======================================================
    # Condition C: Always-RAG (optional)
    # ======================================================
    always_rag_results = []
    if not args.skip_always_rag:
        logging.info("\n" + "=" * 60)
        logging.info("Condition C: Always-RAG")
        logging.info("=" * 60)

        for item in tqdm(baseline_results, desc="Always-RAG"):
            rid = item["id"]
            try:
                passages = retrieve_passages(es_client, CORPUS_NAME, item["question"], top_k=args.top_k)
            except Exception as e:
                logging.warning(f"Retrieval failed for {rid}: {e}. Using baseline answer.")
                passages = []

            if passages:
                prompt = build_rag_prompt(tokenizer, item["question"], passages)
                rag_answer = generate_answer(model, tokenizer, prompt)
            else:
                rag_answer = item["answer"]

            is_correct_ar = check_correct(rag_answer, item["ground_truths"]) if rag_answer else False

            entry = {
                **item,
                "answer_always_rag":    rag_answer,
                "is_correct_always_rag": is_correct_ar,
            }
            always_rag_results.append(entry)

    # ======================================================
    # Summary & Analysis
    # ======================================================
    logging.info("\n" + "=" * 60)
    logging.info("Results Summary")
    logging.info("=" * 60)

    # Overall accuracies
    overall_baseline = sum(1 for r in baseline_results  if r["is_correct"])          / n * 100
    overall_sg       = sum(1 for r in sc_gated_results  if r["is_correct_sc_gated"]) / len(sc_gated_results) * 100

    logging.info(f"\n  A. No-RAG baseline:  {overall_baseline:.1f}%  (n={n})")
    logging.info(f"  B. SC-Gated RAG:     {overall_sg:.1f}%  (retrieves for {len(q1_ids)}/{n} = {len(q1_ids)/n*100:.0f}% of questions)")

    gain_sg = overall_sg - overall_baseline
    logging.info(f"     Gain vs baseline: {gain_sg:+.1f} pp")

    overall_ar   = None
    random_mean  = None
    random_std   = None
    random_accs  = None
    if always_rag_results:
        overall_ar = sum(1 for r in always_rag_results if r["is_correct_always_rag"]) / len(always_rag_results) * 100
        gain_ar = overall_ar - overall_baseline
        logging.info(f"  C. Always-RAG:       {overall_ar:.1f}%  (retrieves for all {n})")
        logging.info(f"     Gain vs baseline: {gain_ar:+.1f} pp")
        efficiency = (gain_sg / gain_ar * 100) if gain_ar > 0 else float("nan")
        logging.info(f"\n  SC-Gated efficiency: {efficiency:.0f}% of Always-RAG gain with only {len(q1_ids)/n*100:.0f}% of retrieval cost")

        # Compute Random-Gated baseline (simulated from Always-RAG, no extra GPU)
        logging.info(f"\n  Computing Random-Gated baseline ({args.n_random_seeds} random trials)...")
        random_mean, random_std, random_accs = compute_random_gated_acc(
            baseline_results, always_rag_results,
            n_triggered=len(q1_ids),
            n_seeds=args.n_random_seeds,
        )
        gain_random = random_mean - overall_baseline
        percentile  = float(np.mean(np.array(random_accs) <= overall_sg) * 100)
        logging.info(f"  D. Random-Gated RAG: {random_mean:.1f}% ± {random_std:.2f}%  "
                     f"(same budget: {len(q1_ids)} questions get RAG)")
        logging.info(f"     Gain vs baseline: {gain_random:+.1f} pp")
        logging.info(f"\n  SC-Gated ({overall_sg:.1f}%) beats {percentile:.0f}% of random splits"
                     f"  → {'SC adds value beyond budget!' if percentile > 90 else 'SC advantage not significant'}")

    # Q1 specific: how many fixed vs regressed
    q1_baseline = [r for r in baseline_results if r["id"] in q1_ids]
    q1_sg       = [r for r in sc_gated_results if r["id"] in q1_ids]
    sg_by_id    = {r["id"]: r for r in q1_sg}

    fixed      = sum(1 for r in q1_baseline if not r["is_correct"] and sg_by_id.get(r["id"], {}).get("is_correct_sc_gated", False))
    regressed  = sum(1 for r in q1_baseline if r["is_correct"]     and not sg_by_id.get(r["id"], {}).get("is_correct_sc_gated", True))
    q1_acc_after = sum(1 for r in q1_sg if r["is_correct_sc_gated"]) / max(len(q1_sg), 1) * 100

    logging.info(f"\n  Q1 accuracy before RAG: {q1_acc:.1f}%")
    logging.info(f"  Q1 accuracy after RAG:  {q1_acc_after:.1f}%")
    logging.info(f"  Fixed by RAG:           {fixed}")
    logging.info(f"  Regressed by RAG:       {regressed}")

    # Quintile breakdown
    sorted_by_sc = sorted(baseline_results, key=lambda r: r["sc"])
    q_size = n // 5
    sg_by_id_full = {r["id"]: r for r in sc_gated_results}

    logging.info(f"\n  Per-quintile accuracy (No-RAG → SC-Gated RAG):")
    quintile_data = []
    for q_idx in range(5):
        start = q_idx * q_size
        end   = start + q_size if q_idx < 4 else n
        group = sorted_by_sc[start:end]

        b_acc  = sum(1 for r in group if r["is_correct"]) / len(group) * 100
        sg_acc = sum(1 for r in group if sg_by_id_full.get(r["id"], {}).get("is_correct_sc_gated", r["is_correct"])) / len(group) * 100
        triggered = sum(1 for r in group if r["id"] in q1_ids)

        logging.info(f"    Q{q_idx+1}: {b_acc:.1f}% → {sg_acc:.1f}%  (RAG triggered: {triggered}/{len(group)})")
        quintile_data.append({
            "quintile": f"Q{q_idx+1}",
            "n": len(group),
            "baseline_acc": round(b_acc, 2),
            "sc_gated_acc": round(sg_acc, 2),
            "rag_triggered": triggered,
        })

    # ======================================================
    # Save results
    # ======================================================
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "sc_gated":   sc_gated_results,
            "always_rag": always_rag_results,
        }, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved to {OUTPUT_FILE}")

    summary = {
        "dataset":           args.dataset,
        "corpus":            CORPUS_NAME,
        "n_questions":       n,
        "top_k_passages":    args.top_k,
        "gated_quantile":    args.gated_quantile,
        "sc_threshold":      round(sc_threshold, 4),
        "n_rag_triggered":   len(q1_ids),
        "overall": {
            "no_rag_baseline":    round(overall_baseline, 2),
            "sc_gated_rag":       round(overall_sg, 2),
            "always_rag":         round(overall_ar, 2) if overall_ar is not None else None,
            "random_gated_mean":  round(random_mean, 2) if random_mean is not None else None,
            "random_gated_std":   round(random_std, 4)  if random_std  is not None else None,
            "gain_sc_gated":      round(overall_sg - overall_baseline, 2),
            "gain_always_rag":    round(overall_ar - overall_baseline, 2) if overall_ar is not None else None,
            "gain_random_gated":  round(random_mean - overall_baseline, 2) if random_mean is not None else None,
            "sc_beats_random_pct": round(float(np.mean(np.array(random_accs) <= overall_sg) * 100), 1)
                                   if random_accs is not None else None,
        },
        "q1_detail": {
            "acc_before": round(q1_acc, 2),
            "acc_after":  round(q1_acc_after, 2),
            "fixed":      fixed,
            "regressed":  regressed,
        },
        "quintiles": quintile_data,
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"Summary saved to {SUMMARY_FILE}")

    # ======================================================
    # Plots
    # ======================================================
    plot_four_way(baseline_results, sc_gated_results,
                  always_rag_results if not args.skip_always_rag else None,
                  random_mean=random_mean, random_std=random_std, random_accs=random_accs)
    plot_q1_detail(q1_baseline, q1_sg)

    logging.info("\nDone.")


if __name__ == "__main__":
    main()
