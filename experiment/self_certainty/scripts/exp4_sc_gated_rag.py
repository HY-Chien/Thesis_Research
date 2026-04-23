"""
Experiment 4 (Direction B): SC-Gated RAG — Quintile Analysis
=============================================================
Goal: Validate that whole-answer SC correlates with answer correctness,
      so it can be used as a threshold to decide when to trigger RAG.

Method:
  Phase 1 — Baseline (no RAG):
    1. Greedy generate answer for each question (direct answer, no CoT)
    2. Compute whole-answer SC and PPL
    3. Rank all questions by SC, split into 5 quintiles (Q1=lowest SC, Q5=highest)
    4. Report accuracy per quintile
    5. If Q1 accuracy << Q5 accuracy → SC is a useful gating signal

  Phase 2 — SC-Gated RAG (if Phase 1 succeeds):
    6. For questions in Q1 (lowest SC), retrieve documents and re-generate
    7. Compare: baseline accuracy vs SC-gated accuracy vs always-RAG accuracy

  This script does Phase 1 only. Phase 2 requires the retriever service.

Key insight: We use SC (not PPL) for gating because:
  - PPL has higher AUC for correct/incorrect separation
  - BUT PPL is unstable under paraphrase (60% median change)
  - SC is stable (5% median change) → reliable threshold in production

Usage:
  python exp4_sc_gated_rag.py --dataset nq --max_questions 500
  python exp4_sc_gated_rag.py --dataset nq --max_questions 200 --n_quintiles 4
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# ================= Parse Arguments =================
parser = argparse.ArgumentParser(description="SC-Gated RAG: Quintile Analysis")
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique", "squad", "trivia"])
parser.add_argument("--max_questions", type=int, default=500,
                    help="Max questions to process (default: 500)")
parser.add_argument("--n_quintiles", type=int, default=5,
                    help="Number of groups to split by SC (default: 5)")
parser.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max new tokens for generation (default: 64)")
args = parser.parse_args()

# ================= Dataset Registry =================
DATA_DIR = "/home/S113065528/Adaptive-RAG/processed_data"
DATASET_MAP = {
    "2wiki":    "2wikimultihopqa",
    "nq":       "nq",
    "hotpotqa": "hotpotqa",
    "musique":  "musique",
    "squad":    "squad",
    "trivia":   "trivia",
}

# ================= Settings =================
DATASET_NAME = DATASET_MAP[args.dataset]
DATA_FILE = os.path.join(DATA_DIR, DATASET_NAME, "test_subsampled.jsonl")
DS_TAG = args.dataset

BASE_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE = os.path.join(BASE_DIR, "results", f"exp4_{DS_TAG}_results.json")
SUMMARY_FILE = os.path.join(BASE_DIR, "results", f"exp4_{DS_TAG}_summary.json")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
LOG_FILE = os.path.join(BASE_DIR, "logs", f"exp4_{DS_TAG}.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MAX_NEW_TOKENS = args.max_new_tokens
# =============================================

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

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"


# ---------- Answer evaluation ----------
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


# ---------- Load data ----------
def load_data(filepath, max_n=None):
    data = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            question = item["question_text"]
            answers = []
            for ans_obj in item.get("answers_objects", []):
                answers.extend(ans_obj.get("spans", []))
            if question and answers:
                data.append({
                    "id": item.get("question_id", item.get("id", "")),
                    "question": question,
                    "answers": answers,
                })
            if max_n and len(data) >= max_n:
                break
    return data


# ---------- Prompt ----------
def build_prompt(tokenizer, question: str) -> str:
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


# ---------- Generation + SC/PPL ----------
@torch.no_grad()
def generate_and_score(model, tokenizer, question):
    """
    Greedy generate an answer, then compute SC and PPL on it.
    Returns dict with answer, sc, ppl, n_tokens.
    """
    V = model.config.vocab_size
    prompt_text = build_prompt(tokenizer, question)

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    # Greedy generation
    output_ids = model.generate(
        prompt_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    answer_ids = output_ids[0][prompt_len:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    answer_len = len(answer_ids)

    if answer_len == 0:
        return {"answer": "", "sc": 0.0, "ppl": float("inf"), "n_tokens": 0}

    # Re-encode to get exact tokenization (decode/re-encode can differ)
    full_ids = output_ids[:, :prompt_len + answer_len]

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    answer_logits = logits[
        :, prompt_len - 1 : prompt_len + answer_len - 1, :
    ].squeeze(0).float()

    answer_token_ids = answer_ids.to(model.device)
    log_probs_all = F.log_softmax(answer_logits, dim=-1)

    # PPL
    token_log_probs = log_probs_all.gather(
        1, answer_token_ids.unsqueeze(1)
    ).squeeze(1)
    avg_nll = -token_log_probs.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    # SC
    logprob_sum = log_probs_all.sum(dim=-1)
    V_float = float(V)
    sc_per_token = -1.0 / V_float * logprob_sum - math.log(V_float)
    sc = sc_per_token.mean().item()

    return {"answer": answer_text, "sc": sc, "ppl": ppl, "n_tokens": answer_len}


# ---------- Plotting ----------
def plot_quintile_analysis(results, n_quintiles, dataset_tag):
    """
    Main result figure: accuracy per SC quintile + comparison with PPL quintile.
    """
    # Sort by SC
    sorted_by_sc = sorted(results, key=lambda r: r["sc"])
    sorted_by_ppl = sorted(results, key=lambda r: r["ppl"])  # lower PPL = more confident

    n = len(sorted_by_sc)
    q_size = n // n_quintiles

    sc_quintile_acc = []
    ppl_quintile_acc = []
    sc_quintile_labels = []

    for q_idx in range(n_quintiles):
        start = q_idx * q_size
        end = start + q_size if q_idx < n_quintiles - 1 else n

        # SC quintile
        sc_group = sorted_by_sc[start:end]
        sc_acc = sum(1 for r in sc_group if r["is_correct"]) / len(sc_group)
        sc_quintile_acc.append(sc_acc * 100)

        sc_lo = sc_group[0]["sc"]
        sc_hi = sc_group[-1]["sc"]
        sc_quintile_labels.append(f"Q{q_idx+1}\n({sc_lo:.1f}-{sc_hi:.1f})")

        # PPL quintile (sorted by PPL ascending = lowest PPL first = most confident)
        ppl_group = sorted_by_ppl[start:end]
        ppl_acc = sum(1 for r in ppl_group if r["is_correct"]) / len(ppl_group)
        ppl_quintile_acc.append(ppl_acc * 100)

    overall_acc = sum(1 for r in results if r["is_correct"]) / len(results) * 100

    # --- Figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Exp 4: SC-Gated RAG — Quintile Analysis ({dataset_tag.upper()})\n"
        f"Can whole-answer SC identify when the model needs RAG?",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: SC quintile accuracy
    ax = axes[0]
    x = np.arange(n_quintiles)
    bars = ax.bar(x, sc_quintile_acc, color=BLUE, edgecolor="black", linewidth=0.5, width=0.6)
    ax.axhline(overall_acc, color="black", linestyle="--", linewidth=1.5,
               label=f"Overall: {overall_acc:.1f}%")
    for bar, val in zip(bars, sc_quintile_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sc_quintile_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by SC Quintile\n(Q1=lowest SC, Q5=highest SC)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    # Panel 2: PPL quintile accuracy
    ax = axes[1]
    # Build labels with PPL range for each quintile
    ppl_labels = []
    for q_idx in range(n_quintiles):
        start = q_idx * q_size
        end = start + q_size if q_idx < n_quintiles - 1 else n
        ppl_group = sorted_by_ppl[start:end]
        ppl_lo = ppl_group[0]["ppl"]
        ppl_hi = ppl_group[-1]["ppl"]
        ppl_labels.append(f"Q{q_idx+1}\n({ppl_lo:.1f}-{ppl_hi:.1f})")
    bars = ax.bar(x, ppl_quintile_acc, color=ORANGE, edgecolor="black", linewidth=0.5, width=0.6)
    ax.axhline(overall_acc, color="black", linestyle="--", linewidth=1.5,
               label=f"Overall: {overall_acc:.1f}%")
    for bar, val in zip(bars, ppl_quintile_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ppl_labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy by PPL Quintile\n(Q1=lowest PPL/most confident, Q5=highest PPL/least confident)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    # Panel 3: SC vs PPL comparison
    # SC:  Q1=lowest SC=least confident,  Q5=highest SC=most confident
    # PPL: Q1=lowest PPL=most confident,  Q5=highest PPL=least confident
    # To make comparison fair, compare SC-Q1 vs PPL-Q5 (both = least confident)
    # and SC-Q5 vs PPL-Q1 (both = most confident)
    ax = axes[2]
    width = 0.3
    labels = ["Least confident\n(SC-Q1 / PPL-Q5)", "Most confident\n(SC-Q5 / PPL-Q1)"]
    x2 = np.arange(2)

    sc_vals  = [sc_quintile_acc[0],   sc_quintile_acc[-1]]   # SC-Q1, SC-Q5
    ppl_vals = [ppl_quintile_acc[-1], ppl_quintile_acc[0]]   # PPL-Q5, PPL-Q1

    bars1 = ax.bar(x2 - width/2, sc_vals, width, label="SC", color=BLUE,
                   edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x2 + width/2, ppl_vals, width, label="PPL", color=ORANGE,
                   edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars1, sc_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=BLUE)
    for bar, val in zip(bars2, ppl_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=ORANGE)

    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("SC vs PPL: Extreme Quintiles\n(larger gap = better gating signal)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp4_quintile_analysis_{dataset_tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


def plot_sc_distribution(results, dataset_tag):
    """SC distribution colored by correctness."""
    sc_correct = [r["sc"] for r in results if r["is_correct"]]
    sc_incorrect = [r["sc"] for r in results if not r["is_correct"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sc_correct, bins=30, alpha=0.6, color=BLUE,
            label=f"Correct (n={len(sc_correct)})")
    ax.hist(sc_incorrect, bins=30, alpha=0.6, color=RED,
            label=f"Incorrect (n={len(sc_incorrect)})")
    ax.set_xlabel("Whole-answer SC (greedy)")
    ax.set_ylabel("Count")
    ax.set_title(f"SC Distribution — Correct vs Incorrect ({dataset_tag.upper()})")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp4_sc_distribution_{dataset_tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


# ---------- Main ----------
def main():
    logging.info(f"Loading model {MODEL_ID}...")
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
    logging.info("Model loaded.")
    logging.info(f"Dataset: {args.dataset} ({DATASET_NAME})")
    logging.info(f"Data file: {DATA_FILE}")
    logging.info(f"Max questions: {args.max_questions}")
    logging.info(f"N quintiles: {args.n_quintiles}")
    logging.info("=" * 60)

    # Load data
    data = load_data(DATA_FILE, args.max_questions)
    logging.info(f"Loaded {len(data)} questions\n")

    # Process each question
    results = []
    for item in tqdm(data, desc="Generating + scoring"):
        scored = generate_and_score(model, tokenizer, item["question"])
        is_correct = check_correct(scored["answer"], item["answers"]) if scored["answer"] else False

        entry = {
            "id": item["id"],
            "question": item["question"],
            "ground_truths": item["answers"],
            "answer": scored["answer"],
            "sc": scored["sc"],
            "ppl": scored["ppl"],
            "n_tokens": scored["n_tokens"],
            "is_correct": is_correct,
        }
        results.append(entry)

        logging.info(f"[{item['id']}] {'O' if is_correct else 'X'} "
                     f"SC={scored['sc']:.4f} PPL={scored['ppl']:.2f} "
                     f"A={scored['answer'][:60]}")

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"\nResults saved to {OUTPUT_FILE}")

    # ========== Analysis ==========
    n = len(results)
    overall_correct = sum(1 for r in results if r["is_correct"])
    overall_acc = overall_correct / n * 100
    logging.info(f"\nOverall accuracy: {overall_correct}/{n} = {overall_acc:.1f}%")

    # AUC
    labels = [1 if r["is_correct"] else 0 for r in results]
    sc_scores = [r["sc"] for r in results]
    ppl_scores = [-r["ppl"] for r in results]  # negate: lower PPL = more confident

    if len(set(labels)) > 1:
        auc_sc = roc_auc_score(labels, sc_scores)
        auc_ppl = roc_auc_score(labels, ppl_scores)
        logging.info(f"AUC — SC: {auc_sc:.4f}, PPL: {auc_ppl:.4f}")
    else:
        auc_sc = auc_ppl = None
        logging.info("AUC: cannot compute (all same label)")

    # Quintile analysis
    sorted_by_sc = sorted(results, key=lambda r: r["sc"])
    q_size = n // args.n_quintiles
    quintile_summary = []

    logging.info(f"\n{'='*60}")
    logging.info(f"Quintile Analysis (n_quintiles={args.n_quintiles})")
    logging.info(f"{'='*60}")

    for q_idx in range(args.n_quintiles):
        start = q_idx * q_size
        end = start + q_size if q_idx < args.n_quintiles - 1 else n
        group = sorted_by_sc[start:end]

        acc = sum(1 for r in group if r["is_correct"]) / len(group) * 100
        mean_sc = np.mean([r["sc"] for r in group])
        sc_range = (group[0]["sc"], group[-1]["sc"])

        q_info = {
            "quintile": f"Q{q_idx+1}",
            "n": len(group),
            "accuracy": round(acc, 2),
            "mean_sc": round(mean_sc, 4),
            "sc_range": [round(sc_range[0], 4), round(sc_range[1], 4)],
        }
        quintile_summary.append(q_info)

        logging.info(f"  Q{q_idx+1}: SC [{sc_range[0]:.2f}, {sc_range[1]:.2f}] "
                     f"| n={len(group)} | acc={acc:.1f}% | mean_sc={mean_sc:.2f}")

    # Gap between Q1 and Q5
    q1_acc = quintile_summary[0]["accuracy"]
    q5_acc = quintile_summary[-1]["accuracy"]
    gap = q5_acc - q1_acc
    logging.info(f"\n  Q1 accuracy: {q1_acc:.1f}%")
    logging.info(f"  Q5 accuracy: {q5_acc:.1f}%")
    logging.info(f"  Gap (Q5 - Q1): {gap:.1f} percentage points")

    if gap > 15:
        logging.info("  --> STRONG signal: SC can meaningfully distinguish confident/unconfident answers")
        logging.info("  --> SC-gated RAG is viable: trigger RAG for Q1, skip for Q5")
    elif gap > 5:
        logging.info("  --> MODERATE signal: some correlation, may need tuning")
    else:
        logging.info("  --> WEAK signal: SC does not clearly separate confident/unconfident")

    # Potential RAG benefit
    q1_n = quintile_summary[0]["n"]
    q1_wrong = int(q1_n * (1 - quintile_summary[0]["accuracy"] / 100))
    logging.info(f"\n  If RAG fixes all Q1 errors: +{q1_wrong} correct answers")
    logging.info(f"  New accuracy would be: {(overall_correct + q1_wrong) / n * 100:.1f}%")

    # Save summary
    summary = {
        "dataset": args.dataset,
        "n_questions": n,
        "overall_accuracy": round(overall_acc, 2),
        "auc_sc": round(auc_sc, 4) if auc_sc else None,
        "auc_ppl": round(auc_ppl, 4) if auc_ppl else None,
        "quintiles": quintile_summary,
        "q1_q5_gap": round(gap, 2),
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nSummary saved to {SUMMARY_FILE}")

    # Plot
    plot_quintile_analysis(results, args.n_quintiles, DS_TAG)
    plot_sc_distribution(results, DS_TAG)
    logging.info("Done.")


if __name__ == "__main__":
    main()
