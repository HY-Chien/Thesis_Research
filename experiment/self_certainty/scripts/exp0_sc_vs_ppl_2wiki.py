"""
Experiment 0: SC vs PPL — Best-of-N with Borda Voting
=====================================================
Faithful reproduction of the SC paper's methodology:
  1. For each question, sample N=8 candidates via temperature sampling (CoT)
  2. Compute SC and PPL on each candidate's FULL response
  3. Use Borda voting (p=0.3) with SC ranking to select the best answer
  4. Compare: SC-Borda vs PPL-Borda vs Majority Vote vs Greedy vs Random vs Oracle
  5. Save per-token SC/PPL trajectories for all candidates
  6. Plot distribution + trajectory figures

Supported datasets: 2wiki, nq, hotpotqa, musique, squad, trivia

Usage:
  python exp0_sc_vs_ppl_2wiki.py --dataset 2wiki
  python exp0_sc_vs_ppl_2wiki.py --dataset nq --max_questions 50
  python exp0_sc_vs_ppl_2wiki.py --dataset hotpotqa --n_candidates 16
"""

import argparse
import json
import logging
import math
import os
import random
import re
import string
import collections

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ================= Parse Arguments =================
parser = argparse.ArgumentParser(description="SC vs PPL Best-of-N experiment")
parser.add_argument("--dataset", type=str, default="2wiki",
                    choices=list(DATASET_MAP.keys()),
                    help="Dataset to use (default: 2wiki)")
parser.add_argument("--max_questions", type=int, default=None,
                    help="Max questions to process (default: all)")
parser.add_argument("--n_candidates", type=int, default=8,
                    help="Number of candidates per question (default: 8)")
parser.add_argument("--borda_p", type=float, default=0.3,
                    help="Borda exponent (default: 0.3)")
parser.add_argument("--temperature", type=float, default=0.6,
                    help="Sampling temperature (default: 0.6)")
parser.add_argument("--max_new_tokens", type=int, default=256,
                    help="Max new tokens for generation (default: 256)")
args = parser.parse_args()

# ================= Settings =================
DATASET_NAME = DATASET_MAP[args.dataset]
DATA_FILE = os.path.join(DATA_DIR, DATASET_NAME, "test_subsampled.jsonl")
DS_TAG = args.dataset  # short name for file outputs

OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"exp0_{DS_TAG}_results.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, f"exp0_{DS_TAG}_summary.json")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_FILE = os.path.join(OUTPUT_DIR, f"exp0_{DS_TAG}.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

N_CANDIDATES = args.n_candidates
BORDA_P = args.borda_p
TEMPERATURE = args.temperature
TOP_P = 0.9
MAX_NEW_TOKENS = args.max_new_tokens
MAX_QUESTIONS = args.max_questions
# =============================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# ---------- Answer evaluation ----------
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if not gold_toks or not pred_toks:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def check_correct(prediction, ground_truths):
    em = max_over_ground_truths(compute_exact, prediction, ground_truths)
    f1 = max_over_ground_truths(compute_f1, prediction, ground_truths)
    return em == 1 or f1 >= 0.5


# ---------- Extract final answer from CoT response ----------
def extract_final_answer(response):
    patterns = [
        r"[Aa]nswer\s*:\s*(.+?)(?:\n|$)",
        r"[Ff]inal [Aa]nswer\s*:\s*(.+?)(?:\n|$)",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s+(.+?)(?:\.|,|\n|$)",
        r"[Ss]o\s+the\s+answer\s+is\s+(.+?)(?:\.|,|\n|$)",
        r"[Tt]herefore,?\s+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|,|\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response)
        if m:
            answer = m.group(1).strip().rstrip(".")
            answer = answer.split("\n")[0].strip()
            if answer:
                return answer
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["Answer:", "answer:", "So,", "Therefore,", "Thus,"]:
            if last.startswith(prefix):
                last = last[len(prefix):].strip()
        return last
    return response.strip()


# ---------- Load data ----------
def load_data(filepath):
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
    return data


# ---------- Prompts ----------
def build_cot_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant. "
                "Think through the question step by step, then provide the final answer.\n"
                "Format your response as:\n"
                "Reasoning: <your step-by-step thinking>\n"
                "Answer: <short final answer>"
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def build_greedy_prompt(tokenizer, question: str) -> str:
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


# ---------- SC and PPL computation (with per-token trajectories) ----------
@torch.no_grad()
def compute_sc_and_ppl(model, tokenizer, prompt_text, response_text):
    """
    Compute Self-Certainty and PPL on the FULL generated response.
    Returns: (sc, ppl, n_tokens, sc_per_token_list, ppl_per_token_list)
    """
    if not response_text or not response_text.strip():
        return 0.0, float("inf"), 0, [], []

    V = model.config.vocab_size

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    prompt_len = prompt_ids.shape[1]

    response_ids = tokenizer(
        response_text, return_tensors="pt", add_special_tokens=False
    ).input_ids
    response_len = response_ids.shape[1]

    if response_len == 0:
        return 0.0, float("inf"), 0, [], []

    full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(model.device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits

    response_logits = logits[
        :, prompt_len - 1 : prompt_len + response_len - 1, :
    ].squeeze(0).float()

    response_token_ids = response_ids.squeeze(0).to(model.device)

    log_probs_all = F.log_softmax(response_logits, dim=-1)

    # PPL (per-token)
    token_log_probs = log_probs_all.gather(
        1, response_token_ids.unsqueeze(1)
    ).squeeze(1)
    ppl_per_token_list = (-token_log_probs).tolist()
    avg_nll = -token_log_probs.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    # Self-Certainty (per-token)
    logprob_sum = log_probs_all.sum(dim=-1)
    V_float = float(V)
    sc_per_token = (-1.0 / V_float * logprob_sum - math.log(V_float))
    sc_per_token_list = sc_per_token.tolist()
    sc = sc_per_token.mean().item()

    return sc, ppl, response_len, sc_per_token_list, ppl_per_token_list


# ---------- Borda voting (Eq. 11 from the paper) ----------
def borda_select(candidates, score_key, higher_is_better=True, p=BORDA_P):
    """
    Borda Count: rank candidates by score_key, then vote.
    v(r) = (N - r + 1)^p where r = rank (1 = best).
    p=0 → pure majority vote; p=0.3 → paper's recommended.
    """
    N = len(candidates)
    sorted_idxs = sorted(range(N), key=lambda i: candidates[i][score_key],
                         reverse=higher_is_better)
    ranks = [0] * N
    for rank_pos, idx in enumerate(sorted_idxs):
        ranks[idx] = rank_pos + 1

    answer_votes = collections.defaultdict(float)
    answer_repr = {}
    for i, c in enumerate(candidates):
        r = ranks[i]
        vote = (N - r + 1) ** p
        norm = normalize_answer(c["final_answer"])
        answer_votes[norm] += vote
        if norm not in answer_repr:
            answer_repr[norm] = c["final_answer"]

    best_norm = max(answer_votes, key=answer_votes.__getitem__)
    return answer_repr[best_norm]


def majority_vote(candidates):
    """Pure majority vote (Borda with p=0, all votes equal)."""
    answer_counts = collections.Counter()
    answer_repr = {}
    for c in candidates:
        norm = normalize_answer(c["final_answer"])
        answer_counts[norm] += 1
        if norm not in answer_repr:
            answer_repr[norm] = c["final_answer"]
    best_norm = answer_counts.most_common(1)[0][0]
    return answer_repr[best_norm]


# ---------- Plotting ----------
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"
GRAY = "#999999"


def plot_distributions(results, dataset_name):
    """Plot SC and PPL distributions for correct vs incorrect CANDIDATES."""
    sc_correct, sc_incorrect = [], []
    ppl_correct, ppl_incorrect = [], []

    for r in results:
        for c in r["candidates"]:
            # Skip empty answer placeholders (sc=-1e9, ppl=inf)
            if c["sc"] <= -1e8 or c["ppl"] == float("inf"):
                continue
            if c["is_correct"]:
                sc_correct.append(c["sc"])
                ppl_correct.append(c["ppl"])
            else:
                sc_incorrect.append(c["sc"])
                ppl_incorrect.append(c["ppl"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_total = len(sc_correct) + len(sc_incorrect)
    fig.suptitle(
        f"SC vs PPL: Correct / Incorrect Candidate Distributions\n"
        f"({dataset_name}, {len(results)} questions x N={N_CANDIDATES}, total={n_total} candidates)",
        fontsize=12, fontweight="bold",
    )

    # SC distribution
    ax = axes[0]
    all_sc = sc_correct + sc_incorrect
    if all_sc:
        bins = np.linspace(np.percentile(all_sc, 1), np.percentile(all_sc, 99), 50)
        ax.hist(sc_correct, bins=bins, alpha=0.6, color=BLUE,
                label=f"Correct (n={len(sc_correct)})")
        ax.hist(sc_incorrect, bins=bins, alpha=0.6, color=ORANGE,
                label=f"Incorrect (n={len(sc_incorrect)})")
        mean_c = np.mean(sc_correct) if sc_correct else 0
        mean_i = np.mean(sc_incorrect) if sc_incorrect else 0
        ax.axvline(mean_c, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Mean correct: {mean_c:.2f}")
        ax.axvline(mean_i, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Mean incorrect: {mean_i:.2f}")
    ax.set_xlabel("Self-Certainty Score", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Self-Certainty (SC)\n↑ higher = more confident", fontsize=11)
    ax.legend(fontsize=9)

    # PPL distribution (negative PPL)
    ax = axes[1]
    neg_ppl_correct = [-p for p in ppl_correct]
    neg_ppl_incorrect = [-p for p in ppl_incorrect]
    all_neg_ppl = neg_ppl_correct + neg_ppl_incorrect
    if all_neg_ppl:
        bins = np.linspace(np.percentile(all_neg_ppl, 1), np.percentile(all_neg_ppl, 99), 50)
        ax.hist(neg_ppl_correct, bins=bins, alpha=0.6, color=BLUE,
                label=f"Correct (n={len(neg_ppl_correct)})")
        ax.hist(neg_ppl_incorrect, bins=bins, alpha=0.6, color=ORANGE,
                label=f"Incorrect (n={len(neg_ppl_incorrect)})")
        mean_c = np.mean(neg_ppl_correct) if neg_ppl_correct else 0
        mean_i = np.mean(neg_ppl_incorrect) if neg_ppl_incorrect else 0
        ax.axvline(mean_c, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Mean correct: {mean_c:.2f}")
        ax.axvline(mean_i, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8,
                   label=f"Mean incorrect: {mean_i:.2f}")
    ax.set_xlabel("Negative Perplexity (−PPL)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Negative Perplexity (−PPL)\n↑ higher = more confident", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp0_sc_vs_ppl_distribution_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


def plot_accuracy_comparison(stats, n_questions):
    """Bar chart comparing all selection methods."""
    methods = ["Random", "PPL\n(Best-of-N)", "SC\n(Best-of-N)",
               "Majority\nVote", "Borda-SC\n(p=0.3)", "Oracle"]
    values = [
        stats["random_correct"] / n_questions * 100,
        stats["ppl_correct"] / n_questions * 100,
        stats["sc_correct"] / n_questions * 100,
        stats["majority_correct"] / n_questions * 100,
        stats["borda_sc_correct"] / n_questions * 100,
        stats["oracle_correct"] / n_questions * 100,
    ]
    colors = [GRAY, ORANGE, BLUE, GREEN, PURPLE, RED]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Best-of-N Selection: {DATASET_NAME} (N={N_CANDIDATES}, CoT, Borda p={BORDA_P})",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp0_accuracy_comparison_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


def plot_token_trajectories(results, n_examples=5):
    """Plot per-token SC trajectories for a few correct vs incorrect candidates."""
    correct_candidates = []
    incorrect_candidates = []
    for r in results:
        for c in r["candidates"]:
            if len(c.get("sc_per_token", [])) >= 10:
                entry = {**c, "question": r["question"]}
                if c["is_correct"]:
                    correct_candidates.append(entry)
                else:
                    incorrect_candidates.append(entry)

    # Pick longest ones for clearer visualization
    correct_candidates.sort(key=lambda x: len(x["sc_per_token"]), reverse=True)
    incorrect_candidates.sort(key=lambda x: len(x["sc_per_token"]), reverse=True)

    correct_samples = correct_candidates[:n_examples]
    incorrect_samples = incorrect_candidates[:n_examples]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Per-Token SC & NLL Trajectories: Correct vs Incorrect ({DS_TAG} CoT, N={N_CANDIDATES})",
                 fontsize=14, fontweight="bold")

    ax = axes[0][0]
    for c in correct_samples:
        ax.plot(c["sc_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"SC — Correct Candidates (n={len(correct_samples)})", fontsize=11, color=BLUE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("SC per token")

    ax = axes[0][1]
    for c in incorrect_samples:
        ax.plot(c["sc_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"SC — Incorrect Candidates (n={len(incorrect_samples)})", fontsize=11, color=ORANGE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("SC per token")

    ax = axes[1][0]
    for c in correct_samples:
        ax.plot(c["ppl_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"NLL — Correct Candidates (n={len(correct_samples)})", fontsize=11, color=BLUE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neg log prob per token")

    ax = axes[1][1]
    for c in incorrect_samples:
        ax.plot(c["ppl_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"NLL — Incorrect Candidates (n={len(incorrect_samples)})", fontsize=11, color=ORANGE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neg log prob per token")

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, f"exp0_token_trajectories_{DS_TAG}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


# ---------- Main ----------
def main():
    logging.info(f"Loading model {MODEL_ID}...")
    logging.info(f"Dataset: {DATASET_NAME}")
    logging.info(f"N={N_CANDIDATES}, T={TEMPERATURE}, Borda p={BORDA_P}")
    logging.info(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    logging.info(f"MAX_QUESTIONS: {MAX_QUESTIONS}")

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
    logging.info("Model loaded.\n" + "=" * 60)

    data = load_data(DATA_FILE)
    if MAX_QUESTIONS is not None:
        data = data[:MAX_QUESTIONS]
    logging.info(f"Processing {len(data)} questions.\n")

    results = []
    stats = {
        "greedy_correct": 0,
        "random_correct": 0,
        "sc_correct": 0,
        "ppl_correct": 0,
        "majority_correct": 0,
        "borda_sc_correct": 0,
        "oracle_correct": 0,
    }

    for item in tqdm(data, desc="Processing"):
        question = item["question"]
        ground_truths = item["answers"]

        # --- Greedy baseline ---
        greedy_prompt = build_greedy_prompt(tokenizer, question)
        greedy_ids = tokenizer(
            greedy_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        with torch.no_grad():
            greedy_out = model.generate(
                greedy_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        greedy_new = greedy_out[0][greedy_ids.shape[1]:]
        greedy_answer = tokenizer.decode(greedy_new, skip_special_tokens=True).strip()
        greedy_correct = check_correct(greedy_answer, ground_truths)

        # --- Sample N candidates (CoT) ---
        cot_prompt = build_cot_prompt(tokenizer, question)
        cot_ids = tokenizer(
            cot_prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        candidates = []
        for n in range(N_CANDIDATES):
            with torch.no_grad():
                sample_out = model.generate(
                    cot_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                )
            sample_new = sample_out[0][cot_ids.shape[1]:]
            full_response = tokenizer.decode(sample_new, skip_special_tokens=True).strip()
            final_answer = extract_final_answer(full_response)

            # Handle empty answers
            if not final_answer.strip():
                sc, ppl, n_tokens = -1e9, float("inf"), 0
                sc_per_token, ppl_per_token = [], []
            else:
                sc, ppl, n_tokens, sc_per_token, ppl_per_token = compute_sc_and_ppl(
                    model, tokenizer, cot_prompt, full_response
                )

            is_corr = check_correct(final_answer, ground_truths)

            candidates.append({
                "candidate_idx": n,
                "full_response": full_response,
                "final_answer": final_answer,
                "is_correct": is_corr,
                "sc": round(sc, 6),
                "ppl": round(ppl, 4),
                "n_tokens": n_tokens,
                "sc_per_token": [round(v, 6) for v in sc_per_token],
                "ppl_per_token": [round(v, 6) for v in ppl_per_token],
            })

        # --- Selection methods ---
        # SC best-of-N (pick highest SC)
        sc_best = max(candidates, key=lambda c: c["sc"])
        sc_selected = sc_best["final_answer"]
        # PPL best-of-N (pick lowest PPL)
        ppl_best = min(candidates, key=lambda c: c["ppl"])
        ppl_selected = ppl_best["final_answer"]
        # Random
        random_selected = random.choice(candidates)["final_answer"]
        # Majority vote
        majority_selected = majority_vote(candidates)
        # Borda-SC (paper's method)
        borda_sc_selected = borda_select(candidates, "sc", higher_is_better=True, p=BORDA_P)
        # Oracle (any correct candidate exists?)
        oracle_correct = any(c["is_correct"] for c in candidates)

        # Update stats
        if greedy_correct:
            stats["greedy_correct"] += 1
        if check_correct(random_selected, ground_truths):
            stats["random_correct"] += 1
        if check_correct(sc_selected, ground_truths):
            stats["sc_correct"] += 1
        if check_correct(ppl_selected, ground_truths):
            stats["ppl_correct"] += 1
        if check_correct(majority_selected, ground_truths):
            stats["majority_correct"] += 1
        if check_correct(borda_sc_selected, ground_truths):
            stats["borda_sc_correct"] += 1
        if oracle_correct:
            stats["oracle_correct"] += 1

        entry = {
            "id": item["id"],
            "question": question,
            "ground_truths": ground_truths,
            "greedy_answer": greedy_answer,
            "greedy_correct": greedy_correct,
            "sc_selected": sc_selected,
            "ppl_selected": ppl_selected,
            "majority_selected": majority_selected,
            "borda_sc_selected": borda_sc_selected,
            "candidates": candidates,
        }
        results.append(entry)

        n_correct_candidates = sum(1 for c in candidates if c["is_correct"])
        logging.info(f"Q: {question}")
        logging.info(f"  GT: {ground_truths[0]}")
        logging.info(f"  Greedy: {greedy_answer} ({'O' if greedy_correct else 'X'})")
        logging.info(f"  Borda-SC: {borda_sc_selected} ({'O' if check_correct(borda_sc_selected, ground_truths) else 'X'})")
        logging.info(f"  Majority: {majority_selected} ({'O' if check_correct(majority_selected, ground_truths) else 'X'})")
        logging.info(f"  Candidates: {n_correct_candidates}/{N_CANDIDATES} correct")
        logging.info("-" * 60)

    # Save results
    n_q = len(results)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------- Summary ----------
    logging.info("\n" + "=" * 60)
    logging.info(f"Dataset: {DATASET_NAME} | N={N_CANDIDATES} | T={TEMPERATURE} | Borda p={BORDA_P}")
    logging.info(f"Total: {n_q} questions\n")

    for method, key in [
        ("Greedy", "greedy_correct"),
        ("Random", "random_correct"),
        ("PPL (Best-of-N)", "ppl_correct"),
        ("SC (Best-of-N)", "sc_correct"),
        ("Majority Vote", "majority_correct"),
        ("Borda-SC (p=0.3)", "borda_sc_correct"),
        ("Oracle", "oracle_correct"),
    ]:
        acc = stats[key] / n_q * 100
        logging.info(f"  {method:<22} {stats[key]:>4}/{n_q}  = {acc:.1f}%")

    # AUC on all candidates
    all_labels, all_sc, all_ppl = [], [], []
    for r in results:
        for c in r["candidates"]:
            all_labels.append(1 if c["is_correct"] else 0)
            all_sc.append(c["sc"])
            all_ppl.append(-c["ppl"] if c["ppl"] != float("inf") else -1e6)

    if len(set(all_labels)) == 2:
        auc_sc = roc_auc_score(all_labels, all_sc)
        auc_ppl = roc_auc_score(all_labels, all_ppl)
        logging.info(f"\n  AUC (SC):  {auc_sc:.4f}")
        logging.info(f"  AUC (PPL): {auc_ppl:.4f}")
        logging.info(f"  Winner: {'SC' if auc_sc > auc_ppl else 'PPL'}")

    # Save summary
    summary = {
        "dataset": DATASET_NAME,
        "n_questions": n_q,
        "n_candidates": N_CANDIDATES,
        "temperature": TEMPERATURE,
        "borda_p": BORDA_P,
        **{f"accuracy_{k.replace('_correct', '')}": round(v / n_q * 100, 2) for k, v in stats.items()},
    }
    if len(set(all_labels)) == 2:
        summary["auc_sc"] = round(auc_sc, 4)
        summary["auc_ppl"] = round(auc_ppl, 4)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\nSummary saved to {SUMMARY_FILE}")

    # ---------- Plot ----------
    plot_distributions(results, DATASET_NAME)
    plot_accuracy_comparison(stats, n_q)
    plot_token_trajectories(results)

    logging.info(f"All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
