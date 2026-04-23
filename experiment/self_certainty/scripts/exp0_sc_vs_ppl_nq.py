"""
Experiment 0: Self-Certainty vs PPL — Distribution & Per-Token Trajectory
=========================================================================
Goal:
  1. Show SC separates correct/incorrect answers better than PPL (distribution plot)
  2. Save per-token SC trajectories for trajectory visualization
  3. Compute AUC to quantify separation quality

Dataset: 2WikiMultiHopQA (multi-hop questions → CoT generates longer reasoning)
Model: meta-llama/Llama-3.1-8B-Instruct (decoder-only)
Prompt: Chain-of-Thought — model reasons step by step, then gives final answer

SC/PPL are computed on the FULL generated response (reasoning + answer),
so SC has many tokens to accumulate signal.

Self-Certainty formula (from arxiv 2502.18581):
  SC = mean_over_tokens( -1/V * sum_j log P(j | x, y<i) - log(V) )

PPL formula:
  PPL = exp( -1/n * sum_i log P(y_i | x, y<i) )
"""

import json
import logging
import math
import os
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

# ================= Settings =================
DATA_FILE = "/home/S113065528/Adaptive-RAG/processed_data/2wikimultihopqa/test_subsampled.jsonl"
DATASET_NAME = "2wikimultihopqa"
OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "exp0_sc_vs_ppl_results.json")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
LOG_FILE = os.path.join(OUTPUT_DIR, "exp0_sc_vs_ppl.log")

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MAX_NEW_TOKENS = 256  # longer for CoT reasoning
MAX_QUESTIONS = 3     # set to None to run all questions
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


# ---------- Extract final answer from CoT response ----------
def extract_final_answer(response):
    """
    Extract the final answer from a CoT response.
    Looks for patterns like "Answer: ...", "Final answer: ...", "the answer is ..."
    Falls back to the last line if no pattern found.
    """
    # Try common patterns
    patterns = [
        r"(?:Final\s+)?[Aa]nswer\s*:\s*(.+?)(?:\n|$)",
        r"[Tt]he\s+(?:final\s+)?answer\s+is\s+(.+?)(?:\.|,|\n|$)",
        r"[Ss]o\s+the\s+answer\s+is\s+(.+?)(?:\.|,|\n|$)",
        r"[Tt]herefore,?\s+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|,|\n|$)",
    ]
    for pat in patterns:
        m = re.search(pat, response)
        if m:
            return m.group(1).strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
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


# ---------- Build CoT chat prompt ----------
def build_cot_prompt(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Think through the question step by step, "
                "then provide the final answer.\n"
                "Format:\n"
                "Step 1: ...\n"
                "Step 2: ...\n"
                "Answer: <your short answer>"
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
    Returns: (sc_score, ppl_score, num_tokens, sc_per_token_list, ppl_per_token_list)
    """
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

    # Logits at position [prompt_len-1 .. prompt_len+response_len-2] predict response tokens
    response_logits = logits[
        :, prompt_len - 1 : prompt_len + response_len - 1, :
    ].squeeze(0).float()  # (response_len, V)

    response_token_ids = response_ids.squeeze(0).to(model.device)

    # --- Log probabilities ---
    log_probs_all = F.log_softmax(response_logits, dim=-1)  # (response_len, V)

    # --- PPL (per-token) ---
    token_log_probs = log_probs_all.gather(
        1, response_token_ids.unsqueeze(1)
    ).squeeze(1)  # (response_len,)
    ppl_per_token_list = (-token_log_probs).tolist()
    avg_nll = -token_log_probs.mean().item()
    ppl = math.exp(min(avg_nll, 20))

    # --- Self-Certainty (per-token) ---
    logprob_sum = log_probs_all.sum(dim=-1)  # (response_len,)
    V_float = float(V)
    sc_per_token = (-1.0 / V_float * logprob_sum - math.log(V_float))
    sc_per_token_list = sc_per_token.tolist()
    sc = sc_per_token.mean().item()

    return sc, ppl, response_len, sc_per_token_list, ppl_per_token_list


# ---------- Plotting ----------
BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"


def plot_distributions(results, dataset_name):
    """Plot SC and PPL distributions for correct vs incorrect answers."""
    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    sc_correct = [r["self_certainty"] for r in correct]
    sc_incorrect = [r["self_certainty"] for r in incorrect]
    ppl_correct = [r["ppl"] for r in correct if r["ppl"] != float("inf")]
    ppl_incorrect = [r["ppl"] for r in incorrect if r["ppl"] != float("inf")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"SC vs PPL: Correct / Incorrect Answer Distributions ({dataset_name}, n={len(results)}, CoT)",
        fontsize=13, fontweight="bold",
    )

    # --- SC distribution ---
    ax = axes[0]
    all_sc = sc_correct + sc_incorrect
    bins = np.linspace(np.percentile(all_sc, 1), np.percentile(all_sc, 99), 50)
    ax.hist(sc_correct, bins=bins, alpha=0.6, color=BLUE,
            label=f"Correct (n={len(sc_correct)})", density=True)
    ax.hist(sc_incorrect, bins=bins, alpha=0.6, color=ORANGE,
            label=f"Incorrect (n={len(sc_incorrect)})", density=True)
    mean_c = np.mean(sc_correct) if sc_correct else 0
    mean_i = np.mean(sc_incorrect) if sc_incorrect else 0
    ax.axvline(mean_c, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Mean correct: {mean_c:.2f}")
    ax.axvline(mean_i, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Mean incorrect: {mean_i:.2f}")
    ax.set_xlabel("Self-Certainty Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Self-Certainty (SC)\n↑ higher = more confident", fontsize=11)
    ax.legend(fontsize=9)

    # --- PPL distribution (use negative PPL so higher = more confident) ---
    ax = axes[1]
    neg_ppl_correct = [-p for p in ppl_correct]
    neg_ppl_incorrect = [-p for p in ppl_incorrect]
    all_neg_ppl = neg_ppl_correct + neg_ppl_incorrect
    bins = np.linspace(np.percentile(all_neg_ppl, 1), np.percentile(all_neg_ppl, 99), 50)
    ax.hist(neg_ppl_correct, bins=bins, alpha=0.6, color=BLUE,
            label=f"Correct (n={len(neg_ppl_correct)})", density=True)
    ax.hist(neg_ppl_incorrect, bins=bins, alpha=0.6, color=ORANGE,
            label=f"Incorrect (n={len(neg_ppl_incorrect)})", density=True)
    mean_c = np.mean(neg_ppl_correct) if neg_ppl_correct else 0
    mean_i = np.mean(neg_ppl_incorrect) if neg_ppl_incorrect else 0
    ax.axvline(mean_c, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Mean correct: {mean_c:.2f}")
    ax.axvline(mean_i, color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Mean incorrect: {mean_i:.2f}")
    ax.set_xlabel("Negative Perplexity (−PPL)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Negative Perplexity (−PPL)\n↑ higher = more confident", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "exp0_sc_vs_ppl_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


def plot_token_trajectories(results, n_examples=10):
    """
    Plot per-token SC trajectories: correct vs incorrect side by side.
    Pick examples with longer responses for clearer visualization.
    """
    correct = sorted(
        [r for r in results if r["is_correct"] and len(r["sc_per_token"]) >= 5],
        key=lambda r: len(r["sc_per_token"]), reverse=True
    )
    incorrect = sorted(
        [r for r in results if not r["is_correct"] and len(r["sc_per_token"]) >= 5],
        key=lambda r: len(r["sc_per_token"]), reverse=True
    )

    correct_samples = correct[:n_examples]
    incorrect_samples = incorrect[:n_examples]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Per-Token SC & NLL Trajectories: Correct vs Incorrect (CoT responses)",
                 fontsize=14, fontweight="bold")

    # Top-left: SC trajectories for correct answers
    ax = axes[0][0]
    for r in correct_samples:
        label = r["predicted_answer"][:30] + "..." if len(r["predicted_answer"]) > 30 else r["predicted_answer"]
        ax.plot(r["sc_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"SC — Correct Answers (n={len(correct_samples)})", fontsize=11, color=BLUE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("SC per token")

    # Top-right: SC trajectories for incorrect answers
    ax = axes[0][1]
    for r in incorrect_samples:
        ax.plot(r["sc_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"SC — Incorrect Answers (n={len(incorrect_samples)})", fontsize=11, color=ORANGE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("SC per token")

    # Bottom-left: NLL trajectories for correct answers
    ax = axes[1][0]
    for r in correct_samples:
        ax.plot(r["ppl_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"NLL — Correct Answers (n={len(correct_samples)})", fontsize=11, color=BLUE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neg log prob per token")

    # Bottom-right: NLL trajectories for incorrect answers
    ax = axes[1][1]
    for r in incorrect_samples:
        ax.plot(r["ppl_per_token"], alpha=0.6, linewidth=1.0)
    ax.set_title(f"NLL — Incorrect Answers (n={len(incorrect_samples)})", fontsize=11, color=ORANGE)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Neg log prob per token")

    fig.tight_layout()
    out = os.path.join(FIGURE_DIR, "exp0_token_trajectories.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {out}")
    plt.close(fig)


def main():
    logging.info(f"Loading model {MODEL_ID}...")
    logging.info(f"Dataset: {DATASET_NAME}")
    logging.info(f"Prompt: Chain-of-Thought (step by step)")
    logging.info(f"MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")

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

    # Load data
    data = load_data(DATA_FILE)
    if MAX_QUESTIONS is not None:
        data = data[:MAX_QUESTIONS]
        logging.info(f"Loaded {len(data)} questions (limited to {MAX_QUESTIONS}).\n")
    else:
        logging.info(f"Loaded {len(data)} questions with answers.\n")

    results = []

    for item in tqdm(data, desc="Processing"):
        question = item["question"]
        ground_truths = item["answers"]

        # Step 1: Generate CoT response (greedy, no retrieval)
        prompt_text = build_cot_prompt(tokenizer, question)
        input_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_ids = output_ids[0][input_ids.shape[1]:]
        full_response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Step 2: Extract final answer from CoT response
        final_answer = extract_final_answer(full_response)

        # Step 3: Evaluate correctness
        em = max_over_ground_truths(compute_exact, final_answer, ground_truths)
        f1 = max_over_ground_truths(compute_f1, final_answer, ground_truths)
        is_correct = em == 1 or f1 >= 0.5

        # Step 4: Compute SC and PPL on the FULL response (reasoning + answer)
        sc, ppl, n_tokens, sc_per_token, ppl_per_token = compute_sc_and_ppl(
            model, tokenizer, prompt_text, full_response
        )

        # Decode response tokens for trajectory labels
        response_ids = tokenizer(
            full_response, return_tensors="pt", add_special_tokens=False
        ).input_ids.squeeze(0)
        response_tokens = [tokenizer.decode([tid]) for tid in response_ids]

        entry = {
            "id": item["id"],
            "question": question,
            "full_response": full_response,
            "predicted_answer": final_answer,
            "ground_truths": ground_truths,
            "em": em,
            "f1": round(f1, 4),
            "is_correct": is_correct,
            "self_certainty": round(sc, 6),
            "ppl": round(ppl, 4),
            "n_response_tokens": n_tokens,
            "sc_per_token": [round(v, 6) for v in sc_per_token],
            "ppl_per_token": [round(v, 6) for v in ppl_per_token],
            "response_tokens": response_tokens,
        }
        results.append(entry)

        logging.info(f"Q: {question}")
        logging.info(f"  Full response ({n_tokens} tokens):")
        for line in full_response.split("\n"):
            logging.info(f"    {line}")
        logging.info(f"  Extracted answer: {final_answer}")
        logging.info(f"  GT: {ground_truths[0]}")
        logging.info(f"  EM={em}  F1={f1:.2f}  correct={is_correct}")
        logging.info(f"  SC={sc:.4f}  PPL={ppl:.2f}  tokens={n_tokens}")
        logging.info("-" * 60)

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------- Summary statistics ----------
    correct = [r for r in results if r["is_correct"]]
    incorrect = [r for r in results if not r["is_correct"]]

    def avg(lst, key):
        vals = [x[key] for x in lst if x[key] != float("inf")]
        return sum(vals) / len(vals) if vals else 0

    avg_tokens_correct = np.mean([r["n_response_tokens"] for r in correct]) if correct else 0
    avg_tokens_incorrect = np.mean([r["n_response_tokens"] for r in incorrect]) if incorrect else 0

    logging.info("\n" + "=" * 60)
    logging.info(f"Dataset: {DATASET_NAME} (CoT prompt)")
    logging.info(f"Total: {len(results)} questions")
    logging.info(f"Correct: {len(correct)} ({100*len(correct)/len(results):.1f}%)")
    logging.info(f"Incorrect: {len(incorrect)} ({100*len(incorrect)/len(results):.1f}%)")
    logging.info("")
    logging.info("                  Correct       Incorrect")
    logging.info(f"  Avg SC:         {avg(correct, 'self_certainty'):.4f}        {avg(incorrect, 'self_certainty'):.4f}")
    logging.info(f"  Avg PPL:        {avg(correct, 'ppl'):.2f}         {avg(incorrect, 'ppl'):.2f}")
    logging.info(f"  Avg tokens:     {avg_tokens_correct:.1f}          {avg_tokens_incorrect:.1f}")

    # ---------- AUC ----------
    labels = [1 if r["is_correct"] else 0 for r in results]
    sc_scores = [r["self_certainty"] for r in results]
    ppl_scores = [-r["ppl"] if r["ppl"] != float("inf") else -1e6 for r in results]

    if len(set(labels)) == 2:
        auc_sc = roc_auc_score(labels, sc_scores)
        auc_ppl = roc_auc_score(labels, ppl_scores)
        logging.info(f"\n  AUC (SC):  {auc_sc:.4f}")
        logging.info(f"  AUC (PPL): {auc_ppl:.4f}")
        winner = "SC" if auc_sc > auc_ppl else "PPL"
        logging.info(f"  Winner: {winner}")
        delta = abs(auc_sc - auc_ppl)
        logging.info(f"  Delta: {delta:.4f}")

    logging.info(f"\nResults saved to {OUTPUT_FILE}")

    # ---------- Plot ----------
    plot_distributions(results, DATASET_NAME)
    plot_token_trajectories(results)

    logging.info(f"All figures saved to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
