"""
Exp3b: Sentence-Level SC Trajectory on CoT Outputs
====================================================
Exp3 was run on direct (short) answers and found weak signal.
Hypothesis: CoT outputs are longer → sentence-level SC signal should be stronger.

Method:
  1. Load exp7 CoT results (already have cot text + correctness labels)
  2. For each item, re-run a single FORWARD PASS (no generation) to get
     per-token logits → sc_per_token list
  3. Apply exp3-style sentence segmentation and trajectory analysis
  4. Compare correct vs incorrect CoT trajectories

Why forward pass instead of re-generating:
  - exp7 already saved the full cot text
  - One forward pass per item is ~10x faster than re-generating
  - Produces identical sc_per_token to what generation would have

Usage:
  conda activate adaptive_rag
  python exp3b_cot_trajectory.py --dataset nq
  python exp3b_cot_trajectory.py --dataset 2wiki
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import json
import math
import re

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ================= Args =================
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique"])
parser.add_argument("--n_examples", type=int, default=50,
                    help="Max examples to show in overlay plot (default: 50)")
args = parser.parse_args()

BASE_DIR  = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
EXP7_FILE = os.path.join(BASE_DIR, "results", f"exp7_cot_{args.dataset}_results.json")
OUT_JSON  = os.path.join(BASE_DIR, "results", f"exp3b_cot_trajectory_{args.dataset}.json")
FIG_DIR   = os.path.join(BASE_DIR, "figures")
MODEL_ID  = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE  = "/home/S113065528/Adaptive-RAG/hf_cache"
HF_TOKEN  = os.environ.get("HF_TOKEN", None)

os.makedirs(FIG_DIR, exist_ok=True)

BLUE   = "#4C72B0"
ORANGE = "#DD8452"
GREEN  = "#55A868"
RED    = "#C44E52"
GRAY   = "#CCCCCC"


# ================= Prompt (must match exp7) =================
def build_prompt(tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content":
            "Answer the given question. Always conduct and show your "
            "step-by-step reasoning first, then give the final answer. "
            "End with a line: 'Final Answer: <your answer>'."},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)


# ================= Forward pass → sc_per_token =================
@torch.no_grad()
def get_sc_per_token(model, tokenizer, question: str, cot_text: str):
    """
    Single forward pass through (prompt + cot_text).
    Returns list of per-token SC values for the cot portion only.
    SC_t = -1/V * sum(log P(v|x_t)) - log(V) = KL(P_t || Uniform)
    """
    V = model.config.vocab_size
    prompt_text = build_prompt(tokenizer, question)

    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    cot_ids = tokenizer(
        cot_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    cot_len = cot_ids.shape[1]

    if cot_len == 0:
        return []

    full_ids = torch.cat([prompt_ids, cot_ids], dim=1)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits  # (1, seq_len, V)

    # We want logits that predict each token in cot_ids
    # Position prompt_len-1 predicts cot token 0, etc.
    cot_logits = logits[0, prompt_len - 1: prompt_len + cot_len - 1, :].float()  # (cot_len, V)

    log_probs = F.log_softmax(cot_logits, dim=-1)  # (cot_len, V)
    lp_sum    = log_probs.sum(dim=-1)               # (cot_len,)
    sc_per    = (-1.0 / V * lp_sum - math.log(V)).tolist()

    return sc_per  # list of floats, length = cot_len


# ================= Sentence splitting (same as exp3) =================
def split_into_sentences(text: str):
    segments = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    return [s.strip() for s in segments if s.strip()]


# ================= Map sc_per_token → sentence-level trajectory =================
def get_sentence_trajectory(tokenizer, cot_text: str, sc_per_token: list):
    """
    Returns list of (sentence_text, mean_sc, n_tokens) or None on failure.
    """
    sentences = split_into_sentences(cot_text)
    if not sentences:
        return None

    sent_token_counts = [
        len(tokenizer(s, add_special_tokens=False).input_ids)
        for s in sentences
    ]
    total_sent = sum(sent_token_counts)
    total_sc   = len(sc_per_token)

    if total_sent == 0:
        return None
    ratio = total_sc / total_sent
    if ratio < 0.5 or ratio > 2.0:
        return None

    scaled = [max(1, round(c * ratio)) for c in sent_token_counts]

    # Trim or pad last segment to match total_sc exactly
    diff = total_sc - sum(scaled)
    scaled[-1] = max(1, scaled[-1] + diff)
    if sum(scaled) != total_sc:
        return None

    trajectory = []
    pos = 0
    for sent, n in zip(sentences, scaled):
        chunk = sc_per_token[pos: pos + n]
        trajectory.append((sent, float(np.mean(chunk)), n))
        pos += n

    return trajectory  # list of (text, mean_sc, n_tokens)


# ================= Trajectory feature extraction =================
def extract_features(trajectory):
    """
    From a list of (text, mean_sc, n_tokens), extract scalar features.
    """
    sc_vals = [t[1] for t in trajectory]
    n = len(sc_vals)
    if n < 2:
        return None

    # Decline ratio: fraction of consecutive pairs where SC drops in 2nd half
    half = n // 2
    second_half = sc_vals[half:]
    if len(second_half) > 1:
        drops = sum(1 for a, b in zip(second_half, second_half[1:]) if b < a)
        decline_ratio = drops / (len(second_half) - 1)
    else:
        decline_ratio = 0.0

    # Max single-sentence SC drop
    max_drop = max(sc_vals[i] - sc_vals[i + 1] for i in range(n - 1))

    # Linear slope of trajectory
    x = np.arange(n)
    slope = float(np.polyfit(x, sc_vals, 1)[0])

    # Normalized position of minimum SC
    min_pos = float(np.argmin(sc_vals)) / max(n - 1, 1)

    return {
        "decline_ratio": decline_ratio,
        "max_drop":      max_drop,
        "slope":         slope,
        "min_pos":       min_pos,
        "n_sentences":   n,
        "sc_vals":       sc_vals,
    }


# ================= Plotting =================
def plot_overlay(correct_trajs, incorrect_trajs, tag):
    """Mean trajectory ± std for correct vs incorrect."""
    def pad_and_stack(trajs, max_len=20):
        padded = []
        for sc_vals in trajs:
            v = sc_vals[:max_len]
            padded.append(v + [v[-1]] * (max_len - len(v)))  # pad with last value
        return np.array(padded)

    max_len = 20
    c_mat = pad_and_stack([f["sc_vals"] for f in correct_trajs],   max_len)
    i_mat = pad_and_stack([f["sc_vals"] for f in incorrect_trajs], max_len)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Exp3b: CoT Sentence-Level SC Trajectory — Correct vs Incorrect ({tag.upper()})\n"
        f"n_correct={len(correct_trajs)}, n_incorrect={len(incorrect_trajs)}",
        fontsize=13, fontweight="bold",
    )

    # Mean ± std
    ax = axes[0]
    x = np.arange(1, max_len + 1)
    for mat, color, label in [(c_mat, BLUE, "Correct"), (i_mat, ORANGE, "Incorrect")]:
        mean = mat.mean(axis=0)
        std  = mat.std(axis=0)
        ax.plot(x, mean, color=color, lw=2.5, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_xlabel("Sentence position")
    ax.set_ylabel("Mean SC")
    ax.set_title("Mean SC Trajectory (± 1 std)")
    ax.legend()

    # Individual overlays (sampled)
    ax = axes[1]
    rng = np.random.default_rng(42)
    n_show = min(args.n_examples, len(correct_trajs), len(incorrect_trajs))
    ci = rng.choice(len(correct_trajs),   n_show, replace=False)
    ii = rng.choice(len(incorrect_trajs), n_show, replace=False)
    for idx in ci:
        v = correct_trajs[idx]["sc_vals"][:max_len]
        ax.plot(range(1, len(v) + 1), v, color=BLUE,   alpha=0.15, lw=0.8)
    for idx in ii:
        v = incorrect_trajs[idx]["sc_vals"][:max_len]
        ax.plot(range(1, len(v) + 1), v, color=ORANGE, alpha=0.15, lw=0.8)
    # Bold means on top
    ax.plot(x, c_mat.mean(axis=0), color=BLUE,   lw=2.5, label="Correct mean")
    ax.plot(x, i_mat.mean(axis=0), color=ORANGE, lw=2.5, label="Incorrect mean")
    ax.set_xlabel("Sentence position")
    ax.set_ylabel("SC")
    ax.set_title(f"Individual Trajectories (n={n_show} each)")
    ax.legend()

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"exp3b_trajectories_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_feature_stats(correct_feats, incorrect_feats, tag):
    """Distribution of trajectory features: decline ratio, max drop, slope."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Exp3b: CoT Trajectory Feature Statistics — Correct vs Incorrect ({tag.upper()})",
        fontsize=13, fontweight="bold",
    )

    features = [
        ("decline_ratio", "SC Decline Ratio\n(fraction of 2nd-half sentences with SC drop)",  (-0.5, 1.0)),
        ("max_drop",      "Max Single-Sentence SC Drop",                                       (-2, 15)),
        ("slope",         "SC Trajectory Slope\n(negative = overall downward trend)",          (-8, 4)),
    ]

    for ax, (key, xlabel, xlim) in zip(axes, features):
        c_vals = [f[key] for f in correct_feats]
        i_vals = [f[key] for f in incorrect_feats]

        ax.hist(c_vals, bins=30, alpha=0.55, color=BLUE,   label=f"Correct (n={len(c_vals)})",
                density=True)
        ax.hist(i_vals, bins=30, alpha=0.55, color=ORANGE, label=f"Incorrect (n={len(i_vals)})",
                density=True)

        c_med = float(np.median(c_vals))
        i_med = float(np.median(i_vals))
        ax.axvline(c_med, color=BLUE,   ls="--", lw=1.8, label=f"Correct median={c_med:.3f}")
        ax.axvline(i_med, color=ORANGE, ls="--", lw=1.8, label=f"Incorrect median={i_med:.3f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_xlim(xlim)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"exp3b_feature_stats_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


def plot_sc_heatmap(correct_feats, incorrect_feats, tag):
    """
    Heatmap: rows = examples (sorted by n_sentences), cols = sentence position.
    Helps visually see if incorrect answers have a characteristic SC profile.
    """
    max_len = 20

    def make_matrix(feats):
        rows = []
        for f in feats:
            v = f["sc_vals"][:max_len]
            v = v + [np.nan] * (max_len - len(v))
            rows.append(v)
        return np.array(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Exp3b: SC Heatmap — CoT Trajectories ({tag.upper()})\n"
        "Each row = one response, each column = one sentence",
        fontsize=12, fontweight="bold",
    )

    rng = np.random.default_rng(0)
    n_show = min(100, len(correct_feats), len(incorrect_feats))

    for ax, feats, title in [
        (axes[0], rng.choice(correct_feats,   n_show, replace=False).tolist(), f"Correct (n={n_show})"),
        (axes[1], rng.choice(incorrect_feats, n_show, replace=False).tolist(), f"Incorrect (n={n_show})"),
    ]:
        mat = make_matrix(feats)
        # Sort rows by first-sentence SC (high → low) for visual clarity
        order = np.argsort(mat[:, 0])[::-1]
        mat = mat[order]

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn",
                       vmin=np.nanpercentile(mat, 5),
                       vmax=np.nanpercentile(mat, 95))
        ax.set_xlabel("Sentence position")
        ax.set_ylabel("Response (sorted by SC₁)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="SC")

    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"exp3b_heatmap_{tag}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)


# ================= Main =================
def main():
    # Load exp7 results
    if not os.path.exists(EXP7_FILE):
        print(f"Exp7 results not found: {EXP7_FILE}")
        print("Run exp7_cot_sc_vs_ppl.py first.")
        return

    with open(EXP7_FILE, encoding="utf-8") as f:
        exp7_data = json.load(f)
    print(f"Loaded {len(exp7_data)} CoT results from {EXP7_FILE}")

    # Filter: only items with non-empty cot
    exp7_data = [item for item in exp7_data if item.get("cot", "").strip()]
    print(f"Items with non-empty CoT: {len(exp7_data)}")

    # Load model
    print(f"Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, token=HF_TOKEN, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, device_map="cuda",
        torch_dtype=torch.bfloat16, token=HF_TOKEN, local_files_only=True)
    model.eval()
    print("Model loaded.\n")

    # Forward pass for each item → sc_per_token → sentence trajectory
    processed = []
    skipped = 0

    skip_reasons = {"empty_sc": 0, "traj_none": 0, "feats_none": 0}

    for item in tqdm(exp7_data, desc="Forward pass + trajectory"):
        sc_per_token = get_sc_per_token(model, tokenizer, item["question"], item["cot"])
        if not sc_per_token:
            skip_reasons["empty_sc"] += 1
            skipped += 1
            continue

        traj = get_sentence_trajectory(tokenizer, item["cot"], sc_per_token)
        if traj is None:
            skip_reasons["traj_none"] += 1
            skipped += 1
            continue

        feats = extract_features(traj)
        if feats is None:
            skip_reasons["feats_none"] += 1
            skipped += 1
            continue

        processed.append({
            "id":         item["id"],
            "is_correct": item["is_correct"],
            "n_tokens":   item["n_tokens"],
            "sc_whole":   item["sc"],
            "features":   feats,
            # Store trajectory without the text (saves space)
            "trajectory": [(sc, n) for _, sc, n in traj],
        })

    print(f"\nProcessed: {len(processed)}, Skipped: {skipped}")
    for reason, count in skip_reasons.items():
        print(f"  Skip reason [{reason}]: {count}")

    correct_feats   = [p["features"] for p in processed if     p["is_correct"]]
    incorrect_feats = [p["features"] for p in processed if not p["is_correct"]]
    print(f"Correct: {len(correct_feats)}, Incorrect: {len(incorrect_feats)}")

    # Save
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"Saved trajectory data: {OUT_JSON}")

    # ========== Statistics ==========
    print("\n" + "=" * 60)
    print(f"CoT Sentence-Level SC Trajectory Statistics ({args.dataset.upper()})")
    print("=" * 60)

    for name, feats in [("Correct", correct_feats), ("Incorrect", incorrect_feats)]:
        print(f"\n{name} (n={len(feats)}):")
        for key in ("decline_ratio", "max_drop", "slope", "n_sentences"):
            vals = [f[key] for f in feats]
            print(f"  {key:<18} — mean={np.mean(vals):.4f}, median={np.median(vals):.4f}")

    # Key comparison
    print("\n" + "=" * 60)
    print("VERDICT: Does CoT sentence-level SC show stronger signal than direct answers?")
    print("=" * 60)

    for key, direction in [
        ("decline_ratio", "Incorrect should be HIGHER (more drops in 2nd half)"),
        ("slope",         "Incorrect should be MORE NEGATIVE (steeper decline)"),
        ("max_drop",      "Incorrect should be HIGHER (bigger crash)"),
    ]:
        c_med = float(np.median([f[key] for f in correct_feats]))
        i_med = float(np.median([f[key] for f in incorrect_feats]))
        gap   = i_med - c_med
        signal = "STRONG" if abs(gap) > 0.05 * max(abs(c_med), abs(i_med), 1) else "WEAK"
        print(f"\n  {key}")
        print(f"    Correct median:   {c_med:.4f}")
        print(f"    Incorrect median: {i_med:.4f}")
        print(f"    Gap:              {gap:+.4f}  [{direction}]")
        print(f"    Signal:           {signal}")

    # ========== Plots ==========
    plot_overlay(correct_feats, incorrect_feats, args.dataset)
    plot_feature_stats(correct_feats, incorrect_feats, args.dataset)
    plot_sc_heatmap(np.array(correct_feats, dtype=object),
                    np.array(incorrect_feats, dtype=object), args.dataset)

    print("\nDone.")


if __name__ == "__main__":
    main()
