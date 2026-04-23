"""
Experiment 3 (Direction A): Sentence-Level SC Trajectory Analysis
=================================================================
Goal: Instead of per-token SC (too noisy) or arbitrary sliding windows,
      segment the generated response by SENTENCES (semantic units),
      compute per-sentence average SC, and check if incorrect answers
      show detectable patterns:
        Pattern B: gradual decline (SC slowly drops sentence by sentence)
        Pattern C: sudden crash (SC drops sharply at one sentence)

Method:
  1. Load existing exp0 results (already have sc_per_token + full_response)
  2. Load tokenizer ONLY (no GPU needed, no model)
  3. Split full_response into sentences by period/newline/etc.
  4. Re-tokenize each sentence → count tokens → map to sc_per_token boundaries
  5. Average SC within each sentence's token range → sentence-level trajectory
  6. Plot and compute statistics to see if correct/incorrect differ

Why sentence-level > sliding window:
  - Sentences are semantic units — SC drop at sentence boundary is meaningful
  - No arbitrary hyperparameter (window size)
  - Matches the intuition: model "knows" something → SC stable; starts
    hallucinating → SC drops, and it drops for a whole sentence

Usage:
  python exp3_smoothed_trajectory.py --dataset nq
  python exp3_smoothed_trajectory.py --dataset 2wiki
  python exp3_smoothed_trajectory.py --dataset nq --n_examples 30
"""

import argparse
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ================= Parse Arguments =================
parser = argparse.ArgumentParser(description="Sentence-level SC trajectory analysis")
parser.add_argument("--dataset", type=str, default="nq",
                    choices=["nq", "2wiki", "hotpotqa", "musique", "squad", "trivia"])
parser.add_argument("--n_examples", type=int, default=50,
                    help="Max incorrect examples to include in overlay plot (default: 50)")
args = parser.parse_args()

# ================= Paths =================
BASE_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
# Some datasets use legacy filenames
RESULTS_FILENAME_MAP = {
    "2wiki": "exp0_sc_vs_ppl_2wiki_results.json",
}
_results_filename = RESULTS_FILENAME_MAP.get(args.dataset, f"exp0_{args.dataset}_results.json")
RESULTS_FILE = os.path.join(BASE_DIR, "results", _results_filename)
FIGURE_DIR = os.path.join(BASE_DIR, "figures")
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_CACHE = "/home/S113065528/Adaptive-RAG/hf_cache"
os.makedirs(FIGURE_DIR, exist_ok=True)

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
GRAY = "#CCCCCC"


# ================= Sentence Splitting =================
def split_into_sentences(text):
    """
    Split text into sentences. Uses period/newline/exclamation/question.
    Keeps CoT structure: splits on "." + space or newline.
    Filters out empty segments.
    """
    # Split on sentence-ending punctuation followed by space/newline,
    # or just newlines (for CoT "Reasoning: ...\nAnswer: ...")
    segments = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    segments = [s.strip() for s in segments if s.strip()]
    return segments


# ================= Token Boundary Mapping =================
def get_sentence_sc_trajectory(tokenizer, full_response, sc_per_token):
    """
    Map sc_per_token values to sentence-level averages.

    Returns list of (sentence_text, mean_sc, n_tokens) tuples.
    Returns None if sentences can't be mapped (too short, mismatch, etc.).
    """
    sentences = split_into_sentences(full_response)
    if not sentences:
        return None

    # Tokenize each sentence to get token counts
    sentence_token_counts = []
    for sent in sentences:
        ids = tokenizer(sent, add_special_tokens=False).input_ids
        sentence_token_counts.append(len(ids))

    total_from_sentences = sum(sentence_token_counts)
    total_sc_tokens = len(sc_per_token)

    # Sentence tokenization may not perfectly match the original tokenization
    # (context window difference). Only proceed if reasonably close.
    if total_from_sentences == 0:
        return None
    ratio = total_sc_tokens / total_from_sentences
    if ratio < 0.5 or ratio > 2.0:
        # Too much mismatch — skip
        return None

    # Scale token counts to match actual sc_per_token length
    scaled_counts = []
    for cnt in sentence_token_counts:
        scaled = max(1, round(cnt * ratio))
        scaled_counts.append(scaled)

    # Trim or pad to match total_sc_tokens
    diff = sum(scaled_counts) - total_sc_tokens
    if diff > 0:
        # Trim from last segment
        scaled_counts[-1] = max(1, scaled_counts[-1] - diff)
    elif diff < 0:
        # Add to last segment
        scaled_counts[-1] += (-diff)

    # Build per-sentence SC averages
    trajectory = []
    pos = 0
    for i, (sent, cnt) in enumerate(zip(sentences, scaled_counts)):
        end = min(pos + cnt, total_sc_tokens)
        segment_sc = sc_per_token[pos:end]
        if segment_sc:
            trajectory.append({
                "sentence": sent,
                "mean_sc": float(np.mean(segment_sc)),
                "n_tokens": end - pos,
            })
        pos = end
        if pos >= total_sc_tokens:
            break

    return trajectory if len(trajectory) >= 2 else None


# ================= Trajectory Stats =================
def compute_traj_stats(traj):
    """
    Given a sentence-level trajectory (list of mean_sc values), compute:
    - decline_ratio: (first_half_mean - second_half_mean) / |first_half_mean|
    - max_drop: biggest single-step drop between consecutive sentences
    - slope: linear regression slope
    """
    sc_vals = [t["mean_sc"] for t in traj]
    n = len(sc_vals)
    if n < 2:
        return None

    half = max(1, n // 2)
    first_mean = np.mean(sc_vals[:half])
    second_mean = np.mean(sc_vals[half:])
    decline_ratio = (first_mean - second_mean) / abs(first_mean) if first_mean != 0 else 0.0

    diffs = [sc_vals[i+1] - sc_vals[i] for i in range(n - 1)]
    max_drop = -min(diffs)  # positive = big drop

    x = np.arange(n)
    slope = float(np.polyfit(x, sc_vals, 1)[0])

    return {
        "decline_ratio": decline_ratio,
        "max_drop": max_drop,
        "slope": slope,
        "n_sentences": n,
        "sc_vals": sc_vals,
    }


# ================= Load Data =================
print(f"Loading {RESULTS_FILE}...")
with open(RESULTS_FILE, encoding="utf-8") as f:
    all_results = json.load(f)
print(f"Total questions: {len(all_results)}")

print(f"Loading tokenizer from {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, cache_dir=HF_CACHE, local_files_only=True
)
print("Tokenizer loaded.\n")

# ================= Process All Candidates =================
correct_entries = []   # {traj, stats, question, answer}
incorrect_entries = []

skipped = 0
for r in all_results:
    for c in r["candidates"]:
        sc_traj = c.get("sc_per_token", [])
        full_resp = c.get("full_response", "")
        if not sc_traj or not full_resp or len(sc_traj) < 5:
            skipped += 1
            continue

        traj = get_sentence_sc_trajectory(tokenizer, full_resp, sc_traj)
        if traj is None:
            skipped += 1
            continue

        stats = compute_traj_stats(traj)
        if stats is None:
            skipped += 1
            continue

        entry = {
            "question": r["question"],
            "answer": c["final_answer"],
            "full_response": full_resp,
            "traj": traj,
            "stats": stats,
            "sc": c["sc"],
            "is_correct": c["is_correct"],
        }
        if c["is_correct"]:
            correct_entries.append(entry)
        else:
            incorrect_entries.append(entry)

print(f"Processed — Correct: {len(correct_entries)}, Incorrect: {len(incorrect_entries)}, Skipped: {skipped}")

# ================= Print Statistics =================
print(f"\n{'='*60}")
print("Sentence-Level SC Trajectory Statistics")
print(f"{'='*60}")

for label, entries in [("Correct", correct_entries), ("Incorrect", incorrect_entries)]:
    if not entries:
        continue
    dec = [e["stats"]["decline_ratio"] for e in entries]
    drop = [e["stats"]["max_drop"] for e in entries]
    slope = [e["stats"]["slope"] for e in entries]
    n_sent = [e["stats"]["n_sentences"] for e in entries]
    print(f"\n{label} (n={len(entries)}):")
    print(f"  Decline ratio  — mean={np.mean(dec):.4f}, median={np.median(dec):.4f}")
    print(f"  Max drop       — mean={np.mean(drop):.4f}, median={np.median(drop):.4f}")
    print(f"  Slope          — mean={np.mean(slope):.6f}, median={np.median(slope):.6f}")
    print(f"  Avg sentences  — {np.mean(n_sent):.1f}")

# ================= Figure 1: Aligned Trajectories =================
# Normalize trajectory position to [0, 1] so we can overlay
# trajectories of different lengths

N_SHOW = min(args.n_examples, len(incorrect_entries))
N_CORRECT_SHOW = min(args.n_examples, len(correct_entries))
INTERP_POINTS = 20  # number of points to interpolate to

def interpolate_traj(sc_vals, n_points=INTERP_POINTS):
    x_orig = np.linspace(0, 1, len(sc_vals))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_orig, sc_vals)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    f"Exp 3: Sentence-Level SC Trajectories ({args.dataset.upper()})\n"
    f"Each line = one generated answer's SC across its sentences (normalized to same length)\n"
    f"Left: Incorrect | Right: Correct",
    fontsize=12, fontweight="bold",
)

x_axis = np.linspace(0, 1, INTERP_POINTS)

# Incorrect panel
ax = axes[0]
interp_incorrect = []
for entry in incorrect_entries[:N_SHOW]:
    vals = entry["stats"]["sc_vals"]
    interp = interpolate_traj(vals)
    ax.plot(x_axis, interp, alpha=0.2, color=RED, linewidth=0.8)
    interp_incorrect.append(interp)

if interp_incorrect:
    mean_incorrect = np.mean(interp_incorrect, axis=0)
    ax.plot(x_axis, mean_incorrect, color="black", linewidth=2.5, label="Mean")

ax.set_xlabel("Sentence position (normalized)", fontsize=10)
ax.set_ylabel("Mean SC of sentence", fontsize=10)
ax.set_title(f"Incorrect Answers (n={N_SHOW})", fontsize=11)
ax.legend(fontsize=9)

# Correct panel
ax = axes[1]
interp_correct = []
for entry in correct_entries[:N_CORRECT_SHOW]:
    vals = entry["stats"]["sc_vals"]
    interp = interpolate_traj(vals)
    ax.plot(x_axis, interp, alpha=0.2, color=BLUE, linewidth=0.8)
    interp_correct.append(interp)

if interp_correct:
    mean_correct = np.mean(interp_correct, axis=0)
    ax.plot(x_axis, mean_correct, color="black", linewidth=2.5, label="Mean")

ax.set_xlabel("Sentence position (normalized)", fontsize=10)
ax.set_ylabel("Mean SC of sentence", fontsize=10)
ax.set_title(f"Correct Answers (n={N_CORRECT_SHOW})", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
out1 = os.path.join(FIGURE_DIR, f"exp3_sentence_trajectories_{args.dataset}.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out1}")
plt.close(fig)

# ================= Figure 2: Mean Trajectories Overlaid =================
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle(
    f"Exp 3: Mean Sentence-Level SC — Correct vs Incorrect ({args.dataset.upper()})\n"
    f"If incorrect shows downward trend → trajectory-based RAG trigger is viable",
    fontsize=12, fontweight="bold",
)

if interp_correct:
    ax.plot(x_axis, mean_correct, color=BLUE, linewidth=2.5, marker="o",
            markersize=4, label=f"Correct (n={len(correct_entries)})")
    ax.fill_between(x_axis,
                    mean_correct - np.std(interp_correct, axis=0),
                    mean_correct + np.std(interp_correct, axis=0),
                    alpha=0.15, color=BLUE)

if interp_incorrect:
    ax.plot(x_axis, mean_incorrect, color=RED, linewidth=2.5, marker="o",
            markersize=4, label=f"Incorrect (n={len(incorrect_entries)})")
    ax.fill_between(x_axis,
                    mean_incorrect - np.std(interp_incorrect, axis=0),
                    mean_incorrect + np.std(interp_incorrect, axis=0),
                    alpha=0.15, color=RED)

ax.set_xlabel("Sentence position (normalized: 0=first sentence, 1=last)", fontsize=11)
ax.set_ylabel("Mean SC", fontsize=11)
ax.legend(fontsize=11)
fig.tight_layout()
out2 = os.path.join(FIGURE_DIR, f"exp3_mean_trajectories_{args.dataset}.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig)

# ================= Figure 3: Statistics Distributions =================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Exp 3: Trajectory Pattern Statistics — Correct vs Incorrect ({args.dataset.upper()})",
    fontsize=12, fontweight="bold",
)

c_dec = [e["stats"]["decline_ratio"] for e in correct_entries]
i_dec = [e["stats"]["decline_ratio"] for e in incorrect_entries]
c_drop = [e["stats"]["max_drop"] for e in correct_entries]
i_drop = [e["stats"]["max_drop"] for e in incorrect_entries]
c_slope = [e["stats"]["slope"] for e in correct_entries]
i_slope = [e["stats"]["slope"] for e in incorrect_entries]

for ax, c_vals, i_vals, xlabel, title in [
    (axes[0], c_dec, i_dec,
     "Decline ratio (>0 = SC drops in 2nd half)", "SC Decline Ratio"),
    (axes[1], c_drop, i_drop,
     "Max single-sentence drop", "Max SC Drop"),
    (axes[2], c_slope, i_slope,
     "Linear slope of SC trajectory (<0 = downward)", "SC Trajectory Slope"),
]:
    ax.hist(c_vals, bins=25, alpha=0.6, color=BLUE,
            label=f"Correct (n={len(c_vals)})")
    ax.hist(i_vals, bins=25, alpha=0.6, color=RED,
            label=f"Incorrect (n={len(i_vals)})")
    ax.axvline(np.median(c_vals), color=BLUE, linestyle="--", linewidth=2,
               label=f"Correct median={np.median(c_vals):.3f}")
    ax.axvline(np.median(i_vals), color=RED, linestyle="--", linewidth=2,
               label=f"Incorrect median={np.median(i_vals):.3f}")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)

fig.tight_layout()
out3 = os.path.join(FIGURE_DIR, f"exp3_trajectory_stats_{args.dataset}.png")
fig.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig)

# ================= Figure 4: 6 Cherry-Picked Incorrect Cases =================
# Show cases with the biggest decline_ratio (most dramatic SC drop)
sorted_incorrect = sorted(incorrect_entries,
                           key=lambda e: e["stats"]["decline_ratio"], reverse=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle(
    f"Exp 3: Individual Incorrect Cases with Biggest SC Decline ({args.dataset.upper()})\n"
    f"X-axis = sentence index, Y-axis = mean SC of that sentence",
    fontsize=12, fontweight="bold",
)

for plot_idx in range(min(6, len(sorted_incorrect))):
    ax = axes[plot_idx // 3][plot_idx % 3]
    entry = sorted_incorrect[plot_idx]
    traj = entry["traj"]
    stats = entry["stats"]

    sc_vals = [t["mean_sc"] for t in traj]
    sentence_labels = [f"S{i+1}" for i in range(len(traj))]

    ax.plot(range(len(sc_vals)), sc_vals, color=RED, linewidth=2,
            marker="o", markersize=6)
    ax.fill_between(range(len(sc_vals)), sc_vals, min(sc_vals) - 0.5,
                    alpha=0.1, color=RED)

    # Annotate each point with first 20 chars of sentence
    for i, (val, t) in enumerate(zip(sc_vals, traj)):
        label = t["sentence"][:25] + "..." if len(t["sentence"]) > 25 else t["sentence"]
        ax.annotate(label, (i, val), textcoords="offset points",
                    xytext=(0, 6), fontsize=6, ha="center", rotation=15)

    q_short = entry["question"][:55] + "..." if len(entry["question"]) > 55 else entry["question"]
    a_short = entry["answer"][:30] + "..." if len(entry["answer"]) > 30 else entry["answer"]
    ax.set_title(
        f"Q: {q_short}\nA: {a_short}\n"
        f"decline={stats['decline_ratio']:.3f}, slope={stats['slope']:.4f}",
        fontsize=7.5
    )
    ax.set_xticks(range(len(sc_vals)))
    ax.set_xticklabels(sentence_labels, fontsize=8)
    ax.set_ylabel("SC", fontsize=9)

fig.tight_layout()
out4 = os.path.join(FIGURE_DIR, f"exp3_cherry_picked_{args.dataset}.png")
fig.savefig(out4, dpi=150, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close(fig)

# ================= Final Verdict =================
print(f"\n{'='*60}")
print("VERDICT: Does sentence-level SC show detectable patterns?")
print(f"{'='*60}")

med_dec_c = np.median(c_dec) if c_dec else 0
med_dec_i = np.median(i_dec) if i_dec else 0
med_slope_c = np.median(c_slope) if c_slope else 0
med_slope_i = np.median(i_slope) if i_slope else 0
gap_dec = med_dec_i - med_dec_c
gap_slope = med_slope_c - med_slope_i  # correct slope > incorrect slope = correct more stable

print(f"  Decline ratio — Correct: {med_dec_c:.4f}, Incorrect: {med_dec_i:.4f}, gap: {gap_dec:.4f}")
print(f"  Slope         — Correct: {med_slope_c:.6f}, Incorrect: {med_slope_i:.6f}, gap: {gap_slope:.6f}")

if gap_dec > 0.05 or gap_slope > 0.001:
    print("\n  --> POSITIVE signal: Incorrect answers show more declining SC trajectories.")
    print("  --> Direction A (sentence-level trajectory monitoring) is viable.")
    print("  --> Next step: define threshold (e.g., decline_ratio > 0.1 = trigger RAG)")
else:
    print("\n  --> WEAK signal: Correct and incorrect trajectories look similar.")
    print("  --> Direction A unlikely to work well.")
    print("  --> Focus on Direction B (SC-gated RAG: whole-answer SC threshold).")
