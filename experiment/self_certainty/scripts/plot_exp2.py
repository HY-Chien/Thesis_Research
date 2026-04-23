"""
Plot Experiment 2: Best-of-N Selection Results

Figure 1: Accuracy comparison bar chart (Greedy vs Random vs PPL vs SC vs Majority vs Oracle)
Figure 2: SC vs PPL score distribution for correct/incorrect candidates (like paper's Figure 1)
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= Paths =================
RESULT_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
SUMMARY_FILE = os.path.join(RESULT_DIR, "exp2_summary.json")
OUTPUT_DIR = os.path.join(RESULT_DIR, "figures")
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

BLUE = "#4C72B0"
ORANGE = "#DD8452"
GREEN = "#55A868"
RED = "#C44E52"
PURPLE = "#8172B3"
GRAY = "#999999"

with open(SUMMARY_FILE, encoding="utf-8") as f:
    summaries = json.load(f)

# ============================================================
#  Figure 1: Accuracy Comparison Bar Chart
# ============================================================
methods = ["Greedy", "Random", "PPL\n(Best-of-N)", "SC\n(Best-of-N)", "Self-\nConsistency", "Borda-SC", "Oracle"]
method_keys = ["accuracy_greedy", "accuracy_random", "accuracy_ppl", "accuracy_sc", "accuracy_self_consistency", "accuracy_borda_sc", "accuracy_oracle"]
colors = [GRAY, GRAY, ORANGE, BLUE, GREEN, PURPLE, RED]

fig1, axes = plt.subplots(1, len(summaries), figsize=(8 * len(summaries), 6))
if len(summaries) == 1:
    axes = [axes]

fig1.suptitle(f"Best-of-N Selection: SC vs PPL vs Baselines (N={summaries[0]['n_candidates']})",
              fontsize=14, fontweight="bold")

for idx, (s, ax) in enumerate(zip(summaries, axes)):
    values = [s[k] for k in method_keys]
    bars = ax.bar(methods, values, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"{s['dataset'].upper()}", fontsize=13)
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(axis="x", labelsize=9)

fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "exp2_accuracy_comparison.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")

# ============================================================
#  Figure 2: SC / PPL distribution for correct vs incorrect candidates
#  (Reproducing the paper's Figure 1 style)
# ============================================================
for s in summaries:
    ds = s["dataset"]
    result_file = os.path.join(RESULT_DIR, f"exp2_{ds}_results.json")
    if not os.path.exists(result_file):
        continue

    with open(result_file, encoding="utf-8") as f:
        results = json.load(f)

    # Collect all candidates' SC and PPL grouped by correctness
    sc_correct, sc_incorrect = [], []
    ppl_correct, ppl_incorrect = [], []

    for r in results:
        for c in r["candidates"]:
            if c["is_correct"]:
                sc_correct.append(c["sc"])
                if c["ppl"] != float("inf"):
                    ppl_correct.append(c["ppl"])
            else:
                sc_incorrect.append(c["sc"])
                if c["ppl"] != float("inf"):
                    ppl_incorrect.append(c["ppl"])

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f"Candidate Score Distributions — {ds.upper()} (N={s['n_candidates']})",
                  fontsize=13, fontweight="bold")

    # 2a: SC distribution
    ax = axes2[0]
    all_sc = sc_correct + sc_incorrect
    bins = np.linspace(np.percentile(all_sc, 1), np.percentile(all_sc, 99), 50)
    ax.hist(sc_correct, bins=bins, alpha=0.6, color=BLUE, label=f"Correct (n={len(sc_correct)})", density=True)
    ax.hist(sc_incorrect, bins=bins, alpha=0.6, color=ORANGE, label=f"Incorrect (n={len(sc_incorrect)})", density=True)
    ax.set_xlabel("Self-Certainty Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Self-Certainty (SC)\n↑ higher = more confident", fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(np.mean(sc_correct), color=BLUE, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(sc_incorrect), color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8)

    # 2b: Negative PPL distribution (higher = more confident, like the paper)
    ax = axes2[1]
    neg_ppl_correct = [-p for p in ppl_correct]
    neg_ppl_incorrect = [-p for p in ppl_incorrect]
    all_neg_ppl = neg_ppl_correct + neg_ppl_incorrect
    bins = np.linspace(np.percentile(all_neg_ppl, 1), np.percentile(all_neg_ppl, 99), 50)
    ax.hist(neg_ppl_correct, bins=bins, alpha=0.6, color=BLUE, label=f"Correct (n={len(neg_ppl_correct)})", density=True)
    ax.hist(neg_ppl_incorrect, bins=bins, alpha=0.6, color=ORANGE, label=f"Incorrect (n={len(neg_ppl_incorrect)})", density=True)
    ax.set_xlabel("Negative Perplexity (−PPL)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Negative Perplexity (−PPL)\n↑ higher = more confident", fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(np.mean(neg_ppl_correct), color=BLUE, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.axvline(np.mean(neg_ppl_incorrect), color=ORANGE, linestyle="--", linewidth=1.5, alpha=0.8)

    fig2.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, f"exp2_{ds}_distribution.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")

# ============================================================
#  Figure 3: Combined summary table
# ============================================================
fig3, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")
ax.set_title(f"Best-of-N Selection Summary (N={summaries[0]['n_candidates']}, T={summaries[0]['temperature']})",
             fontsize=13, fontweight="bold", pad=15)

col_labels = ["Dataset", "Greedy", "Random", "PPL (BoN)", "SC (BoN)", "Self-Consist.", "Borda-SC", "Oracle"]
rows = []
for s in summaries:
    row = [
        s["dataset"].upper(),
        f"{s['accuracy_greedy']:.1f}%",
        f"{s['accuracy_random']:.1f}%",
        f"{s['accuracy_ppl']:.1f}%",
        f"{s['accuracy_sc']:.1f}%",
        f"{s['accuracy_self_consistency']:.1f}%",
        f"{s['accuracy_borda_sc']:.1f}%",
        f"{s['accuracy_oracle']:.1f}%",
    ]
    rows.append(row)

table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2.0)

for j in range(len(col_labels)):
    table[0, j].set_facecolor("#D5E8D4")
    table[0, j].set_text_props(fontweight="bold")

# Highlight SC vs PPL winner
for i, s in enumerate(summaries, 1):
    if s["accuracy_sc"] > s["accuracy_ppl"]:
        table[i, 4].set_facecolor("#D6EAF8")  # SC wins - blue
    else:
        table[i, 3].set_facecolor("#FADBD8")  # PPL wins - orange
    # Highlight Borda-SC
    table[i, 6].set_facecolor("#E8F8E8")

fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "exp2_summary_table.png")
fig3.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")

print(f"\nAll figures saved to: {OUTPUT_DIR}")
