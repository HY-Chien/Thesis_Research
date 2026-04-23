"""
Plot results for SC vs PPL experiments.

Figure 1 (exp0): SC and PPL separation — correct vs incorrect answers
  1a. Box plot: SC distribution (correct vs incorrect)
  1b. Box plot: PPL distribution (correct vs incorrect)  [y-axis zoomed to p5~p95]
  1c. ROC curve: SC vs PPL as binary classifiers for correctness

Figure 2 (exp1): Stability under query rewriting
  2a. Scatter: SC orig vs rewritten
  2b. Scatter: PPL orig vs rewritten  [x/y axis zoomed to actual range]
  2c. Bar chart: Relative delta comparison (SC vs PPL)

Also prints 15 showcase examples: SC stable but PPL unstable.
"""

import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= Paths =================
EXP0_FILE  = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/exp0_sc_vs_ppl_results.json"
EXP1_FILE  = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/exp1_stability_results.json"
OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/self_certainty/figures"
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
BLUE   = "#4C72B0"
ORANGE = "#DD8452"

# ---------- Load ----------
with open(EXP0_FILE, encoding="utf-8") as f:
    exp0 = json.load(f)
with open(EXP1_FILE, encoding="utf-8") as f:
    exp1 = json.load(f)

# ---------- Exp0 ----------
correct   = [r for r in exp0 if r["is_correct"]]
incorrect = [r for r in exp0 if not r["is_correct"]]

sc_correct    = [r["self_certainty"] for r in correct]
sc_incorrect  = [r["self_certainty"] for r in incorrect]
ppl_correct   = [r["ppl"] for r in correct   if r["ppl"] != float("inf")]
ppl_incorrect = [r["ppl"] for r in incorrect if r["ppl"] != float("inf")]

# ---------- ROC ----------
def compute_roc(scores_pos, scores_neg, higher_is_positive=True):
    labels = [1] * len(scores_pos) + [0] * len(scores_neg)
    scores = scores_pos + scores_neg
    if not higher_is_positive:
        scores = [-s for s in scores]
    pairs = sorted(zip(scores, labels), reverse=True)
    tprs, fprs = [0.0], [0.0]
    tp = fp = 0
    total_pos = sum(labels)
    total_neg = len(labels) - total_pos
    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tprs.append(tp / total_pos)
        fprs.append(fp / total_neg)
    auc = sum(
        (fprs[i+1] - fprs[i]) * (tprs[i+1] + tprs[i]) / 2
        for i in range(len(fprs) - 1)
    )
    return fprs, tprs, auc

fpr_sc,  tpr_sc,  auc_sc  = compute_roc(sc_correct,  sc_incorrect,  higher_is_positive=True)
fpr_ppl, tpr_ppl, auc_ppl = compute_roc(ppl_correct, ppl_incorrect, higher_is_positive=False)

# ---------- Exp1 ----------
sc_orig_list  = [r["sc_orig"]  for r in exp1]
sc_rew_list   = [r["sc_rew"]   for r in exp1]
ppl_orig_list = [r["ppl_orig"] for r in exp1 if r["ppl_orig"] != float("inf")]
ppl_rew_list  = [r["ppl_rew"]  for r in exp1 if r["ppl_orig"] != float("inf")]

rel_sc  = [abs(r["sc_rew"]  - r["sc_orig"])  / abs(r["sc_orig"])  * 100
           for r in exp1 if r["sc_orig"] != 0]
rel_ppl = [abs(r["ppl_rew"] - r["ppl_orig"]) / abs(r["ppl_orig"]) * 100
           for r in exp1 if r["ppl_orig"] not in (0, float("inf"))]

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    den = (sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys)) ** 0.5
    return num / den if den != 0 else 0.0

corr_sc  = pearson(sc_orig_list, sc_rew_list)
corr_ppl = pearson(ppl_orig_list, ppl_rew_list)
avg_rel_sc  = sum(rel_sc)  / len(rel_sc)
avg_rel_ppl = sum(rel_ppl) / len(rel_ppl)

# ============================================================
#  Figure 1: Exp0 — Separation
# ============================================================
fig1, axes = plt.subplots(1, 3, figsize=(14, 5))
fig1.suptitle("Exp 0: SC vs PPL — Separating Correct / Incorrect Answers",
              fontsize=13, fontweight="bold")

# 1a: SC box plot
ax = axes[0]
bp = ax.boxplot(
    [sc_correct, sc_incorrect],
    labels=["Correct", "Incorrect"],
    patch_artist=True, widths=0.5,
    medianprops=dict(color="black", linewidth=2),
)
bp["boxes"][0].set_facecolor(BLUE)
bp["boxes"][1].set_facecolor(ORANGE)
ax.set_title("Self-Certainty (SC)\n↑ higher = more confident", fontsize=10)
ax.set_ylabel("SC score")
ax.text(0.97, 0.97,
        f"Correct mean:   {sum(sc_correct)/len(sc_correct):.2f}\n"
        f"Incorrect mean: {sum(sc_incorrect)/len(sc_incorrect):.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 1b: PPL box plot — zoom y-axis to p5~p95 to avoid outlier distortion
ax = axes[1]
all_ppl = ppl_correct + ppl_incorrect
p5  = float(np.percentile(all_ppl, 5))
p95 = float(np.percentile(all_ppl, 95))
margin = (p95 - p5) * 0.2
bp = ax.boxplot(
    [ppl_correct, ppl_incorrect],
    labels=["Correct", "Incorrect"],
    patch_artist=True, widths=0.5,
    medianprops=dict(color="black", linewidth=2),
)
bp["boxes"][0].set_facecolor(BLUE)
bp["boxes"][1].set_facecolor(ORANGE)
ax.set_ylim(max(0, p5 - margin), p95 + margin)
ax.set_title("Perplexity (PPL)\n↓ lower = more confident  [y-axis: p5~p95]", fontsize=10)
ax.set_ylabel("PPL score")
ax.text(0.97, 0.97,
        f"Correct mean:   {sum(ppl_correct)/len(ppl_correct):.2f}\n"
        f"Incorrect mean: {sum(ppl_incorrect)/len(ppl_incorrect):.2f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 1c: ROC
ax = axes[2]
ax.plot(fpr_sc,  tpr_sc,  color=BLUE,   lw=2, label=f"SC  (AUC={auc_sc:.3f})")
ax.plot(fpr_ppl, tpr_ppl, color=ORANGE, lw=2, label=f"PPL (AUC={auc_ppl:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve\n(predicting correct answers)", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "exp0_separation.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")

# ============================================================
#  Figure 2: Exp1 — Stability
# ============================================================
fig2, axes = plt.subplots(1, 3, figsize=(14, 5))
fig2.suptitle("Exp 1: SC vs PPL — Stability Under Query Rewriting",
              fontsize=13, fontweight="bold")

ALPHA = 0.25

# 2a: SC scatter
ax = axes[0]
ax.scatter(sc_orig_list, sc_rew_list, alpha=ALPHA, s=15, color=BLUE)
mn = float(np.percentile(sc_orig_list + sc_rew_list, 1))
mx = float(np.percentile(sc_orig_list + sc_rew_list, 99))
ax.plot([mn, mx], [mn, mx], "k--", lw=1.5, label="y = x")
ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
ax.set_xlabel("SC (original)")
ax.set_ylabel("SC (rewritten)")
ax.set_title("Self-Certainty Stability", fontsize=11)
ax.legend(fontsize=8)
ax.text(0.05, 0.95, f"r = {corr_sc:.3f}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 2b: PPL scatter — zoom to p2~p95 to exclude extreme outliers
ax = axes[1]
ax.scatter(ppl_orig_list, ppl_rew_list, alpha=ALPHA, s=15, color=ORANGE)
combined_ppl = ppl_orig_list + ppl_rew_list
lo = float(np.percentile(combined_ppl, 2))
hi = float(np.percentile(combined_ppl, 95))
ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="y = x")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("PPL (original)")
ax.set_ylabel("PPL (rewritten)")
ax.set_title("Perplexity Stability", fontsize=11)
ax.legend(fontsize=8)
ax.text(0.05, 0.95, f"r = {corr_ppl:.3f}",
        transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

# 2c: Bar chart
ax = axes[2]
bars = ax.bar(
    ["SC", "PPL"],
    [avg_rel_sc, avg_rel_ppl],
    color=[BLUE, ORANGE], width=0.4,
    edgecolor="black", linewidth=0.8,
)
ax.set_ylabel("Avg relative change (%)")
ax.set_title("Avg |Δ| / original\n(lower = more stable)", fontsize=11)
for bar, val in zip(bars, [avg_rel_sc, avg_rel_ppl]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, max(avg_rel_sc, avg_rel_ppl) * 1.35)

fig2.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "exp1_stability.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")

# ============================================================
#  Showcase: 15 examples where SC is stable but PPL is not
# ============================================================
def find_showcase_examples(results, n=15):
    """
    Find examples where SC change is small but PPL change is large.
    Score = rel_delta_ppl - rel_delta_sc  (higher = more contrast)
    """
    scored = []
    for r in results:
        if r["sc_orig"] == 0 or r["ppl_orig"] in (0, float("inf")):
            continue
        rel_sc_val  = abs(r["sc_rew"]  - r["sc_orig"])  / abs(r["sc_orig"])
        rel_ppl_val = abs(r["ppl_rew"] - r["ppl_orig"]) / abs(r["ppl_orig"])
        scored.append((rel_ppl_val - rel_sc_val, r, rel_sc_val, rel_ppl_val))
    scored.sort(reverse=True)
    return scored[:n]

showcase = find_showcase_examples(exp1, n=15)

showcase_path = os.path.join(OUTPUT_DIR, "showcase_sc_stable_ppl_not.txt")
lines = ["=" * 70, "Top 15 examples: SC stable, PPL unstable", "=" * 70, ""]
for rank, (score, r, rel_sc_val, rel_ppl_val) in enumerate(showcase, 1):
    lines.append(f"#{rank}  [{r['id']}]")
    lines.append(f"  Original : {r['original_question']}")
    lines.append(f"  Rewritten: {r['rewritten_question']}")
    lines.append(f"  Answer (orig) : {r['answer_orig']}")
    lines.append(f"  Answer (rew)  : {r['answer_rew']}")
    lines.append(f"  SC  : {r['sc_orig']:.3f} → {r['sc_rew']:.3f}  (rel Δ = {rel_sc_val*100:.1f}%)")
    lines.append(f"  PPL : {r['ppl_orig']:.3f} → {r['ppl_rew']:.3f}  (rel Δ = {rel_ppl_val*100:.1f}%)")
    lines.append("")

with open(showcase_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"Saved: {showcase_path}")

# Also print to terminal
print("\n" + "\n".join(lines[:60]))

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
