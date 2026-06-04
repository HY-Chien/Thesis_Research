"""
Scatter plot: fe_min_lp (PPL-style) vs fe_max_entropy (SC/entropy-style)
on 1000Q HotpotQA, colored by HI_correct vs HI_wrong.

Goal: visualize whether the two signals separate confident-correct vs
confident-wrong cases identically, or whether entropy gives extra info.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def roc_auc_score(y_true, y_score):
    """Mann-Whitney U based AUROC, no sklearn needed."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    # Each pair (p, n) contributes 1 if p>n, 0.5 if equal
    all_vals = np.concatenate([pos, neg])
    ranks = pd.Series(all_vals).rank(method="average").values
    rank_pos = ranks[:n_pos]
    auc = (rank_pos.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

df = pd.read_csv("step1_output_1000q/per_qid_features.csv")
df = df.dropna(subset=["fe_min_lp_mean", "fe_max_entropy_mean"])

# Focus on HI cases (5/5 same answer = high confidence by chosen-token agreement)
hi = df[df["stratum"].isin(["HI_correct", "HI_wrong"])].copy()
hi["is_wrong"] = (hi["stratum"] == "HI_wrong").astype(int)

# x: fe_min_lp_mean   (closer to PPL: chosen-token log-prob, min over first-ent span)
# y: fe_max_entropy_mean (closer to SC: full-distribution entropy, max over first-ent span)
x = hi["fe_min_lp_mean"].values
y = hi["fe_max_entropy_mean"].values
labels = hi["is_wrong"].values  # 1 = HI_wrong (hallucination), 0 = HI_correct

# AUROC for using each axis ALONE to predict is_wrong
# Higher value -> wrong, so we use negative x (since x is log-prob, more negative = more uncertain)
auroc_lp = roc_auc_score(labels, -x)
auroc_ent = roc_auc_score(labels, y)

# Combined: simple sum of standardized features
xs = (x - x.mean()) / x.std()
ys = (y - y.mean()) / y.std()
combined = ys - xs  # entropy + neg_lp both indicate uncertainty
auroc_combined = roc_auc_score(labels, combined)

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left panel: scatter
ax = axes[0]
mask_correct = labels == 0
mask_wrong = labels == 1
ax.scatter(
    x[mask_correct],
    y[mask_correct],
    s=18,
    alpha=0.55,
    c="#2ca02c",
    label=f"HI_correct  (n={mask_correct.sum()})",
    edgecolor="none",
)
ax.scatter(
    x[mask_wrong],
    y[mask_wrong],
    s=18,
    alpha=0.55,
    c="#d62728",
    label=f"HI_wrong   (n={mask_wrong.sum()})",
    edgecolor="none",
)
ax.set_xlabel("fe_min_lp_mean  (PPL-like: min log-prob over first-entity span)")
ax.set_ylabel("fe_max_entropy_mean  (SC-like: max entropy over first-entity span)")
ax.set_title(
    f"PPL-like vs Entropy-like on first-entity span (HotpotQA 1000Q HI cases)\n"
    f"AUROC fe_min_lp={auroc_lp:.3f}   fe_max_entropy={auroc_ent:.3f}   combined={auroc_combined:.3f}"
)
ax.axvline(-0.80, color="grey", linestyle="--", linewidth=1, alpha=0.7, label="our gate th=-0.80")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Right panel: marginal histograms to see overlap on each axis
ax2 = axes[1]
bins_lp = np.linspace(x.min(), x.max(), 30)
bins_ent = np.linspace(y.min(), y.max(), 30)

ax2_top = ax2
ax2_top.hist(
    x[mask_correct], bins=bins_lp, alpha=0.55, color="#2ca02c",
    density=True, label="HI_correct fe_min_lp"
)
ax2_top.hist(
    x[mask_wrong], bins=bins_lp, alpha=0.55, color="#d62728",
    density=True, label="HI_wrong  fe_min_lp"
)
ax2_top.set_xlabel("fe_min_lp_mean")
ax2_top.set_ylabel("density")
ax2_top.set_title(f"Marginal: fe_min_lp_mean  (AUROC={auroc_lp:.3f})")
ax2_top.legend(fontsize=9)
ax2_top.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ppl_vs_entropy_scatter.png", dpi=140, bbox_inches="tight")

# second figure: entropy marginal
fig2, ax3 = plt.subplots(figsize=(6.5, 4.5))
ax3.hist(
    y[mask_correct], bins=bins_ent, alpha=0.55, color="#2ca02c",
    density=True, label="HI_correct fe_max_entropy"
)
ax3.hist(
    y[mask_wrong], bins=bins_ent, alpha=0.55, color="#d62728",
    density=True, label="HI_wrong  fe_max_entropy"
)
ax3.set_xlabel("fe_max_entropy_mean")
ax3.set_ylabel("density")
ax3.set_title(f"Marginal: fe_max_entropy_mean  (AUROC={auroc_ent:.3f})")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ppl_vs_entropy_entropy_marginal.png", dpi=140, bbox_inches="tight")

# ---- diagnostics: count cases where they DISAGREE on ranking ----
# Rank each, find where rank order differs
rank_lp = pd.Series(-x).rank(method="average")  # high = uncertain
rank_ent = pd.Series(y).rank(method="average")  # high = uncertain
rank_diff = (rank_lp - rank_ent).abs()

print("=" * 60)
print(f"HI subset: n={len(hi)} ({mask_correct.sum()} correct, {mask_wrong.sum()} wrong)")
print(f"AUROC fe_min_lp (PPL-like)    = {auroc_lp:.4f}")
print(f"AUROC fe_max_entropy (SC-like) = {auroc_ent:.4f}")
print(f"AUROC combined (z-sum)         = {auroc_combined:.4f}")
print()
print(f"Pearson r(fe_min_lp, fe_max_entropy) = {np.corrcoef(-x, y)[0,1]:.4f}")
print(f"Spearman rank disagreement (top-50 by lp vs top-50 by ent):")
top50_lp = set(pd.Series(-x).nlargest(50).index)
top50_ent = set(pd.Series(y).nlargest(50).index)
print(f"  overlap = {len(top50_lp & top50_ent)} / 50")
print(f"  only-lp = {len(top50_lp - top50_ent)},  only-ent = {len(top50_ent - top50_lp)}")
print()
print("Files written:")
print("  ppl_vs_entropy_scatter.png")
print("  ppl_vs_entropy_entropy_marginal.png")
