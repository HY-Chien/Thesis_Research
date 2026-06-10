"""Generate slide figures for progress_part1.md.
Run: /home/S113065528/miniconda3/envs/thesis/bin/python3 slides/make_figs.py
(run from main_methodology/). Outputs slides/figures/*.png. CJK via Thesis Kaiti.
"""
import json, sys, os
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
from seer.retriever import PerQuestionBM25, _tok

# --- CJK font (same Kaiti the thesis uses) ---
FONT = "/home/S113065528/Research/Thesis/fonts/chinese/Kaiti.ttf"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

def L(p): return [json.loads(l) for l in open(p) if l.strip()]
def emf(f):
    r = [json.loads(l) for l in open(f)]; e = [x["em"] for x in r if "em" in x]
    return 100 * sum(e) / len(e)

# ===================== FIG A: M3 recall A vs B-matched @5 =====================
DSm = [("2Wiki", "2wiki", "datasets/2wiki/raw/dev_first1000.jsonl"),
       ("HotpotQA", "hotpotqa", "datasets/hotpotqa/raw/dev_first1000.jsonl"),
       ("MuSiQue", "musique", "datasets/musique/raw/dev_first1000.jsonl")]
K = 5
recallA, recallB, fcA, fcB = {}, {}, {}, {}
for name, tag, devp in DSm:
    dev = {str(r["id"]): r for r in L(devp)}
    trace = L(f"results/{tag}_1000_C_maxent0.3_3ds_terse.jsonl")
    ft = hitA = hitB = fA = fB = gh = 0
    for t in trace:
        rec = dev.get(str(t["id"]))
        if not rec: continue
        paras = rec["paragraphs"]; gold = {p["idx"] for p in paras if p.get("is_supporting")}
        if not gold: continue
        bm = PerQuestionBM25(paras)
        sA = bm.bm25.get_scores(_tok(t["question"]))
        Aidx = {paras[i]["idx"] for i in sorted(range(len(paras)), key=lambda i: -sA[i])[:K]}
        subqs = [s.get("sub_q") or s.get("raw_sub_q") for s in t.get("steps", [])
                 if (s.get("sub_q") or s.get("raw_sub_q"))] or [t["question"]]
        ss = [bm.bm25.get_scores(_tok(q)) for q in subqs]
        smax = [max(s[i] for s in ss) for i in range(len(paras))]
        Bidx = {paras[i]["idx"] for i in sorted(range(len(paras)), key=lambda i: -smax[i])[:K]}
        ft += len(gold); hitA += len(gold & Aidx); hitB += len(gold & Bidx)
        fA += (len(gold & Aidx) == len(gold)); fB += (len(gold & Bidx) == len(gold)); gh += 1
    recallA[name], recallB[name] = 100 * hitA / ft, 100 * hitB / ft
    fcA[name], fcB[name] = 100 * fA / gh, 100 * fB / gh
    print(f"M3 {name}: recall {recallA[name]:.1f}->{recallB[name]:.1f}  fullcov {fcA[name]:.1f}->{fcB[name]:.1f}")

names = [n for n, _, _ in DSm]; x = np.arange(len(names)); w = 0.34
fig, ax = plt.subplots(1, 1, figsize=(6, 4.3))
A = [fcA[n] for n in names]; B = [fcB[n] for n in names]
ax.bar(x - w/2, A, w, label="原問題（直接檢索）", color="#b0b7c6")
ax.bar(x + w/2, B, w, label="拆解問題（子問題檢索）", color="#2f6fb0")
for i in range(len(names)):
    ax.text(x[i] + w/2, B[i] + 1.5, f"+{B[i]-A[i]:.0f}", ha="center",
            fontsize=11, color="#1a4d80", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 108)
ax.set_title("Full-coverage@5（整題所有 gold 都撈到）", fontsize=12); ax.set_ylabel("%")
ax.legend(fontsize=9, loc="lower right"); ax.grid(axis="y", alpha=.3)
fig.suptitle("拆解讓檢索更準（k=5）", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("slides/figures/fig_m3_recall.png", dpi=150)
print("saved fig_m3_recall.png")

# ===================== FIG B: router EM vs tau (2x2) =====================
taus = [0.5, 0.8, 1.2]
cfg = {"2Wiki": ("2wiki", "wiki_k8", "single_k8"),
       "HotpotQA": ("hotpotqa", "wiki_k8", "wiki_single_k8"),
       "MuSiQue": ("musique", "wiki_k8", "single_k8"),
       "PopQA": ("popqa", "wiki_top3", None)}
fig, axs = plt.subplots(2, 2, figsize=(11, 7.2)); axs = axs.ravel()
for ax, (name, (tag, decsuf, sinsuf)) in zip(axs, cfg.items()):
    rvals = [emf(f"results/{tag}_1000_router05.jsonl"),
             emf(f"results/{tag}_1000_router08.jsonl"),
             emf(f"results/{tag}_1000_router12.jsonl")]
    dec = emf(f"results/{tag}_1000_{decsuf}.jsonl")
    ax.plot(taus, rvals, "-o", color="#2f6fb0", lw=2.2, ms=8, label="router（依熵 skip）")
    ax.axhline(dec, ls="--", color="#2ca02c", lw=1.8, label=f"全分解 {dec:.1f}")
    if sinsuf:
        sin = emf(f"results/{tag}_1000_{sinsuf}.jsonl")
        ax.axhline(sin, ls=":", color="#d62728", lw=1.8, label=f"全單步 {sin:.1f}")
    for t, v in zip(taus, rvals):
        ax.annotate(f"{v:.1f}", (t, v), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, fontweight="bold")
    win = name == "HotpotQA"
    ax.set_title(name + ("　【router 勝兩種純策略】" if win else "　【一 skip 就變差】"),
                 fontsize=12, color=("#1a7d1a" if win else "#b03030"), fontweight="bold")
    ax.set_xlabel("τ（越大越常 skip 拆解）"); ax.set_ylabel("EM"); ax.set_xticks(taus)
    ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="best")
fig.suptitle("M2 組B：自適應 router EM vs τ（honest wiki-21M）— 單一全域 τ 救不了",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("slides/figures/fig_router_tau.png", dpi=150)
print("saved fig_router_tau.png")

# ===================== FIG C: confident-hallucination subtype 分佈 =====================
# 來源:deeprag_compare/check/step4_output/subtype_labels_1000q.csv(GPT-OSS judge, 998 題)
sub = [("HI_correct\n(自信且對)", 264, "#2ca02c"),
       ("B2 鏈斷/隨機填", 443, "#d62728"),
       ("B1 知識錯", 178, "#ff7f0e"),
       ("Other", 107, "#9aa0aa"),
       ("B3 角色混淆", 6, "#7b4fb0")]
fig, ax = plt.subplots(figsize=(10, 3.6))
ys = np.arange(len(sub))[::-1]
ax.barh(ys, [s[1] for s in sub], color=[s[2] for s in sub])
for y, (lab, n, _) in zip(ys, sub):
    ax.text(n + 6, y, f"{n}", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in sub], fontsize=11)
ax.set_xlabel("題數(共 998)"); ax.set_xlim(0, 500)
ax.set_title("Confident hallucination 不是單一現象:GPT-OSS judge 子型分佈",
             fontsize=13, fontweight="bold")
ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig("slides/figures/fig_subtype_dist.png", dpi=150)
print("saved fig_subtype_dist.png")

# ===================== FIG D: 單一訊號分不出「自信對 vs 自信錯」 =====================
# step1_output_100q/group_comparison.md(HI_correct vs HI_wrong, 100Q);n_uniq 來自 B2 row(memory)
feats = [("max_ent\n(SEER gate 用的)", 0.488, "#d62728"),
         ("mean_entropy", 0.586, "#e8772e"),
         ("min_margin", 0.579, "#e8772e"),
         ("first-ent mean-lp", 0.744, "#3a7fbf"),
         ("first-ent min-lp", 0.762, "#3a7fbf"),
         ("multi-sample n_uniq", 0.81, "#2f6fb0")]
fig, ax = plt.subplots(figsize=(10, 4.2))
xs = np.arange(len(feats))
ax.bar(xs, [f[1] for f in feats], color=[f[2] for f in feats])
ax.axhline(0.5, ls="--", color="gray", lw=1.3); ax.text(len(feats)-0.5, 0.51, "亂猜 0.5", fontsize=9, color="gray", ha="right")
for x, (lab, v, _) in zip(xs, feats):
    ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xs); ax.set_xticklabels([f[0] for f in feats], fontsize=9.5)
ax.set_ylabel("AUROC(分辨 自信對 vs 自信錯)"); ax.set_ylim(0.4, 0.9)
ax.set_title("方向1:在「已經自信」的子集裡,單一訊號都不夠(熵 ≈ 亂猜)",
             fontsize=12.5, fontweight="bold")
ax.grid(axis="y", alpha=.3)
ax.annotate("B3 角色混淆:token/SC 全盲\nB1 知識錯:自信且一致 → 需 RAG",
            xy=(0.015, 0.97), xycoords="axes fraction", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#fff3e0", ec="#e8a33d"))
fig.tight_layout(); fig.savefig("slides/figures/fig_detector_auroc.png", dpi=150)
print("saved fig_detector_auroc.png")
