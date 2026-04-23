"""
fig_exp10_case_study.py
========================
Visualise one complete Dynamic SC-RAG (exp10) inference trace.

Example: "who wrote the world is waiting for the sunrise"

Outputs:
    results/exp10_case_study.json   — full annotated flow
    figures/fig_exp10_case_study.png
"""

import os, sys, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

BASE    = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
SRC     = os.path.join(BASE, "results", "exp10_dynamic_nq_results.json")
OUT_JSON = os.path.join(BASE, "results",  "exp10_case_study.json")
OUT_FIG  = os.path.join(BASE, "figures",  "fig_exp10_case_study.png")

# ── 1. Load target example ─────────────────────────────────────────────────
with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

TARGET_ID = "single_nq_dev_3720"
r = next(x for x in data if x["id"] == TARGET_ID)

question      = r["question"]
ground_truths = r["ground_truths"]
baseline_cot  = r["baseline_cot"]
baseline_ans  = r["baseline_final"]
dynamic_cot   = r["dynamic_cot"]
dynamic_ans   = r["dynamic_final"]
sentence_scs  = r["sentence_scs"]
trigger_events = r["trigger_events"]
n_retrievals  = r["n_retrievals"]

# ── 2. Reconstruct sentence texts from dynamic_cot ────────────────────────
def split_sentences(text):
    """Split on sentence-ending punctuation, keep the delimiter."""
    raw = re.split(r'(?<=[.!?])\s+|\n+', text.strip())
    sents = [s.strip() for s in raw if s.strip()]
    return sents

dynamic_sents  = split_sentences(dynamic_cot)
baseline_sents = split_sentences(baseline_cot)

# Align with sentence_scs (may differ if tokeniser splits differently)
n_sents = len(sentence_scs)
# Pad / trim sentences to match SC list length
while len(dynamic_sents) < n_sents:
    dynamic_sents.append("")
dynamic_sents = dynamic_sents[:n_sents]

# ── 3. Build annotated flow dict ──────────────────────────────────────────
trigger_idx = {ev["sentence_idx"]: ev for ev in trigger_events}

sentences_flow = []
for i, (sc, sent) in enumerate(zip(sentence_scs, dynamic_sents)):
    ev = trigger_idx.get(i)
    entry = {
        "sentence_idx": i,
        "text": sent,
        "mean_sc": round(sc, 3),
        "triggered_rag": ev is not None,
    }
    if ev:
        entry["trigger_reason"] = ev["reason"]
        entry["trigger_step"]   = ev["step"]
        entry["retrieved_titles"] = ev["titles"]
        entry["partial_cot_tail"] = ev["partial_cot_tail"]
    sentences_flow.append(entry)

case_study = {
    "id": TARGET_ID,
    "question": question,
    "ground_truths": ground_truths,
    "sc_threshold": 12.0,
    "n_retrievals": n_retrievals,
    "baseline": {
        "answer": baseline_ans,
        "correct": False,
        "cot": baseline_cot,
        "overall_sc": round(r["baseline_sc"], 3),
    },
    "dynamic": {
        "answer": dynamic_ans,
        "correct": True,
        "cot": dynamic_cot,
        "overall_sc": round(r["dynamic_sc"], 3),
        "sentence_scs": [round(s, 3) for s in sentence_scs],
    },
    "sentences_flow": sentences_flow,
}

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(case_study, f, ensure_ascii=False, indent=2)
print(f"Saved JSON: {OUT_JSON}")

# ── 4. Visualisation ──────────────────────────────────────────────────────
SC_THRESH = 12.0
BLUE   = "#3A72C0"
ORANGE = "#E07020"
GREEN  = "#2E9E5B"
RED    = "#CC3333"
GRAY   = "#888888"
LIGHT  = "#F4F4F4"

fig = plt.figure(figsize=(20, 18))
# Layout: title row / SC chart / two CoT text boxes / comparison bar
gs = fig.add_gridspec(4, 2, height_ratios=[2.2, 1.0, 1.0, 0.6],
                      hspace=0.55, wspace=0.06)

ax_sc    = fig.add_subplot(gs[0, :])   # SC trajectory (full width)
ax_base  = fig.add_subplot(gs[1, 0])   # Baseline CoT text
ax_dyn   = fig.add_subplot(gs[1, 1])   # Dynamic CoT text
ax_sent  = fig.add_subplot(gs[2, :])   # Per-sentence SC table
ax_bar   = fig.add_subplot(gs[3, :])   # Accuracy comparison bar

fig.suptitle(
    f"Exp10 Case Study — Dynamic SC-Triggered RAG\n"
    f"Question: \"{question}\"\n"
    f"Ground Truth: {ground_truths[0]}   |   "
    f"No-RAG Answer: \"{baseline_ans}\" ✗   |   "
    f"Dynamic-RAG Answer: \"{dynamic_ans}\" ✓",
    fontsize=12, fontweight="bold", y=0.995
)

# ── Panel A: SC trajectory ─────────────────────────────────────────────────
xs = list(range(n_sents))
colors = [RED if sc < SC_THRESH else BLUE for sc in sentence_scs]

ax_sc.axhspan(0, SC_THRESH, alpha=0.08, color="red", zorder=0)
ax_sc.axhline(SC_THRESH, color="red", linewidth=1.5, linestyle="--",
              label=f"SC threshold = {SC_THRESH:.0f}", zorder=2)

ax_sc.plot(xs, sentence_scs, color=GRAY, linewidth=1.5, zorder=3)
for x, sc, c in zip(xs, sentence_scs, colors):
    ax_sc.scatter(x, sc, color=c, s=120, zorder=5, linewidths=0)
    ax_sc.text(x, sc + 0.4, f"{sc:.1f}", ha="center", fontsize=8.5,
               color=c, fontweight="bold")

# Mark trigger events
for ev in trigger_events:
    i   = ev["sentence_idx"]
    sc  = sentence_scs[i]
    ax_sc.annotate(
        f"  RAG #{trigger_events.index(ev)+1}\n  ({ev['reason']})",
        xy=(i, sc), xytext=(i + 0.3, sc - 2.2),
        fontsize=9, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5),
        zorder=6,
    )
    # Vertical line at trigger
    ax_sc.axvline(i, color=RED, linewidth=1.0, linestyle=":", alpha=0.6)

# SC recovery annotation
if len(trigger_events) >= 1:
    t_idx = trigger_events[-1]["sentence_idx"]
    if t_idx + 2 < n_sents:
        ax_sc.annotate(
            "SC recovers after RAG\n(model becomes certain again)",
            xy=(t_idx + 2, sentence_scs[t_idx + 2]),
            xytext=(t_idx + 2.5, sentence_scs[t_idx + 2] + 1.5),
            fontsize=8.5, color=GREEN, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.3),
        )

ax_sc.set_xlim(-0.5, n_sents - 0.5)
ax_sc.set_ylim(0, max(sentence_scs) + 3)
ax_sc.set_xticks(xs)
ax_sc.set_xticklabels([f"S{i}" for i in xs], fontsize=9)
ax_sc.set_ylabel("Sentence mean SC", fontsize=11)
ax_sc.set_title(
    "Sentence-level SC Trajectory\n"
    "Blue = SC ≥ threshold (model confident)  |  "
    "Red = SC < threshold → RAG triggered  |  "
    "SC recovers after retrieval",
    fontsize=9, loc="left"
)
ax_sc.legend(fontsize=9, loc="upper right")

# ── Panel B: Baseline CoT ──────────────────────────────────────────────────
ax_base.axis("off")
import textwrap
b_wrapped = "\n".join(textwrap.wrap(baseline_cot, width=60))
ax_base.text(0.02, 0.98, f"[Baseline — No RAG]\n\n{b_wrapped}",
             transform=ax_base.transAxes,
             fontsize=8, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF0F0",
                       edgecolor=RED, linewidth=1.5))
ax_base.set_title(
    f"Baseline Answer: \"{baseline_ans}\"  ✗  (Wrong)",
    fontsize=10, color=RED, fontweight="bold"
)

# ── Panel C: Dynamic CoT ───────────────────────────────────────────────────
ax_dyn.axis("off")
# Highlight the part after first retrieval
pre_rag = dynamic_cot[:trigger_events[0]["step"] * 3]  # rough split
d_wrapped = "\n".join(textwrap.wrap(dynamic_cot, width=60))
ax_dyn.text(0.02, 0.98, f"[Dynamic RAG CoT]\n\n{d_wrapped}",
            transform=ax_dyn.transAxes,
            fontsize=8, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0FFF0",
                      edgecolor=GREEN, linewidth=1.5))
ax_dyn.set_title(
    f"Dynamic Answer: \"{dynamic_ans}\"  ✓  (Correct)",
    fontsize=10, color=GREEN, fontweight="bold"
)

# ── Panel D: Per-sentence table ────────────────────────────────────────────
ax_sent.axis("off")

col_labels = ["Sent", "SC", "Trigger?", "Retrieved titles", "Sentence text (truncated)"]
rows_data  = []
for entry in sentences_flow:
    i    = entry["sentence_idx"]
    sc   = entry["mean_sc"]
    trig = "✓ RAG" if entry["triggered_rag"] else "—"
    titles = ", ".join(entry["retrieved_titles"][:2]) if entry.get("retrieved_titles") else "—"
    text   = entry["text"][:65] + ("…" if len(entry["text"]) > 65 else "")
    rows_data.append([f"S{i}", f"{sc:.2f}", trig, titles, text])

table = ax_sent.table(
    cellText=rows_data,
    colLabels=col_labels,
    cellLoc="left", loc="center",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.auto_set_column_width([0, 1, 2, 3, 4])

# Colour header
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Colour triggered rows
for entry in sentences_flow:
    i = entry["sentence_idx"] + 1  # +1 for header row
    if entry["triggered_rag"]:
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#FFE8E8")
    elif entry["mean_sc"] < SC_THRESH:
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#FFF4F4")
    else:
        for j in range(len(col_labels)):
            table[i, j].set_facecolor("#F8F8FF" if i % 2 == 0 else "white")

ax_sent.set_title("Per-sentence Details  (pink rows = RAG triggered)", fontsize=10,
                   fontweight="bold", pad=8)

# ── Panel E: Accuracy comparison bar ──────────────────────────────────────
ax_bar.axis("off")

# Simple text comparison
ax_bar.text(0.05, 0.75,
    f"No-RAG CoT:       \"{baseline_ans}\"  ✗  (overall_sc = {r['baseline_sc']:.2f})",
    transform=ax_bar.transAxes, fontsize=10.5, color=RED,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF0F0", edgecolor=RED))

ax_bar.text(0.05, 0.25,
    f"Dynamic SC-RAG:   \"{dynamic_ans}\"  ✓  (overall_sc = {r['dynamic_sc']:.2f})",
    transform=ax_bar.transAxes, fontsize=10.5, color=GREEN,
    fontfamily="monospace",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0FFF0", edgecolor=GREEN))

ax_bar.text(0.72, 0.5,
    f"Ground truth: {ground_truths[0]}",
    transform=ax_bar.transAxes, fontsize=11, color="#333333",
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFACD", edgecolor=ORANGE))

fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved figure: {OUT_FIG}")
print("Done.")
