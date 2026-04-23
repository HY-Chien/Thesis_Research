"""
fig_exp11_case_study.py
========================
Visualise two Exp11 (Decomposed SC-RAG) case studies:
  1. Fixed case   — baseline wrong, decomposed correct (RAG helped)
  2. Regressed case — baseline correct, decomposed wrong (RAG hurt)

Outputs:
    figures/fig_exp11_fixed.png
    figures/fig_exp11_regressed.png
"""

import os, json, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE     = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
SRC_NQ   = os.path.join(BASE, "results", "exp11_decomposed_nq_results.json")
OUT_DIR  = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE   = "#3A72C0"
GREEN  = "#2E9E5B"
RED    = "#CC3333"
ORANGE = "#E07020"
GRAY   = "#888888"
SC_THRESH = 12.0

# ── Load data ───────────────────────────────────────────────────────────────
with open(SRC_NQ, encoding="utf-8") as f:
    data = json.load(f)

# Fixed: use the "pass laws" example (clear factual correction)
fixed_case = next(r for r in data if r["id"] == "single_nq_dev_6331")

# Regressed: use "captain hook" — retrieval got wrong year info, led to failure
regressed_case = next(r for r in data if r["id"] == "single_nq_dev_4109")


# ── Core drawing function ────────────────────────────────────────────────────
def draw_case(r, out_path, case_type="fixed"):
    """
    Draw a 3-panel figure for one Exp11 example:
      Panel A (top):    Subquery SC bar chart with retrieve markers
      Panel B (middle): Step-by-step pipeline table
      Panel C (bottom): Baseline vs Decomposed answer comparison
    """
    question       = r["question"]
    ground_truths  = r["ground_truths"]
    subqueries     = r["subqueries"]
    subquery_ans   = r["subquery_answers"]
    subquery_scs   = r["subquery_scs"]
    subquery_retr  = r["subquery_retrieved"]
    passage_titles = r["passage_titles"]
    final_cot      = r.get("final_cot", "")
    final_ans      = r["final_answer"]
    baseline_ans   = r["baseline_final"]
    baseline_ok    = r.get("baseline_correct", False)
    decomposed_ok  = r.get("decomposed_correct", False)
    n_subq         = len(subqueries)

    is_fixed = (case_type == "fixed")
    accent   = GREEN if is_fixed else RED
    label    = "Fixed ✓ (RAG Helped)" if is_fixed else "Regressed ✗ (RAG Hurt)"

    fig = plt.figure(figsize=(22, 20))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.2, 3.5, 0.9],
                           hspace=0.55)
    ax_sc    = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_cmp   = fig.add_subplot(gs[2])

    fig.suptitle(
        f"Exp11 Case Study — Decomposed SC-RAG  [{label}]\n"
        f"Question: \"{question}\"\n"
        f"Ground Truth: {ground_truths[0]}   |   "
        f"Baseline: \"{baseline_ans[:60]}\" {'✓' if baseline_ok else '✗'}   |   "
        f"Decomposed: \"{final_ans[:60]}\" {'✓' if decomposed_ok else '✗'}",
        fontsize=11, fontweight="bold", y=0.99
    )

    # ── Panel A: Subquery SC bar chart ──────────────────────────────────────
    xs     = list(range(n_subq))
    colors = [RED if sc < SC_THRESH else BLUE for sc in subquery_scs]

    bars = ax_sc.bar(xs, subquery_scs, color=colors, edgecolor="k",
                     linewidth=0.8, width=0.5, zorder=3)
    ax_sc.axhline(SC_THRESH, color="red", linewidth=1.5, linestyle="--",
                  label=f"SC threshold = {SC_THRESH:.0f}", zorder=4)
    ax_sc.axhspan(0, SC_THRESH, alpha=0.07, color="red", zorder=0)

    for i, (bar, sc, retr) in enumerate(zip(bars, subquery_scs, subquery_retr)):
        # SC value on top of bar
        ax_sc.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.3,
                   f"SC={sc:.2f}", ha="center", fontsize=9,
                   color=RED if sc < SC_THRESH else BLUE, fontweight="bold")
        # Retrieve badge
        if retr:
            ax_sc.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() / 2,
                       "🔍 RETRIEVE", ha="center", fontsize=8.5,
                       color="white", fontweight="bold")

    # Short subquery labels on x-axis
    short_labels = [f"SQ{i+1}\n{subqueries[i][:38]}…" if len(subqueries[i]) > 38
                    else f"SQ{i+1}\n{subqueries[i]}" for i in range(n_subq)]
    ax_sc.set_xticks(xs)
    ax_sc.set_xticklabels(short_labels, fontsize=8.5)
    ax_sc.set_ylabel("Mean SC of subquery answer", fontsize=10)
    ax_sc.set_ylim(0, max(subquery_scs) + 3)
    ax_sc.set_title(
        "Stage 2: Per-subquery SC (threshold = 12.0)\n"
        "Note: SC shown is AFTER RAG regeneration. For retrieved subqueries, "
        "the original parametric SC was < 12.0 (triggering retrieval); "
        "SC recovered after passages were injected.\n"
        "Blue = confident, no retrieve  |  Red bar + 🔍 = SC < 12.0 before RAG → retrieve triggered",
        fontsize=9, loc="left"
    )
    ax_sc.legend(fontsize=9, loc="upper right")

    # ── Panel B: Pipeline table ─────────────────────────────────────────────
    ax_table.axis("off")

    col_labels = ["Stage", "Sub-question", "SC", "Retrieve?",
                  "Retrieved (top title)", "Sub-answer (truncated)"]
    rows = []

    # Stage 1: Decompose
    sq_list = "\n".join(f"SQ{i+1}: {subqueries[i][:45]}…" if len(subqueries[i])>45
                        else f"SQ{i+1}: {subqueries[i]}" for i in range(n_subq))
    rows.append(["Stage 1\nDecompose", sq_list, "—", "—", "—", "—"])

    # Stage 2: each subquery
    for i in range(n_subq):
        top_title = passage_titles[i][0] if passage_titles[i] else "—"
        # Truncate sub-answer to first 120 chars, then wrap at 55
        ans_text = subquery_ans[i].replace("\n", " ").strip()
        ans_text = ans_text[:120] + ("…" if len(ans_text) > 120 else "")
        ans_wrapped = "\n".join(textwrap.wrap(ans_text, width=55))
        # Wrap sub-question too
        sq_wrapped = "\n".join(textwrap.wrap(subqueries[i], width=40))
        # Wrap title
        title_wrapped = "\n".join(textwrap.wrap(top_title, width=30))
        sc_str   = f"{subquery_scs[i]:.3f}\n(after RAG)" if subquery_retr[i] \
                   else f"{subquery_scs[i]:.3f}"
        retr_str = "✓ Yes\n(SC<12\nbefore)" if subquery_retr[i] else "— No\n(SC≥12)"
        rows.append([
            f"Stage 2\nSQ{i+1}",
            sq_wrapped,
            sc_str,
            retr_str,
            title_wrapped,
            ans_wrapped,
        ])

    # Stage 3: synthesize
    cot_text = final_cot.replace("\n", " ").strip()
    cot_wrapped = "\n".join(textwrap.wrap(cot_text, width=55))
    rows.append(["Stage 3\nSynthesize", "Combine all\nsub-answers",
                 "—", "—", "—", cot_wrapped])

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(list(range(len(col_labels))))

    # Set row heights: header shorter, data rows taller
    n_rows = len(rows) + 1  # +1 for header
    for row_i in range(n_rows):
        for col_j in range(len(col_labels)):
            cell = table[row_i, col_j]
            if row_i == 0:
                cell.set_height(0.06)   # header
            elif "Stage 1" in rows[row_i - 1][0]:
                cell.set_height(0.12)
            elif "Stage 3" in rows[row_i - 1][0]:
                cell.set_height(0.30)
            else:
                cell.set_height(0.20)   # Stage 2 subquery rows — tall for full text

    # Header style
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Row styles
    for i, row_entry in enumerate(rows):
        row_i = i + 1
        stage = row_entry[0]
        if "Stage 1" in stage:
            bg = "#E8F4FD"
        elif "Stage 2" in stage:
            # check retrieve
            sq_idx = i - 1  # offset by Stage1 row
            if sq_idx < n_subq and subquery_retr[sq_idx]:
                bg = "#FFE8E8"   # retrieved → pink
            else:
                bg = "#F0F8F0"   # no retrieve → light green
        else:
            bg = "#FFF9E6"       # Stage 3 synthesis
        for j in range(len(col_labels)):
            table[row_i, j].set_facecolor(bg)

    ax_table.set_title(
        "Full 3-Stage Pipeline\n"
        "Pink rows = retrieval triggered  |  Green rows = parametric only  |  Yellow = synthesis",
        fontsize=9, loc="left", pad=8
    )

    # ── Panel C: Comparison ─────────────────────────────────────────────────
    ax_cmp.axis("off")

    # Baseline box
    b_color = GREEN if baseline_ok else RED
    b_mark  = "✓ Correct" if baseline_ok else "✗ Wrong"
    ax_cmp.text(0.02, 0.85,
        f"[No-RAG Baseline]  {b_mark}\n\n"
        f"\"{baseline_ans[:120]}{'…' if len(baseline_ans)>120 else ''}\"",
        transform=ax_cmp.transAxes,
        fontsize=9.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#FFF0F0" if not baseline_ok else "#F0FFF0",
                  edgecolor=b_color, linewidth=2)
    )

    # Decomposed box
    d_color = GREEN if decomposed_ok else RED
    d_mark  = "✓ Correct" if decomposed_ok else "✗ Wrong"
    ax_cmp.text(0.52, 0.85,
        f"[Decomposed SC-RAG]  {d_mark}\n\n"
        f"\"{final_ans[:120]}{'…' if len(final_ans)>120 else ''}\"",
        transform=ax_cmp.transAxes,
        fontsize=9.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="#FFF0F0" if not decomposed_ok else "#F0FFF0",
                  edgecolor=d_color, linewidth=2)
    )

    # Ground truth
    ax_cmp.text(0.27, 0.08,
        f"Ground Truth: {ground_truths[0]}",
        transform=ax_cmp.transAxes,
        fontsize=11, ha="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFFACD",
                  edgecolor=ORANGE, linewidth=1.5)
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Draw both ────────────────────────────────────────────────────────────────
draw_case(fixed_case,     os.path.join(OUT_DIR, "fig_exp11_fixed.png"),    "fixed")
draw_case(regressed_case, os.path.join(OUT_DIR, "fig_exp11_regressed.png"), "regressed")
print("Done.")
