#!/usr/bin/env python3
"""Final-system three-way retrieval ablation with retrieval-rate highlight."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "Thesis" / "figures" / "fig_3way_ablation_final_em_highlight.png"
RESULTS = ROOT / "results" / "tlsum_fixA_step_gate_final.json"

dataset_keys = ["hotpotqa", "2wiki", "musique", "popqa"]
datasets = ["HotpotQA", "2Wiki", "MuSiQue", "PopQA"]

with RESULTS.open(encoding="utf-8") as handle:
    results = json.load(handle)["results"]

# Match the final-system EM values shown in the user's reference figure.
never = [results[ds]["never"]["em"] for ds in dataset_keys]
accuracy_first = [results[ds]["tau_0.002305"]["em"] for ds in dataset_keys]
always = [results[ds]["always"]["em"] for ds in dataset_keys]
gate_retrieval = 100 * sum(
    results[ds]["tau_0.002305"]["retrieved_steps"] for ds in dataset_keys
) / sum(results[ds]["tau_0.002305"]["total_steps"] for ds in dataset_keys)

x = np.arange(len(datasets))
width = 0.24

fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=170)
bars_never = ax.bar(
    x - width,
    never,
    width,
    label="Never retrieve (0%)",
    color="#c05a47",
)
bars_accuracy = ax.bar(
    x,
    accuracy_first,
    width,
    label=rf"Tail-Leakage gate ($\tau=0.002305$; {gate_retrieval:.2f}%)",
    color="#27ae60",
)
bars_always = ax.bar(
    x + width,
    always,
    width,
    label="Always retrieve (100%)",
    color="#7f8c8d",
)

for bars in (bars_never, bars_accuracy, bars_always):
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)

ax.set_ylabel("Exact Match (EM)")
ax.set_title("Tail-Leakage Gating after Decomposition", fontsize=14, pad=12)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 61)
legend = ax.legend(loc="upper left", fontsize=8.6, frameon=False)
legend.get_texts()[1].set_color("#c62828")
legend.get_texts()[1].set_fontweight("bold")
ax.grid(axis="y", linestyle="--", alpha=0.28)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT}")
