#!/usr/bin/env python3
"""Grouped bar chart for the never/gate/always three-way RAG ablation (RESULTS_MASTER §4.6).

Shows both core findings in one figure:
  (1) never -> gate huge jump  => retrieval after decomposition is necessary
  (2) gate ~= always overlap   => the per-step gate costs ~0 accuracy (efficiency, not accuracy)
"""
import os
import re
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# register a CJK font so Chinese labels render (not as boxes)
_cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_cjk)
matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# gate tau is configurable: `python make_3way_fig.py 0.8` swaps the gate bar to
# tau=0.8 and writes a separate PNG (keeps the canonical tau=0.3 figure).
GATE_TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
DS_KEYS = ["hotpotqa", "2wiki", "musique", "popqa"]
datasets = ["HotpotQA", "2Wiki", "MuSiQue", "PopQA"]
_EM_RE = re.compile(r"^EM:\s+([\d.]+)%", re.M)


def read_em(path):
    m = _EM_RE.search(open(os.path.join(SCRIPTS, path)).read())
    if not m:
        raise SystemExit(f"no EM found in {path} (run not finished?)")
    return float(m.group(1))


def gate_log(ds):
    if abs(GATE_TAU - 0.3) < 1e-9:
        return f"3way_gate_{ds}.log"
    return f"sweep_{ds}_tau{str(GATE_TAU).replace('.', 'p')}.log"


never  = [read_em(f"3way_never_{ds}.log") for ds in DS_KEYS]
gate   = [read_em(gate_log(ds)) for ds in DS_KEYS]
always = [read_em(f"3way_always_{ds}.log") for ds in DS_KEYS]
jump   = [g - n for g, n in zip(gate, never)]
print(f"gate τ={GATE_TAU}: never={never} gate={gate} always={always}")

x = np.arange(len(datasets))
w = 0.26

fig, ax = plt.subplots(figsize=(8.4, 4.8))
b_never  = ax.bar(x - w, never,  w, label="never-RAG", color="#c0392b")
b_gate   = ax.bar(x,     gate,   w, label=f"gate τ={GATE_TAU:g}", color="#27ae60")
b_always = ax.bar(x + w, always, w, label="always-RAG", color="#7f8c8d")

# value labels on every bar
for bars in (b_never, b_gate, b_always):
    for r in bars:
        ax.annotate(f"{r.get_height():.1f}",
                    (r.get_x() + r.get_width() / 2, r.get_height()),
                    textcoords="offset points", xytext=(0, 2),
                    ha="center", va="bottom", fontsize=8)

# annotate the never->gate jump (assumption (1))
for i, d in enumerate(jump):
    ax.annotate(f"+{d:.1f}", (x[i] - w / 2, max(never[i], gate[i]) + 4),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#196f3d")

ax.set_ylabel("EM")
ax.set_title("Necessity of Retrieval after Decomposition")
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylim(0, 62)
ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()

suffix = "" if abs(GATE_TAU - 0.3) < 1e-9 else f"_tau{str(GATE_TAU).replace('.', 'p')}"
out = os.path.join(SCRIPTS, "..", "slides", "figures",
                   f"fig_3way_ablation{suffix}.png")
out = os.path.normpath(out)
fig.savefig(out, dpi=150)
print(f"saved {out}")
