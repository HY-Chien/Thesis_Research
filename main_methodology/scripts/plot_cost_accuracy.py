#!/usr/bin/env python3
"""Cost-accuracy tradeoff for the per-step retrieval gate (tau sweep).

Reads run_seek.py logs from the tau sweep plus the existing 3-way endpoints
(never tau=1e9, gate tau=0.3, always tau=0) and reports, per dataset, the
retrieval rate (cost) vs EM/F1 (accuracy) at each tau. Emits a markdown table
and a cost-accuracy scatter/line figure.

Usage: python scripts/plot_cost_accuracy.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES_DIR = ROOT.parent
DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
SWEEP_TAUS = [0.1, 0.2, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

EM_RE = re.compile(r"^EM:\s+([\d.]+)%", re.M)
F1_RE = re.compile(r"^F1:\s+([\d.]+)%", re.M)
RETR_RE = re.compile(r"^retrieval:\s+\d+/\d+ steps = ([\d.]+)%", re.M)


def parse_log(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="ignore")
    em = EM_RE.search(txt)
    f1 = F1_RE.search(txt)
    rr = RETR_RE.search(txt)
    if not (em and rr):
        return None
    return {
        "em": float(em.group(1)),
        "f1": float(f1.group(1)) if f1 else float("nan"),
        "retr": float(rr.group(1)),
    }


def collect(ds: str):
    """Return sorted list of points (tau, retr_rate, em, f1) for one dataset."""
    pts = []
    # endpoints from the 3-way ablation logs
    endpoints = {
        1e9: ROOT / f"3way_never_{ds}.log",
        0.3: ROOT / f"3way_gate_{ds}.log",
        0.0: ROOT / f"3way_always_{ds}.log",
    }
    for tau, log in endpoints.items():
        r = parse_log(log)
        if r:
            pts.append((tau, r["retr"], r["em"], r["f1"]))
    # swept points
    for tau in SWEEP_TAUS:
        tag = str(tau).replace(".", "p")
        r = parse_log(ROOT / f"sweep_{ds}_tau{tag}.log")
        if r:
            pts.append((tau, r["retr"], r["em"], r["f1"]))
    # de-dup by tau, sort by retrieval rate ascending (cost axis)
    seen = {}
    for tau, rr, em, f1 in pts:
        seen[tau] = (tau, rr, em, f1)
    return sorted(seen.values(), key=lambda x: x[1])


def main() -> None:
    all_pts = {ds: collect(ds) for ds in DATASETS}

    # markdown table
    lines = ["# Cost-Accuracy Tradeoff — per-step retrieval gate (tau sweep)", "",
             "honest wiki-21M, 1000Q, mode C, max_ent, path-aware title. "
             "Cost = % decompose steps that retrieve; tau=0.3 is canonical.", ""]
    for ds in DATASETS:
        pts = all_pts[ds]
        if not pts:
            continue
        lines.append(f"## {ds}")
        lines.append("| tau | retrieval% | EM | F1 |")
        lines.append("|----:|-----------:|---:|---:|")
        for tau, rr, em, f1 in sorted(pts, key=lambda x: x[0]):
            tau_s = "1e9" if tau >= 1e8 else f"{tau:g}"
            note = " (canonical)" if abs(tau - 0.3) < 1e-9 else ""
            lines.append(f"| {tau_s}{note} | {rr:.1f} | {em:.1f} | {f1:.1f} |")
        lines.append("")
    out_md = RES_DIR / "results" / "cost_accuracy_tradeoff.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_md}")
    print("\n".join(lines))

    # figure: EM vs retrieval rate, one line per dataset
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"hotpotqa": "#d62728", "2wiki": "#1f77b4",
              "musique": "#2ca02c", "popqa": "#9467bd"}
    for ds in DATASETS:
        pts = all_pts[ds]
        if not pts:
            continue
        xs = [p[1] for p in pts]   # retrieval rate (cost)
        ys = [p[2] for p in pts]   # EM
        ax.plot(xs, ys, "-o", color=colors[ds], label=ds, markersize=5)
        # annotate canonical point
        for tau, rr, em, f1 in pts:
            if abs(tau - 0.3) < 1e-9:
                ax.annotate("τ=0.3", (rr, em), textcoords="offset points",
                            xytext=(4, 5), fontsize=8, color=colors[ds])
    ax.set_xlabel("Retrieval rate (% of decompose steps that retrieve) — cost")
    ax.set_ylabel("EM (%) — accuracy")
    ax.set_title("Cost-Accuracy Tradeoff of the Per-Step Retrieval Gate (τ)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="dataset")
    fig.tight_layout()
    out_png = RES_DIR / "results" / "cost_accuracy_tradeoff.png"
    fig.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
