#!/usr/bin/env python3
"""Signal-agnostic replay analysis for the sub-question retrieval gate.

Compares candidate gate signals (entropy family vs margin/TARG/MAQA family)
on their ability to decide WHEN retrieval is needed, using only existing
JSONL traces — no new inference.

Per dataset it joins three sources by question id:
  * ``{ds}_3way_never.jsonl``  -> em_never  (parametric-only endpoint)
  * ``{ds}_3way_always.jsonl`` -> em_always (always-retrieval endpoint)
  * one dual-sweep trace       -> per-step signal values (all signals from the
    SAME trace so the cross-signal comparison is fair)

If no dual trace exists yet (MuSiQue / PopQA until the sweep finishes), it
falls back to the never trace, which only carries the entropy-family signals.

Analyses:
  1. AUROC of each signal for "parametric-only fails" (needs retrieval) and
     for "retrieval flips wrong->right"; plus a step-1-only AUROC variant
     (step-1 signals are identical across traces, i.e. chain-independent).
  2. Matched-budget curves: at each retrieval budget B%, the top-B% most
     uncertain questions take em_always and the rest em_never. Removes the
     retrieval-rate confound present in end-to-end tau sweeps.
  3. Separation stats (group means / Cohen's d) between questions the
     parametric-only run gets right vs wrong.

Caveat (also printed in the report): the budget simulation is question-level
(retrieve for all steps or none) while the deployed gate is per-step; it is a
signal-quality comparison, not a full-pipeline replay.

Usage:
  python scripts/replay_gate_signal_analysis.py [--datasets hotpotqa 2wiki ...]
      [--results-dir results] [--out results/gate_signal_replay_analysis.md]
      [--figures]
"""

import argparse
import json
import math
import os
import random
from collections import OrderedDict

# signal name -> True if HIGHER value means MORE uncertain.
# Values are mapped to an uncertainty score u (higher = more uncertain):
#   u = value        if higher_is_uncertain
#   u = -value       otherwise
SIGNALS = OrderedDict(
    [
        ("max_ent", True),
        ("mean_ent", True),
        ("min_lp", False),
        ("maqa_margin", False),
        ("mean_margin", False),
        ("max_margin", False),
        ("targ_unc", True),
        ("targ_logit_gap", False),
    ]
)

BUDGETS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
DEFAULT_DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]


def load_jsonl(path):
    recs = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            recs[r["id"]] = r
    return recs


def get_steps(rec):
    """Steps live at top level in 3way traces, under 'decomp' in dual traces."""
    if "steps" in rec:
        return rec["steps"]
    if "decomp" in rec and isinstance(rec["decomp"], dict):
        return rec["decomp"].get("steps", [])
    return []


def pick_signal_source(results_dir, ds):
    """Prefer the dual-sweep trace whose retrieval pattern is closest to
    always-RAG (its step chain then best matches the em_always endpoint).
    Fall back to the never trace (entropy signals only)."""
    candidates = []
    for fn in sorted(os.listdir(results_dir)):
        if fn.startswith(f"{ds}_dual_") and fn.endswith(".jsonl"):
            candidates.append(os.path.join(results_dir, fn))
    best, best_key = None, None
    for path in candidates:
        try:
            recs = load_jsonl(path)
        except (json.JSONDecodeError, OSError):
            continue  # partial / still-running file
        n_steps = n_retr = n_margin = 0
        for r in recs.values():
            for st in get_steps(r):
                n_steps += 1
                n_retr += bool(st.get("retrieved"))
                n_margin += "mean_margin" in st
        if not n_steps or not n_margin:
            continue
        key = (len(recs), n_retr / n_steps)  # coverage first, then retrieval rate
        if best_key is None or key > best_key:
            best, best_key = (path, recs), key
    if best:
        return best[0], best[1]
    fallback = os.path.join(results_dir, f"{ds}_3way_never.jsonl")
    if os.path.exists(fallback):
        return fallback, load_jsonl(fallback)
    return None, None


def uncertainty(step, sig):
    if sig not in step:
        return None
    v = step[sig]
    if v is None:
        return None
    return v if SIGNALS[sig] else -v


def aggregate(steps, sig, step1_only=False):
    """Question-level score = most-uncertain step (max of u)."""
    us = []
    for st in steps[:1] if step1_only else steps:
        u = uncertainty(st, sig)
        if u is not None:
            us.append(u)
    return max(us) if us else None


def auroc(scores, labels):
    """Rank-based AUROC (Mann-Whitney, ties averaged). labels: 1 = positive."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # ranks are 1-based
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (ma - mb) / pooled if pooled else float("nan")


def budget_em(rows, sig, budget_pct):
    """rows: list of (u_by_signal dict, em_never, em_always)."""
    scored = [(r[0][sig], r[1], r[2]) for r in rows if r[0].get(sig) is not None]
    if not scored:
        return float("nan")
    scored.sort(key=lambda x: x[0], reverse=True)  # most uncertain first
    k = round(len(scored) * budget_pct / 100.0)
    hits = sum(ea for _, _, ea in scored[:k]) + sum(en for _, en, _ in scored[k:])
    return 100.0 * hits / len(scored)


def analyze_dataset(results_dir, ds):
    never_p = os.path.join(results_dir, f"{ds}_3way_never.jsonl")
    always_p = os.path.join(results_dir, f"{ds}_3way_always.jsonl")
    if not (os.path.exists(never_p) and os.path.exists(always_p)):
        print(f"[skip] {ds}: missing 3way never/always traces")
        return None
    never = load_jsonl(never_p)
    always = load_jsonl(always_p)
    src_path, src = pick_signal_source(results_dir, ds)
    if src is None:
        print(f"[skip] {ds}: no signal source trace")
        return None

    rows = []  # (u dict, em_never, em_always, u_step1 dict)
    n_no_steps = 0
    for qid, rec in src.items():
        if qid not in never or qid not in always:
            continue
        steps = get_steps(rec)
        if not steps:
            n_no_steps += 1
            continue
        u = {s: aggregate(steps, s) for s in SIGNALS}
        u1 = {s: aggregate(steps, s, step1_only=True) for s in SIGNALS}
        rows.append((u, never[qid]["em"], always[qid]["em"], u1))
    if not rows:
        print(f"[skip] {ds}: no joinable records")
        return None

    n = len(rows)
    em_never = 100.0 * sum(r[1] for r in rows) / n
    em_always = 100.0 * sum(r[2] for r in rows) / n
    n_up = sum(1 for r in rows if r[1] == 0 and r[2] == 1)
    n_down = sum(1 for r in rows if r[1] == 1 and r[2] == 0)

    out = {
        "dataset": ds,
        "source": os.path.basename(src_path),
        "n": n,
        "n_no_steps": n_no_steps,
        "em_never": em_never,
        "em_always": em_always,
        "n_up": n_up,
        "n_down": n_down,
        "signals": OrderedDict(),
        "budget": OrderedDict(),
        "dist": OrderedDict(),  # sig -> (u values | parametric wrong, | right)
    }

    for sig in SIGNALS:
        pts = [(r[0][sig], r[1], r[2], r[3][sig]) for r in rows if r[0].get(sig) is not None]
        if len(pts) < 50:
            continue
        scores = [p[0] for p in pts]
        need = [1 - p[1] for p in pts]  # positive = parametric-only wrong
        fix = [1 if (p[1] == 0 and p[2] == 1) else 0 for p in pts]
        s1 = [(p[3], 1 - p[1]) for p in pts if p[3] is not None]
        wrong_u = [p[0] for p in pts if p[1] == 0]
        right_u = [p[0] for p in pts if p[1] == 1]
        out["signals"][sig] = {
            "n": len(pts),
            "auroc_need": auroc(scores, need),
            "auroc_fix": auroc(scores, fix),
            "auroc_need_step1": auroc([x for x, _ in s1], [y for _, y in s1]),
            "mean_u_wrong": sum(wrong_u) / len(wrong_u) if wrong_u else float("nan"),
            "mean_u_right": sum(right_u) / len(right_u) if right_u else float("nan"),
            "cohens_d": cohens_d(wrong_u, right_u),
        }
        out["budget"][sig] = [budget_em(rows, sig, b) for b in BUDGETS]
        out["dist"][sig] = (wrong_u, right_u)

    # paired bootstrap over questions: AUROC(need) CI per signal and
    # delta-AUROC vs the reference signal (max_ent), so the entropy-vs-margin
    # gap carries an uncertainty estimate instead of a point value.
    ref = "max_ent"
    if ref in out["signals"]:
        rng = random.Random(0)
        n_boot = 1000
        boot = {s: [] for s in out["signals"]}
        idx_all = range(n)
        for _ in range(n_boot):
            sample = [rows[i] for i in rng.choices(idx_all, k=n)]
            for s in out["signals"]:
                pts = [(r[0][s], 1 - r[1]) for r in sample if r[0].get(s) is not None]
                boot[s].append(auroc([p[0] for p in pts], [p[1] for p in pts]))
        for s, st in out["signals"].items():
            bs = sorted(boot[s])
            st["ci_lo"], st["ci_hi"] = bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot)]
            if s != ref:
                deltas = sorted(a - b for a, b in zip(boot[ref], boot[s]))
                st["delta"] = out["signals"][ref]["auroc_need"] - st["auroc_need"]
                st["delta_lo"] = deltas[int(0.025 * n_boot)]
                st["delta_hi"] = deltas[int(0.975 * n_boot)]
                st["p_leq0"] = sum(1 for d in deltas if d <= 0) / n_boot

    # oracle / random baselines for budget curves
    oracle = []
    for b in BUDGETS:
        k = round(n * b / 100.0)
        fixed = min(k, n_up)
        oracle.append(100.0 * (sum(r[1] for r in rows) + fixed) / n)
    random_line = [em_never + (em_always - em_never) * b / 100.0 for b in BUDGETS]
    out["budget_oracle"] = oracle
    out["budget_random"] = random_line
    return out


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_markdown(results):
    lines = []
    lines.append("# Gate Signal Replay Analysis\n")
    lines.append(
        "Signal-agnostic comparison of retrieval-gate signals, replayed offline from "
        "existing traces (`scripts/replay_gate_signal_analysis.py`). Labels come from "
        "the 3way never/always endpoints; all signal values for a dataset are read "
        "from a single trace so the cross-signal comparison is fair.\n"
    )
    lines.append(
        "**Caveats:** (1) question-level simulation (retrieve for all steps or none), "
        "while the deployed gate is per-step; (2) step-2+ signal values are conditioned "
        "on the source trace's retrieval branch — the `AUROC(need) s1` column uses only "
        "step-1 signals, which are chain-independent, as a robustness check; "
        "(3) datasets without a finished dual sweep only cover the entropy-family "
        "signals (rerun after the MuSiQue/PopQA sweeps finish).\n"
    )
    for res in results:
        lines.append(f"\n## {res['dataset']}\n")
        lines.append(
            f"- signal source: `{res['source']}` | joined n={res['n']} "
            f"(skipped {res['n_no_steps']} without steps)"
        )
        lines.append(
            f"- EM never-RAG {res['em_never']:.1f} / always-RAG {res['em_always']:.1f} | "
            f"retrieval flips: wrong→right {res['n_up']}, right→wrong {res['n_down']}\n"
        )
        lines.append(
            "| signal | n | AUROC(need) [95% CI] | ΔAUROC vs max_ent [95% CI] | P(Δ≤0) | "
            "AUROC(need) s1 | AUROC(fix) | mean u wrong | mean u right | Cohen's d |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        ranked = sorted(
            res["signals"].items(),
            key=lambda kv: -(kv[1]["auroc_need"] if not math.isnan(kv[1]["auroc_need"]) else -1),
        )
        for sig, st in ranked:
            ci = (
                f"{fmt(st['auroc_need'])} [{fmt(st.get('ci_lo'))}, {fmt(st.get('ci_hi'))}]"
                if "ci_lo" in st else fmt(st["auroc_need"])
            )
            if "delta" in st:
                delta = f"{st['delta']:+.3f} [{st['delta_lo']:+.3f}, {st['delta_hi']:+.3f}]"
                pleq = fmt(st["p_leq0"])
            else:
                delta, pleq = "ref", "-"
            lines.append(
                f"| {sig} | {st['n']} | {ci} | {delta} | {pleq} | "
                f"{fmt(st['auroc_need_step1'])} | "
                f"{fmt(st['auroc_fix'])} | {fmt(st['mean_u_wrong'], 2)} | "
                f"{fmt(st['mean_u_right'], 2)} | {fmt(st['cohens_d'], 2)} |"
            )
        lines.append("\n**Matched-budget EM** (top-B% most-uncertain questions retrieve):\n")
        header = "| signal | " + " | ".join(f"{b}%" for b in BUDGETS) + " |"
        lines.append(header)
        lines.append("|---|" + "---:|" * len(BUDGETS))
        lines.append(
            "| _oracle_ | " + " | ".join(f"{v:.1f}" for v in res["budget_oracle"]) + " |"
        )
        for sig, _ in ranked:
            vals = res["budget"][sig]
            lines.append(f"| {sig} | " + " | ".join(fmt(v, 1) for v in vals) + " |")
        lines.append(
            "| _random_ | " + " | ".join(f"{v:.1f}" for v in res["budget_random"]) + " |"
        )

    # macro table over datasets for signals present everywhere
    if len(results) > 1:
        common = set.intersection(*(set(r["signals"]) for r in results))
        if common:
            lines.append("\n## Macro AUROC(need) across datasets\n")
            lines.append("| signal | " + " | ".join(r["dataset"] for r in results) + " | macro |")
            lines.append("|---|" + "---:|" * (len(results) + 1))
            rows = []
            for sig in common:
                vals = [r["signals"][sig]["auroc_need"] for r in results]
                rows.append((sig, vals, sum(vals) / len(vals)))
            rows.sort(key=lambda x: -x[2])
            for sig, vals, macro in rows:
                lines.append(
                    f"| {sig} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(macro)} |"
                )
    return "\n".join(lines) + "\n"


C_RIGHT = "#2a78d6"  # parametric-only correct
C_WRONG = "#e34948"  # parametric-only wrong


def save_dist_figure(results, fig_dir, best_signal, plt):
    """Experiment A0 (the motivating observation): distribution of the gate
    signal for questions the parametric-only run gets right vs wrong. The
    separation between the two groups is what licenses using the signal as a
    retrieval gate; AUROC/Cohen's d in each panel quantify it."""
    ncols = 2
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.2 * nrows), squeeze=False)
    for i, res in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        wrong, right = res["dist"].get(best_signal, ([], []))
        if not wrong or not right:
            ax.axis("off")
            continue
        lo, hi = min(wrong + right), max(wrong + right)
        bins = [lo + (hi - lo) * k / 30 for k in range(31)]
        for vals, color, label in ((right, C_RIGHT, "parametric correct"),
                                   (wrong, C_WRONG, "parametric wrong")):
            ax.hist(vals, bins=bins, density=True, histtype="stepfilled",
                    alpha=0.25, color=color)
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    lw=1.8, color=color, label=label)
            ax.axvline(sum(vals) / len(vals), color=color, ls="--", lw=1)
        st = res["signals"][best_signal]
        ax.text(0.97, 0.95,
                f"AUROC {st['auroc_need']:.3f}\nCohen's d {st['cohens_d']:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_title(res["dataset"], fontsize=10)
        ax.set_xlabel(best_signal)
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for j in range(len(results), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle(f"Exp A0 — {best_signal} distribution: parametric-only "
                 "correct vs wrong (dashed = group mean)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    path = os.path.join(fig_dir, "expA0_signal_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[figures] wrote {path}")


def save_figures(results, fig_dir, best_signal):
    """Three thesis figures:
    Experiment A0 (why gate on uncertainty at all): signal distribution for
      parametric-correct vs parametric-wrong questions.
    Experiment A (which signal should gate retrieval): AUROC(need) bar chart,
      all signals grouped by dataset.
    Experiment B (is gating worth it): per-dataset budget curve for the chosen
      signal only, against the random and oracle baselines.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[figures] matplotlib unavailable; skipping")
        return
    os.makedirs(fig_dir, exist_ok=True)

    save_dist_figure(results, fig_dir, best_signal, plt)

    # --- Experiment A: AUROC(need) bars, signals x datasets ---
    sig_order = [s for s in SIGNALS if any(s in r["signals"] for r in results)]
    n_ds = len(results)
    width = 0.8 / n_ds
    plt.figure(figsize=(1.2 * len(sig_order) + 2, 4))
    for di, res in enumerate(results):
        xs, ys = [], []
        for si, sig in enumerate(sig_order):
            if sig in res["signals"]:
                xs.append(si + (di - (n_ds - 1) / 2) * width)
                ys.append(res["signals"][sig]["auroc_need"])
        plt.bar(xs, ys, width=width, label=res["dataset"])
    plt.axhline(0.5, color="k", ls="--", lw=0.8)
    plt.text(len(sig_order) - 0.5, 0.505, "random", fontsize=7, ha="right")
    plt.xticks(range(len(sig_order)), sig_order, rotation=30, ha="right")
    plt.ylabel("AUROC: parametric-only will fail")
    plt.ylim(0.3, 0.9)
    plt.title("Exp A — Which signal should gate retrieval?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(fig_dir, "expA_auroc_need.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[figures] wrote {path}")

    # --- Experiment B: budget curve of the chosen signal vs random/oracle ---
    ncols = 2
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(9, 3.4 * nrows), squeeze=False)
    for i, res in enumerate(results):
        ax = axes[i // ncols][i % ncols]
        ax.plot(BUDGETS, res["budget_oracle"], "k--", label="oracle")
        ax.plot(BUDGETS, res["budget_random"], color="gray", ls=":", label="random")
        if best_signal in res["budget"]:
            ax.plot(BUDGETS, res["budget"][best_signal], "o-", ms=4,
                    color="tab:blue", label=best_signal)
        ax.axhline(res["em_always"], color="tab:green", lw=0.8, ls="-.",
                   label="always-RAG")
        ax.set_title(res["dataset"], fontsize=10)
        ax.set_xlabel("retrieval budget (%)")
        ax.set_ylabel("EM")
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(results), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Exp B — Selective retrieval with {best_signal}: "
                 "accuracy vs retrieval budget", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(fig_dir, "expB_budget_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[figures] wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="results/gate_signal_replay_analysis.md")
    ap.add_argument("--figures", action="store_true", help="save thesis figure PNGs")
    ap.add_argument("--fig-dir", default="results/figures/gate_signal_replay")
    ap.add_argument("--best-signal", default="max_ent",
                    help="signal shown in the Experiment B budget figure")
    args = ap.parse_args()

    results = []
    for ds in args.datasets:
        res = analyze_dataset(args.results_dir, ds)
        if res:
            results.append(res)
            print(
                f"[done] {ds}: n={res['n']} source={res['source']} "
                f"signals={list(res['signals'])}"
            )
    if not results:
        raise SystemExit("no dataset produced results")

    md = render_markdown(results)
    with open(args.out, "w") as fh:
        fh.write(md)
    print(f"[done] wrote {args.out}")
    if args.figures:
        save_figures(results, args.fig_dir, args.best_signal)


if __name__ == "__main__":
    main()
