"""Step A — Subtype-aware gate vs single-signal gate comparison.

Joins:
  - multi_sample_dump_1000q.jsonl (stratum, em_rate, n_uniq from CoT 5-sample)
  - step3_output_1000q/role_overlap_v2_step1.csv (v2_b3_rate)
  - production_sweep_output/hotpotqa_per_q_v2.jsonl (nocot_em, rag_em, neg_max_ent)
  - step4_judge_gptoss_1000q.json (LLM judge subtype label)

Computes EM under 5 gate strategies at multiple budgets:
  A always-NR    : never retrieve (nocot_em for all)
  B always-RAG   : always retrieve (rag_em for all)
  C neg_max_ent  : sort by neg_max_ent ascending; retrieve top-k%
  D subtype-aware (binary): retrieve iff label in {B1, B2, Other}; HI_correct+B3 → NR
  E subtype-aware × neg_max_ent : top-k% by neg_max_ent ascending RESTRICTED to retrieve-eligible

Outputs:
  step4_output/subtype_labels_1000q.csv   per-qid: label, em_rate, n_uniq, neg_max_ent, nocot_em, rag_em
  step4_output/gate_comparison.md          5 strategies × 4 budgets EM table
  step4_output/gate_comparison.csv
  step4_output/budget_em_curve.png         x=budget, y=EM, multiple lines

Usage:
  python step5_gate_comparison.py
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path("/home/S113065528/Research/deeprag_compare/check")
OUT = ROOT / "step4_output"

DUMP_FILE = ROOT / "multi_sample_dump_1000q.jsonl"
B3_FILE = ROOT / "step3_output_1000q" / "role_overlap_v2_step1.csv"
SWEEP_FILE = ROOT.parent / "results" / "_unused"  # set below
JUDGE_FILE = ROOT / "step4_judge_gptoss_1000q.json"
PROD_SWEEP_FILE = ROOT.parent / "check" / "production_sweep_output" / "hotpotqa_per_q_v2.jsonl"

BUDGETS = [0.10, 0.20, 0.30, 0.50]


def load_multi_sample_aggregates() -> dict:
    """Per-qid aggregates from multi_sample_dump_1000q.jsonl."""
    by_qid = defaultdict(list)
    with DUMP_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            by_qid[r["qid"]].append(r)
    out = {}
    for qid, recs in by_qid.items():
        preds = [(r.get("pred") or "").strip().lower() for r in recs]
        out[qid] = {
            "qid": qid,
            "stratum": recs[0]["stratum"],
            "question": recs[0]["question"],
            "em_rate": round(mean(r["em"] for r in recs), 3),
            "n_uniq": len(set(preds)),
            "n_samples": len(recs),
        }
    return out


def load_v2_b3_rates() -> dict:
    """Per-qid v2_b3_rate from step3 output."""
    out = {}
    with B3_FILE.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            qid = r["qid"]
            out[qid] = {
                "v2_b3_rate": float(r.get("v2_b3_rate", 0)),
                "schema": r.get("schema", ""),
            }
    return out


def load_production_sweep() -> dict:
    """Per-qid production sweep data: nocot_em, rag_em, neg_max_ent."""
    out = {}
    with PROD_SWEEP_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            qid = r["qid"]
            nocot_first_ent = r.get("nocot_first_ent") or {}
            out[qid] = {
                "nocot_em": int(r.get("nocot_em", 0)),
                "nocot_f1": float(r.get("nocot_f1", 0.0)),
                "rag_em": int(r.get("rag_em", 0)),
                "neg_max_ent": float(nocot_first_ent.get("neg_max_ent", 0.0)),
                "first_ent_min_lp": float(nocot_first_ent.get("min_lp", 0.0)),
                "has_first_ent": nocot_first_ent != {},
            }
    return out


def load_judge_labels() -> dict:
    """Per-qid LLM judge label."""
    if not JUDGE_FILE.exists():
        return {}
    out = {}
    data = json.load(JUDGE_FILE.open())
    for r in data:
        if r.get("label") in {"HI_correct", "B1", "B2", "B3", "Other"}:
            out[r["qid"]] = {
                "label": r["label"],
                "judge_reason": r.get("reason", ""),
            }
    return out


def merge_per_qid(ms, b3, sweep, judge) -> list:
    """Inner join on qid with all 4 sources."""
    rows = []
    for qid, m in ms.items():
        if qid not in sweep:
            continue
        if qid not in judge:
            continue
        row = dict(m)
        row.update(b3.get(qid, {"v2_b3_rate": 0.0, "schema": ""}))
        row.update(sweep[qid])
        row.update(judge[qid])
        rows.append(row)
    return rows


def gate_em_neg_max_ent(rows: list, budget_frac: float) -> tuple[float, int]:
    """Sort by neg_max_ent ascending (most uncertain first); retrieve top-k%.
    Returns (EM, n_triggered)."""
    n = len(rows)
    k = int(round(budget_frac * n))
    sorted_rows = sorted(rows, key=lambda r: r["neg_max_ent"])
    triggered_qids = set(r["qid"] for r in sorted_rows[:k])
    em_total = 0
    for r in rows:
        if r["qid"] in triggered_qids:
            em_total += r["rag_em"]
        else:
            em_total += r["nocot_em"]
    return em_total / n, len(triggered_qids)


def gate_em_subtype(rows: list, retrieve_labels: set) -> tuple[float, int]:
    """Retrieve iff label in retrieve_labels. Returns (EM, n_triggered)."""
    em_total = 0
    n_trig = 0
    for r in rows:
        if r["label"] in retrieve_labels:
            em_total += r["rag_em"]
            n_trig += 1
        else:
            em_total += r["nocot_em"]
    return em_total / len(rows), n_trig


def gate_em_subtype_x_threshold(rows: list, retrieve_labels: set,
                                budget_frac: float) -> tuple[float, int]:
    """Among retrieve-eligible (label in retrieve_labels), retrieve top-k% globally
    by neg_max_ent ascending. n_eligible = sum(label in retrieve_labels).
    Eligibility caps total budget."""
    n = len(rows)
    k = int(round(budget_frac * n))
    eligible = [r for r in rows if r["label"] in retrieve_labels]
    eligible_sorted = sorted(eligible, key=lambda r: r["neg_max_ent"])
    take_n = min(k, len(eligible_sorted))
    triggered_qids = set(r["qid"] for r in eligible_sorted[:take_n])
    em_total = 0
    for r in rows:
        if r["qid"] in triggered_qids:
            em_total += r["rag_em"]
        else:
            em_total += r["nocot_em"]
    return em_total / n, len(triggered_qids)


def gate_em_oracle(rows: list, budget_frac: float) -> tuple[float, int]:
    """Oracle: retrieve only the questions where retrieval helps (rag_em > nocot_em),
    capped at budget. Sort helping questions arbitrarily (we just take all that fit)."""
    n = len(rows)
    k = int(round(budget_frac * n))
    helps = [r for r in rows if r["rag_em"] > r["nocot_em"]]
    helps_sorted = sorted(helps, key=lambda r: r["neg_max_ent"])  # tie-break by uncertainty
    triggered = set(r["qid"] for r in helps_sorted[:k])
    em_total = 0
    for r in rows:
        if r["qid"] in triggered:
            em_total += r["rag_em"]
        else:
            em_total += r["nocot_em"]
    return em_total / n, len(triggered)


def main():
    OUT.mkdir(exist_ok=True)
    print("Loading data sources...", flush=True)
    ms = load_multi_sample_aggregates()
    b3 = load_v2_b3_rates()
    sweep = load_production_sweep()
    judge = load_judge_labels()

    print(f"  multi_sample qids: {len(ms)}")
    print(f"  v2_b3 entries:     {len(b3)}")
    print(f"  prod sweep qids:   {len(sweep)}")
    print(f"  judge labels:      {len(judge)}")

    if not judge:
        print("\n*** Judge file not yet present — skipping merge. Re-run after judge done. ***")
        return

    rows = merge_per_qid(ms, b3, sweep, judge)
    print(f"  joined rows:       {len(rows)}")

    label_dist = {}
    for r in rows:
        label_dist[r["label"]] = label_dist.get(r["label"], 0) + 1
    print(f"  label distribution: {label_dist}")

    # Write per-qid CSV
    csv_path = OUT / "subtype_labels_1000q.csv"
    fields = ["qid", "stratum", "label", "em_rate", "n_uniq", "v2_b3_rate", "schema",
              "nocot_em", "rag_em", "neg_max_ent", "first_ent_min_lp",
              "judge_reason", "question"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"\nWrote {csv_path}")

    # Strategy comparison
    n = len(rows)
    nr_em = sum(r["nocot_em"] for r in rows) / n
    rag_em = sum(r["rag_em"] for r in rows) / n
    print(f"\nBaselines: always-NR={nr_em:.4f}, always-RAG={rag_em:.4f}")

    # Subtype-aware (binary): retrieve = {B1, B2, Other}, skip = {HI_correct, B3}
    retrieve_set = {"B1", "B2", "Other"}
    d_em, d_trig = gate_em_subtype(rows, retrieve_set)
    print(f"D (subtype-aware binary): EM={d_em:.4f}, trig={d_trig} ({d_trig/n:.1%})")

    # Per-budget rows
    table_rows = []
    for bf in BUDGETS:
        c_em, c_trig = gate_em_neg_max_ent(rows, bf)
        e_em, e_trig = gate_em_subtype_x_threshold(rows, retrieve_set, bf)
        oracle_em, _ = gate_em_oracle(rows, bf)
        table_rows.append({
            "budget_frac": bf,
            "budget_n": int(round(bf * n)),
            "C_neg_max_ent_em": round(c_em, 4),
            "C_trig": c_trig,
            "E_subtype_x_neg_em": round(e_em, 4),
            "E_trig": e_trig,
            "Oracle_em": round(oracle_em, 4),
        })

    # Markdown table
    md = ["# Step A — gate strategy comparison (HotpotQA 1000Q)", "",
          f"**Joined sample size**: n={n}", "",
          f"**Label distribution**: {label_dist}", "",
          f"**Baselines**: always-NR (no-CoT direct answer)={nr_em:.4f}, always-RAG={rag_em:.4f}", "",
          f"**Subtype-aware binary (D)**: retrieve iff label in {{B1, B2, Other}};",
          f"  EM={d_em:.4f} at trig%={d_trig/n:.1%} ({d_trig}/{n})", "",
          "## Budget × strategy table", "",
          "| budget | n_trig | C neg_max_ent | E subtype × neg_max_ent | Oracle |",
          "|---|---|---|---|---|"]
    for tr in table_rows:
        md.append(
            f"| {tr['budget_frac']:.0%} | {tr['budget_n']} | "
            f"{tr['C_neg_max_ent_em']:.4f} ({tr['C_trig']} trig) | "
            f"{tr['E_subtype_x_neg_em']:.4f} ({tr['E_trig']} trig) | "
            f"{tr['Oracle_em']:.4f} |"
        )
    md += ["",
           "## Strategy interpretation",
           "",
           "- **A always-NR**: " + f"{nr_em:.4f} (no retrieval, lower bound)",
           "- **B always-RAG**: " + f"{rag_em:.4f} (always retrieve, upper bound, expensive)",
           "- **C neg_max_ent**: single-token threshold gate (current best); ranks all 1000Q by neg_max_ent ascending and retrieves top k%",
           "- **D subtype-aware binary**: rule on judge label only; ignores neg_max_ent",
           "- **E subtype × neg_max_ent**: retrieves top k% by neg_max_ent BUT skips HI_correct/B3 entirely",
           "- **Oracle**: retrieves only when it actually helps (rag_em > nocot_em); paper-quality upper bound at given budget",
           ""]

    md_path = OUT / "gate_comparison.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_path}")

    csv_path = OUT / "gate_comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        for r in table_rows: w.writerow(r)
    print(f"Wrote {csv_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        budgets_pct = [bf * 100 for bf in BUDGETS]

        # A always-NR (horizontal line at budget=0)
        ax.axhline(nr_em, color="gray", linestyle=":", label=f"A always-NR ({nr_em:.3f})")
        # B always-RAG
        ax.axhline(rag_em, color="black", linestyle="--", label=f"B always-RAG ({rag_em:.3f})")
        # C
        ax.plot(budgets_pct, [tr["C_neg_max_ent_em"] for tr in table_rows],
                "o-", color="steelblue", label="C neg_max_ent gate")
        # E
        ax.plot(budgets_pct, [tr["E_subtype_x_neg_em"] for tr in table_rows],
                "s-", color="darkorange", label="E subtype × neg_max_ent")
        # D as a single point at its own budget
        d_budget_pct = d_trig / n * 100
        ax.plot([d_budget_pct], [d_em], "^", color="green", markersize=10,
                label=f"D subtype-aware binary @ {d_budget_pct:.0f}%")
        # Oracle
        ax.plot(budgets_pct, [tr["Oracle_em"] for tr in table_rows],
                "*-", color="red", alpha=0.6, label="Oracle (gate has truth)")

        ax.set_xlabel("Retrieval budget (% of 1000 questions)")
        ax.set_ylabel("EM")
        ax.set_title("HotpotQA gate strategy comparison (n=1000)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        out_png = OUT / "budget_em_curve.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=140)
        print(f"Wrote {out_png}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
