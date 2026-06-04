"""Per-question diff: where does subtype-aware (D/E) differ from neg_max_ent (C)?

Goal: identify which subset of questions the subtype label catches that
neg_max_ent misses, and vice versa. This is the analytical contribution
even if aggregate EM is tied.

Outputs:
  step4_output/per_q_strategy_diff.csv  per-qid: gate decisions + outcomes for each strategy
  step4_output/strategy_diff_summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path("/home/S113065528/Research/deeprag_compare/check")
OUT = ROOT / "step4_output"
LABELS_CSV = OUT / "subtype_labels_1000q.csv"

BUDGET_FOR_DIFF = 0.30  # focus on 30% budget — most relevant


def load_labels() -> list:
    rows = []
    with LABELS_CSV.open() as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            r["em_rate"] = float(r["em_rate"]) if r["em_rate"] else 0.0
            r["v2_b3_rate"] = float(r["v2_b3_rate"]) if r["v2_b3_rate"] else 0.0
            r["nocot_em"] = int(r["nocot_em"])
            r["rag_em"] = int(r["rag_em"])
            r["neg_max_ent"] = float(r["neg_max_ent"]) if r["neg_max_ent"] else 0.0
            r["n_uniq"] = int(r["n_uniq"]) if r["n_uniq"] else 1
            rows.append(r)
    return rows


def main():
    rows = load_labels()
    n = len(rows)

    # Decisions per strategy at budget=30%
    k = int(round(BUDGET_FOR_DIFF * n))
    sorted_by_score = sorted(rows, key=lambda r: r["neg_max_ent"])
    c_trig = set(r["qid"] for r in sorted_by_score[:k])

    retrieve_set = {"B1", "B2", "Other"}
    eligible = [r for r in rows if r["label"] in retrieve_set]
    eligible_sorted = sorted(eligible, key=lambda r: r["neg_max_ent"])
    e_trig = set(r["qid"] for r in eligible_sorted[:k])

    d_trig = set(r["qid"] for r in rows if r["label"] in retrieve_set)

    # Per-q outcomes
    enriched = []
    for r in rows:
        c_decision = "RAG" if r["qid"] in c_trig else "NR"
        e_decision = "RAG" if r["qid"] in e_trig else "NR"
        d_decision = "RAG" if r["qid"] in d_trig else "NR"
        c_em = r["rag_em"] if c_decision == "RAG" else r["nocot_em"]
        e_em = r["rag_em"] if e_decision == "RAG" else r["nocot_em"]
        d_em = r["rag_em"] if d_decision == "RAG" else r["nocot_em"]
        enriched.append({**r,
                         "C_decision": c_decision, "C_em": c_em,
                         "E_decision": e_decision, "E_em": e_em,
                         "D_decision": d_decision, "D_em": d_em,
                         "would_help": int(r["rag_em"] > r["nocot_em"]),
                         "would_hurt": int(r["rag_em"] < r["nocot_em"])})

    out_csv = OUT / "per_q_strategy_diff.csv"
    out_fields = ["qid", "stratum", "label", "em_rate", "n_uniq", "v2_b3_rate",
                  "neg_max_ent", "nocot_em", "rag_em",
                  "would_help", "would_hurt",
                  "C_decision", "C_em",
                  "E_decision", "E_em",
                  "D_decision", "D_em",
                  "question"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in enriched:
            w.writerow({k: r.get(k, "") for k in out_fields})
    print(f"Wrote {out_csv}")

    # Summary stats
    md = ["# Strategy diff @ budget=30% (HotpotQA n=1000)", "",
          f"**Setup**: budget=30%, top-300 by neg_max_ent ascending vs subtype-aware variants.", ""]

    # Aggregate EM per strategy
    md += ["## Per-strategy aggregate EM at budget=30%", ""]
    md += ["| Strategy | n_trig | EM | save | hurt |"]
    md += ["|---|---|---|---|---|"]
    for strategy in ["C", "E"]:
        em_total = sum(r[f"{strategy}_em"] for r in enriched)
        n_trig = sum(1 for r in enriched if r[f"{strategy}_decision"] == "RAG")
        save = sum(1 for r in enriched
                   if r[f"{strategy}_decision"] == "RAG" and r["rag_em"] > r["nocot_em"])
        hurt = sum(1 for r in enriched
                   if r[f"{strategy}_decision"] == "RAG" and r["rag_em"] < r["nocot_em"])
        md.append(f"| {strategy} | {n_trig} | {em_total/n:.4f} | {save} | {hurt} |")

    # D outside of budget (because it's fixed)
    em_d = sum(r["D_em"] for r in enriched)
    n_trig_d = sum(1 for r in enriched if r["D_decision"] == "RAG")
    save_d = sum(1 for r in enriched
                 if r["D_decision"] == "RAG" and r["rag_em"] > r["nocot_em"])
    hurt_d = sum(1 for r in enriched
                 if r["D_decision"] == "RAG" and r["rag_em"] < r["nocot_em"])
    md.append(f"| D (no budget; trig={n_trig_d/n:.0%}) | {n_trig_d} | {em_d/n:.4f} | {save_d} | {hurt_d} |")

    # C vs E disagreement
    only_c = [r for r in enriched if r["C_decision"] == "RAG" and r["E_decision"] != "RAG"]
    only_e = [r for r in enriched if r["E_decision"] == "RAG" and r["C_decision"] != "RAG"]
    md += ["",
           "## C vs E disagreement (budget=30%)", "",
           f"- C retrieves but E does not: **{len(only_c)} questions**",
           f"- E retrieves but C does not: **{len(only_e)} questions**", "",
           "### Where C retrieves but E doesn't (E saves the budget)"]
    if only_c:
        outcomes = Counter()
        for r in only_c:
            if r["rag_em"] > r["nocot_em"]:
                outcomes["RAG would help (E loses)"] += 1
            elif r["rag_em"] < r["nocot_em"]:
                outcomes["RAG would hurt (E saves)"] += 1
            else:
                outcomes["RAG no diff"] += 1
        for k_, v in outcomes.items():
            md.append(f"- {k_}: {v}")
        md.append("")
        md.append("Labels of these (where E saved budget):")
        label_cnt = Counter(r["label"] for r in only_c)
        for k_, v in label_cnt.items():
            md.append(f"  - {k_}: {v}")

    md.append("")
    md.append("### Where E retrieves but C doesn't (E spends budget on)")
    if only_e:
        outcomes = Counter()
        for r in only_e:
            if r["rag_em"] > r["nocot_em"]:
                outcomes["RAG helps (E gain)"] += 1
            elif r["rag_em"] < r["nocot_em"]:
                outcomes["RAG hurts (E loses)"] += 1
            else:
                outcomes["RAG no diff"] += 1
        for k_, v in outcomes.items():
            md.append(f"- {k_}: {v}")
        md.append("")
        md.append("Labels of these:")
        label_cnt = Counter(r["label"] for r in only_e)
        for k_, v in label_cnt.items():
            md.append(f"  - {k_}: {v}")

    # Per-label gain breakdown
    md += ["", "## Per-label retrieval helpfulness", "",
           "For each subtype, what fraction of questions had rag_em > nocot_em?",
           "(higher = subtype is good RAG candidate)", "",
           "| label | n | helps (rag>nr) | hurts (rag<nr) | tied | help_rate |",
           "|---|---|---|---|---|---|"]
    by_label = defaultdict(list)
    for r in rows: by_label[r["label"]].append(r)
    for lab in ["HI_correct", "B1", "B2", "B3", "Other"]:
        recs = by_label.get(lab, [])
        if not recs: continue
        helps = sum(1 for r in recs if r["rag_em"] > r["nocot_em"])
        hurts = sum(1 for r in recs if r["rag_em"] < r["nocot_em"])
        tied = len(recs) - helps - hurts
        md.append(f"| {lab} | {len(recs)} | {helps} | {hurts} | {tied} | {helps/len(recs):.2%} |")

    # B3 deep-dive
    b3_rows = [r for r in rows if r["label"] == "B3"]
    if b3_rows:
        md += ["", f"## B3 case deep-dive ({len(b3_rows)} cases)", ""]
        md += ["| qid | stratum | em_rate | n_uniq | v2_b3 | nocot_em | rag_em | neg_max_ent |",
               "|---|---|---|---|---|---|---|---|"]
        for r in b3_rows:
            md.append(f"| {r['qid'][:12]} | {r['stratum']} | {r['em_rate']} | "
                      f"{r['n_uniq']} | {r['v2_b3_rate']:.2f} | {r['nocot_em']} | "
                      f"{r['rag_em']} | {r['neg_max_ent']:.3f} |")
        md.append("")
        md.append("**Interpretation**: For B3 cases, does retrieval help? If rag_em mostly = 0, "
                  "this validates the 'B3 needs prompt restructure not RAG' claim.")

    out_md = OUT / "strategy_diff_summary.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
