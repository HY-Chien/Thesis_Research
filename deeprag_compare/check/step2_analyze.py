"""Step 2 analyzer — turn perturbation_dump.jsonl into sensitivity table.

Reads:  step2_output/perturbation_dump.jsonl (one record per qid x mode x sample)
Writes: step2_output/sensitivity_summary.csv  (per qid x mode aggregates)
        step2_output/sensitivity_table.md     (interpretive table per qid)

Per-mode aggregates:
  sc_mean, sc_std, ppl_mean, em_rate, n_uniq,
  min_margin_full_mean, fe_mean_entropy_mean (None-safe), fe_min_lp_mean

Sensitivity (perturbed mode vs `original`):
  d_sc            = sc(mode) - sc(original)            (negative = lost confidence)
  d_min_margin    = min_margin(mode) - min_margin(original)   (more negative = bigger dip introduced)
  d_fe_entropy    = fe_mean_entropy(mode) - fe_mean_entropy(original)
  pred_change_rate = fraction of mode samples whose pred differs from original's modal pred

Subtype hint:
  A1_score = (1 - pred_change_rate(mask)) + (1 - pred_change_rate(replace))
            (high => model ignores input perturbations, A1-like)
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev

OUT_DIR = Path(__file__).parent / "step2_output"
DUMP = OUT_DIR / "perturbation_dump.jsonl"
SUMMARY = OUT_DIR / "sensitivity_summary.csv"
TABLE = OUT_DIR / "sensitivity_table.md"


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def safe_std(xs):
    xs = [x for x in xs if x is not None]
    return pstdev(xs) if len(xs) > 1 else None


def fmt(v, d=3):
    return f"{v:.{d}f}" if isinstance(v, (int, float)) else "n/a"


def fmt_signed(v, d=3):
    return f"{v:+.{d}f}" if isinstance(v, (int, float)) else "n/a"


def main():
    by_qid_mode = defaultdict(list)
    qid_meta = {}
    for line in DUMP.open():
        r = json.loads(line)
        qid = r["qid"]; mode = r["mode"]
        by_qid_mode[(qid, mode)].append(r)
        qid_meta.setdefault(qid, {"stratum": r["stratum"]})

    qids = sorted({q for q, _ in by_qid_mode})

    rows = []
    for qid in qids:
        for mode in ("original", "mask", "replace", "distractor"):
            recs = by_qid_mode.get((qid, mode), [])
            if not recs: continue
            sc = [r.get("sc") for r in recs]
            ppl = [r.get("ppl") for r in recs]
            mmf = [r.get("min_margin_full") for r in recs]
            fme = [r.get("fe_mean_entropy") for r in recs]
            fmi = [r.get("fe_min_lp") for r in recs]
            preds = [(r.get("pred") or "").strip().lower() for r in recs]
            ems = [r.get("em", 0) for r in recs]
            rows.append({
                "qid": qid, "stratum": qid_meta[qid]["stratum"], "mode": mode,
                "n": len(recs),
                "sc_mean": safe_mean(sc), "sc_std": safe_std(sc),
                "ppl_mean": safe_mean(ppl),
                "em_rate": mean(ems) if ems else None,
                "n_uniq": len(set(preds)),
                "modal_pred": Counter(preds).most_common(1)[0][0] if preds else "",
                "modal_pred_share": Counter(preds).most_common(1)[0][1] / len(preds) if preds else None,
                "min_margin_full_mean": safe_mean(mmf),
                "fe_mean_entropy_mean": safe_mean(fme),
                "fe_min_lp_mean": safe_mean(fmi),
                "preds": preds,
            })

    fieldnames = [k for k in rows[0].keys() if k != "preds"]
    with SUMMARY.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    print(f"wrote {SUMMARY}")

    by_qid = defaultdict(dict)
    for r in rows:
        by_qid[r["qid"]][r["mode"]] = r

    md = ["# Step 2 — Perturbation Sensitivity",
          "",
          "Per-question sensitivity to input perturbations. "
          "**Hypothesis**: HI_correct should show big shifts when the key entity is "
          "masked/replaced (model uses input). A1 confident hallucinations should "
          "show *no* shift (model committed internally regardless of input). "
          "A2 hallucinations may shift but already start uncertain.",
          "",
          "Columns:",
          "- `Δsc(m)`: SC change relative to original under condition m (negative = less confident)",
          "- `pcr(m)`: pred-change rate vs original modal pred (1.0 = every sample diverges)",
          "- `Δmm(m)`: change in min_margin_full",
          "- A1_score = (1 - pcr(mask)) + (1 - pcr(replace)). Higher (max=2.0) ⇒ more A1-like.",
          "",
          "| qid | stratum | n_uniq(orig) | em(orig) | modal(orig) | "
          "Δsc(mask)/pcr | Δsc(replace)/pcr | Δsc(distract)/pcr | Δmm(mask) | A1_score |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    a1_table = []
    for qid in qids:
        modes = by_qid[qid]
        if "original" not in modes: continue
        orig = modes["original"]
        modal_orig = orig["modal_pred"]

        def cell(m):
            if m not in modes: return ("n/a", "n/a", "n/a")
            mr = modes[m]
            d_sc = (mr["sc_mean"] - orig["sc_mean"]) if mr["sc_mean"] is not None and orig["sc_mean"] is not None else None
            d_mm = (mr["min_margin_full_mean"] - orig["min_margin_full_mean"]) if mr["min_margin_full_mean"] is not None and orig["min_margin_full_mean"] is not None else None
            pcr = (sum(1 for p in mr["preds"] if p != modal_orig) / len(mr["preds"])) if mr["preds"] else None
            return d_sc, pcr, d_mm

        d_sc_mask, pcr_mask, d_mm_mask = cell("mask")
        d_sc_rep,  pcr_rep,  _         = cell("replace")
        d_sc_dis,  pcr_dis,  _         = cell("distractor")

        a1 = None
        if pcr_mask is not None and pcr_rep is not None:
            a1 = (1 - pcr_mask) + (1 - pcr_rep)
        a1_table.append((qid, orig["stratum"], a1))

        md.append(
            f"| {qid[:10]} | {orig['stratum']} | {orig['n_uniq']} | "
            f"{fmt(orig['em_rate'], 2)} | `{modal_orig[:25]}` | "
            f"{fmt_signed(d_sc_mask)} / {fmt(pcr_mask, 2)} | "
            f"{fmt_signed(d_sc_rep)} / {fmt(pcr_rep, 2)} | "
            f"{fmt_signed(d_sc_dis)} / {fmt(pcr_dis, 2)} | "
            f"{fmt_signed(d_mm_mask, 4)} | "
            f"{fmt(a1, 2)} |"
        )

    md += ["", "## A1_score ranking (higher = more input-insensitive = more A1-like)", ""]
    md.append("| qid | stratum | A1_score |")
    md.append("|---|---|---|")
    a1_table.sort(key=lambda t: -(t[2] if t[2] is not None else -1))
    for qid, stratum, a1 in a1_table:
        md.append(f"| {qid[:10]} | {stratum} | {fmt(a1, 2)} |")

    md += ["", "## Per-qid raw modal predictions (all 4 conditions)", ""]
    for qid in qids:
        modes = by_qid[qid]
        if "original" not in modes: continue
        md.append(f"### {qid[:10]} ({modes['original']['stratum']})")
        for m in ("original", "mask", "replace", "distractor"):
            if m not in modes: continue
            mr = modes[m]
            counts = Counter(mr["preds"]).most_common()
            preds_str = "  |  ".join(f"{p!r}×{c}" for p, c in counts[:5])
            md.append(f"- **{m:11s}** sc={fmt(mr['sc_mean'])} "
                      f"em={fmt(mr['em_rate'], 2)} n_uniq={mr['n_uniq']}: {preds_str}")
        md.append("")

    TABLE.write_text("\n".join(md) + "\n")
    print(f"wrote {TABLE}")

    print("\n=== A1 ranking ===")
    for qid, stratum, a1 in a1_table:
        print(f"  {qid[:10]} ({stratum:11s})  A1_score={fmt(a1, 2)}")


if __name__ == "__main__":
    main()
