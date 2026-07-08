"""Apply a fixed-threshold grid uniformly to all 4 datasets and report:
- per-dataset trigger%, TPR, FPR at each threshold
- spread of trigger% across datasets (lower = more dataset-invariant)

Evidence document for: "should we use fixed-threshold or fixed-quantile?"
Produces check/calibration_output/cross_dataset_drift.md.
"""
import json
from pathlib import Path
from statistics import mean, pstdev

OUT_DIR = Path(__file__).resolve().parent / "calibration_output"
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]

# unified candidate thresholds (chosen to cover ~p10..p90 of pooled wrong distribution)
LP_GRID = [-0.3, -0.5, -0.7, -1.0, -1.3, -1.6, -1.9, -2.2]
SC_GRID = [6.5, 7.0, 7.3, 7.5, 7.8, 8.1, 8.5, 9.0]


def load_records(ds):
    rows = []
    with (OUT_DIR / f"{ds}_records.jsonl").open() as f:
        for ln in f:
            r = json.loads(ln)
            ag = r.get("first_ent_aggs") or {}
            if "min_lp" not in ag or "min_sc" not in ag:
                continue
            rows.append({"em": int(r["em"]), "min_lp": ag["min_lp"], "min_sc": ag["min_sc"]})
    return rows


def stats_at(rows, key, thr):
    n = len(rows)
    triggered = [r for r in rows if r[key] < thr]
    correct = [r for r in rows if r["em"] == 1]
    wrong = [r for r in rows if r["em"] == 0]
    tpr = (sum(1 for r in wrong if r[key] < thr) / len(wrong)) if wrong else 0.0
    fpr = (sum(1 for r in correct if r[key] < thr) / len(correct)) if correct else 0.0
    trig = len(triggered) / n if n else 0.0
    return trig, tpr, fpr


def main():
    data = {ds: load_records(ds) for ds in DATASETS}

    out = ["# Cross-dataset drift: fixed unified threshold vs per-dataset quantile",
           "",
           "If a single fixed threshold is used across datasets, **a metric whose distribution is dataset-invariant** will fire at similar rates everywhere; one whose distribution drifts will over-trigger on some and under-trigger on others.",
           "",
           "Lower `trigger% spread` (stdev across 4 datasets) ⇒ more dataset-invariant ⇒ a single threshold is more defensible.",
           "",
           "## min_lp under unified threshold",
           "",
           "| threshold | hotpot trig% | 2wiki trig% | musique trig% | popqa trig% | spread (pp) | mean TPR | mean FPR |",
           "|---|---|---|---|---|---|---|---|"]

    for thr in LP_GRID:
        trigs, tprs, fprs = [], [], []
        for ds in DATASETS:
            t, tp, fp = stats_at(data[ds], "min_lp", thr)
            trigs.append(t); tprs.append(tp); fprs.append(fp)
        spread = pstdev(trigs) * 100
        out.append(f"| {thr:.2f} | {trigs[0]*100:.0f}% | {trigs[1]*100:.0f}% | "
                   f"{trigs[2]*100:.0f}% | {trigs[3]*100:.0f}% | "
                   f"{spread:.1f} | {mean(tprs):.3f} | {mean(fprs):.3f} |")

    out += ["", "## min_sc under unified threshold", "",
            "| threshold | hotpot trig% | 2wiki trig% | musique trig% | popqa trig% | spread (pp) | mean TPR | mean FPR |",
            "|---|---|---|---|---|---|---|---|"]
    for thr in SC_GRID:
        trigs, tprs, fprs = [], [], []
        for ds in DATASETS:
            t, tp, fp = stats_at(data[ds], "min_sc", thr)
            trigs.append(t); tprs.append(tp); fprs.append(fp)
        spread = pstdev(trigs) * 100
        out.append(f"| {thr:.2f} | {trigs[0]*100:.0f}% | {trigs[1]*100:.0f}% | "
                   f"{trigs[2]*100:.0f}% | {trigs[3]*100:.0f}% | "
                   f"{spread:.1f} | {mean(tprs):.3f} | {mean(fprs):.3f} |")

    # summary stat: mean of spread across grid
    def mean_spread(key, grid):
        sps = []
        for thr in grid:
            trigs = [stats_at(data[ds], key, thr)[0] for ds in DATASETS]
            sps.append(pstdev(trigs) * 100)
        return mean(sps)

    lp_avg = mean_spread("min_lp", LP_GRID)
    sc_avg = mean_spread("min_sc", SC_GRID)

    out += ["",
            "## Summary",
            "",
            f"- Mean `trigger% spread` across grid — **min_lp**: {lp_avg:.1f} pp, **min_sc**: {sc_avg:.1f} pp",
            f"- Lower spread ⇒ {'min_sc' if sc_avg < lp_avg else 'min_lp'} is more dataset-invariant under a single fixed threshold.",
            "- Implication: when using a unified threshold across datasets, "
            f"{'min_sc preserves operating-point semantics; min_lp drifts and would benefit from per-dataset quantile calibration' if sc_avg < lp_avg else 'min_lp is more invariant'}.",
            ""]

    Path(OUT_DIR / "cross_dataset_drift.md").write_text("\n".join(out))
    print(f"wrote {OUT_DIR/'cross_dataset_drift.md'}")
    print(f"min_lp avg spread = {lp_avg:.1f} pp")
    print(f"min_sc avg spread = {sc_avg:.1f} pp")


if __name__ == "__main__":
    main()
