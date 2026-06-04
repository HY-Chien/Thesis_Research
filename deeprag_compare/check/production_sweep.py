"""Production gate sweep: apply unified threshold grid to existing 1000Q × 4-dataset
per_q.jsonl records (which already have nr_em / rag_em / first_ent_min_lp), and
compute the gated EM for each (dataset, threshold) cell.

Smoke test: --n 20 limits to first N questions per dataset.
Full run:    --n 1000 (or omit, default 1000).

Currently only supports `min_lp` — to add `min_sc`, the per_q.jsonl files need
an extra forward-pass column (separate script).
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("/home/S113065528/Research/deeprag_compare")
INPUT_DIR = ROOT / "check/exp1_output"
OUT_DIR = ROOT / "check/production_sweep_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]

# Unified threshold grid (chosen from cross-dataset drift analysis)
MIN_LP_GRID = [-1.5, -1.1, -0.8, -0.5]
MIN_LP_TARGET_TRIG = ["~30%", "~45%", "~60%", "~75%"]


def load_records(ds, n):
    fp = INPUT_DIR / f"{ds}_per_q.jsonl"
    rows = []
    for ln in fp.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        rows.append(r)
        if len(rows) >= n:
            break
    return rows


def evaluate_gate(records, metric_key, threshold):
    """Apply gate: if metric < threshold → use rag_em else nr_em.
    Skip records with metric=None (fall back to nr_em).
    Returns dict with em, retrieve_rate, n_total, n_skipped, save_count, hurt_count.
    """
    em_total = 0
    n_retrieve = 0
    save = 0   # nr_em=0 → rag_em=1 (gate fired correctly)
    hurt = 0   # nr_em=1 → rag_em=0 (gate over-triggered, RAG broke it)
    miss = 0   # nr_em=0, gate did not fire (false negative)
    fixed_already = 0   # nr_em=1 and gate fired — wasteful but harmless
    n_skipped_metric = 0
    n = len(records)

    for r in records:
        nr_em = r["nr_em"]; rag_em = r["rag_em"]; m = r.get(metric_key)
        if m is None:
            # No metric → cannot fire gate → fall back to nr_em (conservative)
            em_total += nr_em
            n_skipped_metric += 1
            if nr_em == 0:
                miss += 1
            continue
        fire = m < threshold
        if fire:
            n_retrieve += 1
            em_total += rag_em
            if nr_em == 0 and rag_em == 1: save += 1
            elif nr_em == 1 and rag_em == 0: hurt += 1
            elif nr_em == 1 and rag_em == 1: fixed_already += 1
        else:
            em_total += nr_em
            if nr_em == 0: miss += 1

    return {
        "em": em_total / n,
        "retrieve_rate": n_retrieve / n,
        "save": save,
        "hurt": hurt,
        "miss": miss,
        "wasted_high_conf_rag": fixed_already,
        "no_metric_fallback": n_skipped_metric,
        "n": n,
    }


def baselines(records):
    n = len(records)
    return {
        "always_nr": sum(r["nr_em"] for r in records) / n,
        "always_rag": sum(r["rag_em"] for r in records) / n,
        "oracle":   sum(max(r["nr_em"], r["rag_em"]) for r in records) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    all_data = {ds: load_records(ds, args.n) for ds in args.datasets}

    lines = ["# Production gate sweep — unified thresholds",
             "",
             f"- N per dataset: {args.n}",
             f"- Metric: `first_ent_min_lp` (no-CoT first-entity span min log-prob)",
             f"- Gate rule: `score < threshold` ⇒ retrieve (use RAG answer)",
             "",
             "## Baselines",
             "",
             "| Dataset | always-NR | always-RAG | Oracle |",
             "|---|---|---|---|"]
    for ds in args.datasets:
        b = baselines(all_data[ds])
        lines.append(f"| {ds} | {b['always_nr']:.3f} | {b['always_rag']:.3f} | {b['oracle']:.3f} |")

    lines += ["", "## Sweep results (gated EM)", ""]
    for ds in args.datasets:
        recs = all_data[ds]
        b = baselines(recs)
        lines.append(f"### {ds} (n={len(recs)})")
        lines.append("")
        lines.append("| threshold | trig% | gated EM | Δ vs always-NR | Δ vs always-RAG | save | hurt |")
        lines.append("|---|---|---|---|---|---|---|")
        for thr, tag in zip(MIN_LP_GRID, MIN_LP_TARGET_TRIG):
            r = evaluate_gate(recs, "first_ent_min_lp", thr)
            d_nr = r["em"] - b["always_nr"]
            d_rag = r["em"] - b["always_rag"]
            lines.append(f"| {thr:.2f} ({tag}) | {r['retrieve_rate']*100:.0f}% | {r['em']:.3f} | "
                         f"{d_nr:+.3f} | {d_rag:+.3f} | {r['save']} | {r['hurt']} |")
        lines.append("")

    # Cross-dataset summary at each threshold
    lines += ["## Cross-dataset summary (mean Δ vs always-NR)", "",
              "| threshold | trig% (mean) | trig% (spread pp) | mean ΔEM vs NR | datasets that improved |",
              "|---|---|---|---|---|"]
    for thr, tag in zip(MIN_LP_GRID, MIN_LP_TARGET_TRIG):
        trigs, deltas, improved = [], [], []
        for ds in args.datasets:
            recs = all_data[ds]
            b = baselines(recs)
            r = evaluate_gate(recs, "first_ent_min_lp", thr)
            trigs.append(r["retrieve_rate"])
            d = r["em"] - b["always_nr"]
            deltas.append(d)
            if d > 0: improved.append(ds)
        spread = pstdev(trigs) * 100 if len(trigs) > 1 else 0.0
        lines.append(f"| {thr:.2f} ({tag}) | {mean(trigs)*100:.0f}% | {spread:.1f} | "
                     f"{mean(deltas):+.3f} | {len(improved)}/{len(args.datasets)} ({', '.join(improved) or '-'}) |")

    out_md = OUT_DIR / f"sweep_n{args.n}.md"
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")
    # also a quick stdout summary
    print("\n=== quick view ===")
    for ds in args.datasets:
        recs = all_data[ds]
        b = baselines(recs)
        print(f"[{ds}] n={len(recs)} always_nr={b['always_nr']:.3f} always_rag={b['always_rag']:.3f} oracle={b['oracle']:.3f}")
        for thr, tag in zip(MIN_LP_GRID, MIN_LP_TARGET_TRIG):
            r = evaluate_gate(recs, "first_ent_min_lp", thr)
            print(f"   thr={thr:+.2f} ({tag:>5}): trig={r['retrieve_rate']*100:5.1f}% em={r['em']:.3f} "
                  f"(Δ_nr {r['em']-b['always_nr']:+.3f}, Δ_rag {r['em']-b['always_rag']:+.3f})")


if __name__ == "__main__":
    main()
