"""Production gate sweep on no-CoT features (per_q_v2.jsonl).

Sweeps 4 unified thresholds × {min_lp, min_sc, neg_max_ent} × 4 datasets.
Reads check/production_sweep_output/{ds}_per_q_v2.jsonl produced by
add_nocot_features.py.

Each record has:
  - nr_em, rag_em       (CoT path baseline EMs, from exp1_output)
  - nocot_em            (no-CoT path EM — separate baseline)
  - nocot_first_ent     (dict with min_lp/min_sc/neg_max_ent/...)

Gate semantics:
  use_rag = score < threshold
  if use_rag: gated_em = rag_em else gated_em = nr_em

(We compare against `nr_em`, not `nocot_em`, because the gate fires on top of
the existing NR pipeline. nocot_em is reported separately as a sanity check.)
"""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("/home/S113065528/Research/deeprag_compare")
IN_DIR = ROOT / "check/production_sweep_output"
OUT_DIR = IN_DIR
DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]

# Unified threshold grid (from cross_dataset_drift analysis, calibrated to ~30/45/60/75% trigger)
GRIDS = {
    "min_lp":      [-1.5, -1.1, -0.8, -0.5],
    "min_sc":      [7.0, 7.4, 7.8, 8.4],
    "neg_max_ent": [-1.20, -0.80, -0.50, -0.25],   # max_ent low ⇒ confident; neg_max_ent high ⇒ confident
}
TARGET_TAGS = ["~30%", "~45%", "~60%", "~75%"]


def load_records(ds, n_max):
    fp = IN_DIR / f"{ds}_per_q_v2.jsonl"
    rows = []
    for ln in fp.read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        rows.append(r)
        if len(rows) >= n_max:
            break
    return rows


def evaluate(records, metric_key, threshold):
    em_total = 0; n_retrieve = 0; save = 0; hurt = 0
    no_metric = 0
    n = len(records)
    for r in records:
        nr = r["nr_em"]; rg = r["rag_em"]
        ag = r.get("nocot_first_ent")
        m = ag.get(metric_key) if ag else None
        if m is None:
            em_total += nr
            no_metric += 1
            continue
        fire = m < threshold
        if fire:
            n_retrieve += 1
            em_total += rg
            if nr == 0 and rg == 1: save += 1
            elif nr == 1 and rg == 0: hurt += 1
        else:
            em_total += nr
    return {
        "em": em_total / n,
        "trig": n_retrieve / n,
        "save": save, "hurt": hurt, "no_metric": no_metric, "n": n,
    }


def baselines(records):
    n = len(records)
    return {
        "always_nr": sum(r["nr_em"] for r in records) / n,
        "always_rag": sum(r["rag_em"] for r in records) / n,
        "oracle":   sum(max(r["nr_em"], r["rag_em"]) for r in records) / n,
        "nocot":    sum(r["nocot_em"] for r in records) / n,
    }


def auroc(y, s):
    pos = [a for b, a in zip(y, s) if b == 1]
    neg = [a for b, a in zip(y, s) if b == 0]
    if not pos or not neg: return float("nan")
    c = sum(1 for p in pos for n in neg if p > n)
    c += sum(0.5 for p in pos for n in neg if p == n)
    return c / (len(pos) * len(neg))


def discr_signals(records, metric_key):
    """AUROC: does score correlate with (1 - nr_em)?  Higher score = more confident
    ⇒ should correlate with nr_em=1 ⇒ AUROC(y=nr_em, s=score)."""
    y, s = [], []
    for r in records:
        ag = r.get("nocot_first_ent")
        m = ag.get(metric_key) if ag else None
        if m is None: continue
        y.append(r["nr_em"]); s.append(m)
    return auroc(y, s) if y else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", default="sweep_nocot.md")
    args = ap.parse_args()

    data = {}
    for ds in args.datasets:
        fp = IN_DIR / f"{ds}_per_q_v2.jsonl"
        if not fp.exists():
            print(f"[SKIP] {fp} not found")
            continue
        data[ds] = load_records(ds, args.n)

    L = ["# Production gate sweep — no-CoT first-entity span",
         "",
         f"- N per dataset: {args.n}",
         "- Gate fires when `score < threshold` ⇒ retrieve",
         "- Score = chosen aggregation over the **no-CoT** direct-answer first-entity span",
         "",
         "## Baselines",
         "",
         "| Dataset | always-NR (CoT) | always-RAG | Oracle | no-CoT EM (sanity) |",
         "|---|---|---|---|---|"]
    for ds in args.datasets:
        if ds not in data: continue
        b = baselines(data[ds])
        L.append(f"| {ds} | {b['always_nr']:.3f} | {b['always_rag']:.3f} | {b['oracle']:.3f} | {b['nocot']:.3f} |")

    # AUROC table — discrimination of nr_em (CoT) by no-CoT signal
    L += ["", "## AUROC — does no-CoT first-ent signal correlate with **nr_em (CoT)**?", "",
          "(higher = better discriminator for the gate's actual decision target)",
          "",
          "| Dataset | min_lp | min_sc | neg_max_ent |",
          "|---|---|---|---|"]
    for ds in args.datasets:
        if ds not in data: continue
        recs = data[ds]
        L.append(f"| {ds} | {discr_signals(recs,'min_lp'):.3f} | "
                 f"{discr_signals(recs,'min_sc'):.3f} | {discr_signals(recs,'neg_max_ent'):.3f} |")

    # Per-metric, per-dataset sweep
    for metric in ["min_lp", "min_sc", "neg_max_ent"]:
        L += ["", f"## Sweep: `{metric}`", ""]
        for ds in args.datasets:
            if ds not in data: continue
            recs = data[ds]; b = baselines(recs)
            L.append(f"### {ds} (n={len(recs)}, always_nr={b['always_nr']:.3f}, always_rag={b['always_rag']:.3f})")
            L.append("")
            L.append("| threshold | trig% | gated EM | Δ vs NR | Δ vs RAG | save | hurt |")
            L.append("|---|---|---|---|---|---|---|")
            for thr, tag in zip(GRIDS[metric], TARGET_TAGS):
                r = evaluate(recs, metric, thr)
                L.append(f"| {thr:.2f} ({tag}) | {r['trig']*100:.0f}% | {r['em']:.3f} | "
                         f"{r['em']-b['always_nr']:+.3f} | {r['em']-b['always_rag']:+.3f} | "
                         f"{r['save']} | {r['hurt']} |")
            L.append("")

        # Cross-dataset summary for this metric
        L += [f"### Cross-dataset summary — `{metric}`", "",
              "| threshold | mean trig% | trig% spread | mean ΔEM vs NR | improved |",
              "|---|---|---|---|---|"]
        for thr, tag in zip(GRIDS[metric], TARGET_TAGS):
            trigs, deltas, imp = [], [], []
            for ds in args.datasets:
                if ds not in data: continue
                b = baselines(data[ds])
                r = evaluate(data[ds], metric, thr)
                trigs.append(r["trig"])
                d = r["em"] - b["always_nr"]
                deltas.append(d)
                if d > 0: imp.append(ds)
            spread = pstdev(trigs) * 100 if len(trigs) > 1 else 0.0
            L.append(f"| {thr:.2f} ({tag}) | {mean(trigs)*100:.0f}% | {spread:.1f} pp | "
                     f"{mean(deltas):+.3f} | {len(imp)}/{len(trigs)} ({', '.join(imp) or '-'}) |")
        L.append("")

    out = OUT_DIR / args.out
    out.write_text("\n".join(L))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
