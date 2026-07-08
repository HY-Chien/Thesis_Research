"""Offline replay of the direct-only confidence-gated RAG baseline.

Joins, per question id:
  parametric side : results/{ds}_1000_direct_norag.jsonl  (no context, no decompose;
                    max_ent recorded)   -- from scripts/run_direct_norag_probe.py
  RAG side        : the existing direct+always-RAG honest wiki-21M traces
                    ({ds}_1000_single_k8.jsonl / popqa_1000_wiki_single_top3.jsonl)

Gate rule (mirrors the canonical per-step gate, seek/pipeline.py):
  max_ent < tau  -> keep the parametric answer (confident, skip retrieval)
  max_ent >= tau -> use the RAG answer

Because decoding is greedy and the two candidate answers are independent,
this replay is exactly equivalent to a live gated run, and the full tau curve
comes for free.  tau=0 => Dir-always (must reproduce the stored single-run EM,
asserted as a sanity check); tau=inf => Dir-never (pure parametric).

Also reports AUROC(max_ent -> parametric answer wrong) per dataset.

Usage:
    python scripts/replay_direct_gate.py [--results-dir results] \
        [--taus 0,0.1,...] [--out results/direct_gate_ablation.md]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seek import em, f1  # noqa: E402

DATASETS = {
    "hotpotqa": "hotpotqa_1000_wiki_single_k8.jsonl",
    "2wiki": "2wiki_1000_single_k8.jsonl",
    "musique": "musique_1000_single_k8.jsonl",
    "popqa": "popqa_1000_wiki_single_top3.jsonl",
}
# stored always-RAG EM (RESULTS_MASTER §3.1 per-path direct) for the tau=0 sanity check
EXPECTED_ALWAYS_EM = {"hotpotqa": 30.3, "2wiki": 18.0, "musique": 10.2, "popqa": 37.1}

DEFAULT_TAUS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0,
                float("inf")]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC of score predicting label=1 (rank-based, ties averaged)."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks for ties
    ranks = {}
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-indexed average
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[k] for k, (_, lab) in enumerate(pairs) if lab == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def replay(ds: str, results_dir: Path, taus: list[float]) -> dict:
    norag = {r["id"]: r for r in load_jsonl(results_dir / f"{ds}_1000_direct_norag.jsonl")}
    rag = {r["id"]: r for r in load_jsonl(results_dir / DATASETS[ds])}
    ids = [i for i in norag if i in rag]
    if len(ids) != len(norag) or len(ids) != len(rag):
        print(f"WARNING {ds}: join {len(ids)} of norag={len(norag)} rag={len(rag)}")

    rows = []
    for qid in ids:
        p, g = norag[qid], rag[qid]
        golds = [p["gold"]] + (p.get("gold_aliases") or [])
        rows.append({
            "id": qid,
            "ent": p["max_ent"],
            "para_pred": p["pred"],
            "rag_pred": g["pred"],
            "para_em": int(em(p["pred"], golds)),
            "para_f1": f1(p["pred"], golds),
            "rag_em": int(em(g["pred"], golds)),
            "rag_f1": f1(g["pred"], golds),
        })
    n = len(rows)

    # sanity: recomputed always-RAG EM must match the stored single-run EM
    always_em = 100 * sum(r["rag_em"] for r in rows) / n
    exp = EXPECTED_ALWAYS_EM[ds]
    assert abs(always_em - exp) < 0.15, \
        f"{ds}: recomputed always-RAG EM {always_em:.1f} != stored {exp} (join/norm bug?)"

    sweep = []
    for tau in taus:
        em_s = f1_s = retr = 0.0
        for r in rows:
            use_rag = r["ent"] >= tau
            retr += use_rag
            em_s += r["rag_em"] if use_rag else r["para_em"]
            f1_s += r["rag_f1"] if use_rag else r["para_f1"]
        sweep.append({"tau": tau, "em": 100 * em_s / n, "f1": 100 * f1_s / n,
                      "retr": 100 * retr / n})

    return {
        "n": n,
        "sweep": sweep,
        "never": next(s for s in sweep if math.isinf(s["tau"])),
        "always": next(s for s in sweep if s["tau"] == 0.0),
        "auroc": auroc([r["ent"] for r in rows], [1 - r["para_em"] for r in rows]),
        "rows": rows,
    }


def fmt_tau(t: float) -> str:
    return "inf" if math.isinf(t) else f"{t:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    ap.add_argument("--taus", default=None, help="comma-separated tau grid (inf allowed)")
    ap.add_argument("--matched-tau", type=float, default=0.3)
    args = ap.parse_args()

    taus = DEFAULT_TAUS if args.taus is None else [float(t) for t in args.taus.split(",")]
    if args.matched_tau not in taus:
        taus = sorted(taus + [args.matched_tau])
    results_dir = Path(args.results_dir)

    all_res = {}
    for ds in DATASETS:
        all_res[ds] = replay(ds, results_dir, taus)

    print("\n=== Direct-only confidence-gated RAG (honest wiki-21M, 1000Q) ===\n")
    hdr = f"{'dataset':10s} {'never(EM/F1)':>14s} {'gate tau=' + str(args.matched_tau):>16s} " \
          f"{'best tau (EM/F1 @tau, retr%)':>30s} {'always(EM/F1)':>14s} {'AUROC':>6s}"
    print(hdr)
    for ds, res in all_res.items():
        finite = [s for s in res["sweep"] if not math.isinf(s["tau"]) and s["tau"] > 0]
        best = max(res["sweep"], key=lambda s: s["em"])
        matched = next(s for s in res["sweep"] if s["tau"] == args.matched_tau)
        print(f"{ds:10s} {res['never']['em']:6.1f}/{res['never']['f1']:4.1f} "
              f"  {matched['em']:6.1f}/{matched['f1']:4.1f} ({matched['retr']:.0f}%) "
              f"  {best['em']:6.1f}/{best['f1']:4.1f} @{fmt_tau(best['tau']):>4s} ({best['retr']:.0f}%) "
              f"  {res['always']['em']:6.1f}/{res['always']['f1']:4.1f} "
              f"  {res['auroc']:.3f}")

    print("\n=== Full tau sweep (EM / F1 / retrieval%) ===")
    for ds, res in all_res.items():
        print(f"\n[{ds}] n={res['n']}")
        for s in res["sweep"]:
            print(f"  tau={fmt_tau(s['tau']):>4s}  EM={s['em']:5.1f}  F1={s['f1']:5.1f}  retr={s['retr']:5.1f}%")

    dump = results_dir / "direct_gate_replay_dump.json"
    with open(dump, "w") as fh:
        json.dump({ds: {k: v for k, v in res.items() if k != "rows"}
                   for ds, res in all_res.items()}, fh, indent=1, default=str)
    print(f"\nsweep dump -> {dump}")


if __name__ == "__main__":
    main()
