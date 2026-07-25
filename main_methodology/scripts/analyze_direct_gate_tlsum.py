"""Recalculate the direct-only retrieval gate with tail-leakage surprisal.

The no-context candidates and token statistics come from
``probe_direct_norag_tlsum.py``.  The always-retrieve direct candidates and
the decomposition-path comparison come from the final tl_sum + prompt-fix-A
trace joined by ``replay_verifier_tlsum_fixA.build``.

Gate rule
---------
Retrieve iff ``tl_sum >= tau``; otherwise keep the no-context answer.

The fixed comparison uses the deployed step-gate threshold 0.008461.  The
best-development threshold is found by exhaustively evaluating every unique
observed tl_sum value, plus the always- and never-retrieve endpoints.  The
deterministic selection order is: maximum EM, maximum F1, minimum retrieval
rate, then smallest threshold.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_confidence_battery import auroc, compute_signals  # noqa: E402
from replay_verifier_tlsum_fixA import build  # noqa: E402
from seek.verifier import exact_match, token_f1  # noqa: E402


DATASETS = ["hotpotqa", "2wiki", "musique", "popqa"]
LEGACY_RAG = {
    "hotpotqa": "hotpotqa_1000_wiki_single_k8.jsonl",
    "2wiki": "2wiki_1000_single_k8.jsonl",
    "musique": "musique_1000_single_k8.jsonl",
    "popqa": "popqa_1000_wiki_single_top3.jsonl",
}
MATCHED_TAU = 0.008461


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_rows(ds: str) -> tuple[list[dict], dict]:
    final_rows = build(ds)
    by_id = {str(row["id"]): row for row in final_rows}

    probes: list[dict] = []
    for shard in (0, 1):
        probes.extend(load_jsonl(
            ROOT / f"results/direct_norag_tlsum_{ds}_shard{shard}.jsonl"))
    probe_by_id = {str(row["id"]): row for row in probes}
    if len(final_rows) != 1000 or len(probes) != 1000 or len(probe_by_id) != 1000:
        raise RuntimeError(
            f"{ds}: expected 1000 unique rows; final={len(final_rows)}, "
            f"probe={len(probes)}, unique_probe={len(probe_by_id)}")
    if set(by_id) != set(probe_by_id):
        raise RuntimeError(f"{ds}: final/probe id sets differ")

    rows = []
    pred_mismatches = 0
    for rid, final in by_id.items():
        probe = probe_by_id[rid]
        pred_mismatches += not bool(probe.get("pred_match"))
        signals = compute_signals(probe)
        if signals is None:
            raise RuntimeError(f"{ds}/{rid}: empty token trace")
        golds = final["golds"]
        rows.append({
            "id": rid,
            "tl_sum": float(signals["tl_sum"]),
            "para_em": int(exact_match(probe["pred"], golds)),
            "para_f1": float(token_f1(probe["pred"], golds)),
            "rag_em": int(final["direct_em"]),
            "rag_f1": float(final["direct_f1"]),
            "decomp_em": int(final["decomp_em"]),
            "decomp_f1": float(final["decomp_f1"]),
        })

    legacy = {
        str(row["id"]): row
        for row in load_jsonl(ROOT / "results" / LEGACY_RAG[ds])
    }
    legacy_mismatches = sum(
        final["direct_pred"].strip()
        != (legacy.get(rid) or {}).get("pred", "").strip()
        for rid, final in by_id.items()
    )
    audit = {
        "probe_prediction_mismatches": int(pred_mismatches),
        "legacy_final_rag_prediction_mismatches": int(legacy_mismatches),
        "legacy_rag_file": LEGACY_RAG[ds],
    }
    return rows, audit


def evaluate(rows: list[dict], tau: float) -> dict:
    em_sum = f1_sum = retrievals = 0.0
    for row in rows:
        retrieve = row["tl_sum"] >= tau
        retrievals += retrieve
        em_sum += row["rag_em"] if retrieve else row["para_em"]
        f1_sum += row["rag_f1"] if retrieve else row["para_f1"]
    n = len(rows)
    return {
        "tau": tau,
        "em": 100.0 * em_sum / n,
        "f1": 100.0 * f1_sum / n,
        "retrieval_rate": 100.0 * retrievals / n,
    }


def best_development_threshold(rows: list[dict]) -> dict:
    thresholds = [0.0, *sorted({row["tl_sum"] for row in rows}), math.inf]
    sweep = [evaluate(rows, tau) for tau in thresholds]
    return max(
        sweep,
        key=lambda result: (
            result["em"],
            result["f1"],
            -result["retrieval_rate"],
            -result["tau"],
        ),
    )


def summarize(ds: str) -> dict:
    rows, audit = load_rows(ds)
    matched = evaluate(rows, MATCHED_TAU)
    never = evaluate(rows, math.inf)
    always = evaluate(rows, 0.0)
    best = best_development_threshold(rows)
    decomp = {
        "em": 100.0 * np.mean([row["decomp_em"] for row in rows]),
        "f1": 100.0 * np.mean([row["decomp_f1"] for row in rows]),
    }
    scores = np.asarray([row["tl_sum"] for row in rows], dtype=float)
    wrong = np.asarray([1 - row["para_em"] for row in rows], dtype=int)
    return {
        "dataset": ds,
        "n": len(rows),
        "matched": matched,
        "best": best,
        "never": never,
        "always": always,
        "decomp": decomp,
        "delta_decomp_vs_best": decomp["em"] - best["em"],
        "auroc_wrong": float(auroc(scores, wrong)),
        "audit": audit,
    }


def fscore(result: dict) -> str:
    return f"{result['em']:.2f}/{result['f1']:.2f}"


def ftau(tau: float) -> str:
    if math.isinf(tau):
        return "inf"
    return f"{tau:.9g}"


def render(results: list[dict]) -> str:
    lines = [
        "# Direct-only tl_sum gate recalculation",
        "",
        "- Gate: retrieve iff `tl_sum >= tau`.",
        f"- Fixed threshold: `tau={MATCHED_TAU}` (the deployed step-gate threshold).",
        "- Best-dev threshold candidates: every unique observed `tl_sum`, plus "
        "always/never endpoints.",
        "- Best-dev tie-break: maximum EM, maximum F1, minimum retrieval rate, "
        "then smallest threshold.",
        "- Always-retrieve direct and decompose+gate candidates: final "
        "`tl_sum` + prompt-fix-A traces (`replay_verifier_tlsum_fixA.build`).",
        "",
        "| Dataset | Direct never EM/F1 | Fixed gate EM/F1 (retr.) | "
        "Best-dev EM/F1 @ tau (retr.) | Direct always EM/F1 | "
        "Decompose+gate EM/F1 | Delta decomp - direct best | AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {fscore(r['never'])} | "
            f"{fscore(r['matched'])} ({r['matched']['retrieval_rate']:.1f}%) | "
            f"{fscore(r['best'])} @ {ftau(r['best']['tau'])} "
            f"({r['best']['retrieval_rate']:.1f}%) | "
            f"{fscore(r['always'])} | {fscore(r['decomp'])} | "
            f"{r['delta_decomp_vs_best']:+.2f} | {r['auroc_wrong']:.3f} |"
        )

    lines.extend(["", "## Source audit", ""])
    for r in results:
        a = r["audit"]
        lines.append(
            f"- {r['dataset']}: no-context regenerated-vs-legacy prediction "
            f"mismatches = {a['probe_prediction_mismatches']}/1000; final-vs-legacy "
            f"always-RAG prediction mismatches = "
            f"{a['legacy_final_rag_prediction_mismatches']}/1000 "
            f"(`results/{a['legacy_rag_file']}`)."
        )
    lines.extend([
        "",
        "The legacy PopQA always-RAG file is a top-3 run. The final trace uses "
        "the common top-8 configuration, so the final direct candidate is used "
        "for the reported PopQA direct-gate values.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(ROOT / "results/direct_gate_tlsum.md"),
    )
    args = parser.parse_args()

    results = [summarize(ds) for ds in DATASETS]
    report = render(results)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
