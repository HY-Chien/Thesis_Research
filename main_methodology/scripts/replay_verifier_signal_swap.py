"""Offline replay: swap the verifier's direct-side confidence signal.

The canonical *_typeaware_replay.jsonl decision stack is:
  1. choose_answer (verifier v2, no evidence): rule cascade; entropy only
     enters _auto_fallback (direct iff skip_probe_ent < 0.3 AND decomp
     unstable via failed/broken/max step post_signal_value > 0.8).
  2. typeaware layer (replay_typeaware_verifier.choose_typeaware) on rows
     still auto_fallback_decomp: direct entropy < tau adds +1 direct score,
     flip to direct at margin 2.5.

This script re-derives both stages from the stored records:
  - sanity: original signal (skip_probe_ent, tau 0.3) must reproduce the
    stored replay_choice / replay_em exactly;
  - swap: replace the direct-side signal with a conf-battery signal
    (min_top5m / max_renyi05), tau rate-matched on the pooled 4-dataset
    rate of (skip_probe_ent < 0.3) so no EM peeking; plus a rate sweep.

The decomp-side gate (per-step post_signal_value > 0.8) is kept as-is:
the battery only probed the synthesis step, not the sub-answer steps.

Usage: python scripts/replay_verifier_signal_swap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from seek.verifier import _decomp_failed, _looks_broken  # noqa: E402
from replay_typeaware_verifier import (  # noqa: E402
    answer_in_question, answer_type_mismatch, decomp_final_equals_intermediate,
    decomp_has_entity_fragment_step, load_jsonl, max_decomp_post_entropy,
    question_kind)
from analyze_confidence_battery import compute_signals  # noqa: E402

DATASETS = ["2wiki", "hotpotqa", "musique", "popqa"]
TRACE = "results/{ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl"
BATTERY = "results/conf_battery_{ds}.jsonl"
DECOMP_POST_TAU = 0.8
MARGIN = 2.5
AUTO_REASONS = {"auto_fallback_decomp", "auto_confident_direct_unstable_decomp"}


def decide(row: dict, sig: float | None, tau: float) -> tuple[str, str]:
    """Re-derive stages 1+2 with `sig` as the direct-side confidence signal."""
    reason = row.get("reason") or ""
    if reason not in AUTO_REASONS:
        choice = row.get("choice") or "decomp"
        return choice, "keep_non_auto"

    # stage 1: _auto_fallback
    decomp = row.get("decomp") or {}
    post = max_decomp_post_entropy(row)
    decomp_unstable = (
        _decomp_failed(decomp) or _looks_broken(decomp)
        or (post >= 0 and post > DECOMP_POST_TAU))
    confident = sig is not None and sig < tau
    if confident and decomp_unstable:
        return "direct", "auto_confident_direct_unstable_decomp"

    # stage 2: typeaware rescue on auto_fallback_decomp rows
    question = row.get("question") or ""
    direct_pred = row.get("direct_pred") or ""
    decomp_pred = row.get("decomp_pred") or ""
    d_score = 1.0 if confident else 0.0
    c_score = -1.0 if (post >= 0 and post > DECOMP_POST_TAU) else 0.0
    if answer_type_mismatch(question, decomp_pred) and not answer_type_mismatch(question, direct_pred):
        d_score += 2.0
    if answer_type_mismatch(question, direct_pred) and not answer_type_mismatch(question, decomp_pred):
        c_score += 2.0
    if decomp_has_entity_fragment_step(row):
        c_score -= 1.5
    if decomp_final_equals_intermediate(row):
        c_score -= 1.5
    if answer_in_question(decomp_pred, question) and question_kind(question) != "entity":
        c_score -= 1.0
    if answer_in_question(direct_pred, question) and question_kind(question) != "entity":
        d_score -= 1.0
    if d_score >= c_score + MARGIN:
        return "direct", "typeaware_direct"
    return "decomp", "auto_fallback_decomp"


def run(rows: list[dict], sigs: list[float | None], tau: float) -> dict:
    em = n_direct = flips = rescued = lost = 0
    for row, sig in zip(rows, sigs):
        choice, _ = decide(row, sig, tau)
        e = int(row.get("direct_em", 0)) if choice == "direct" else int(row.get("decomp_em", 0))
        em += e
        n_direct += choice == "direct"
        if choice != row.get("replay_choice"):
            flips += 1
            base_e = int(row.get("replay_em", 0))
            rescued += e and not base_e
            lost += base_e and not e
    n = len(rows)
    return {"em": 100 * em / n, "direct": n_direct, "flips": flips,
            "rescued": rescued, "lost": lost}


def main() -> None:
    data: dict[str, tuple[list[dict], dict[str, dict[str, float]]]] = {}
    for ds in DATASETS:
        rows = load_jsonl(ROOT / TRACE.format(ds=ds))
        sig_by_id: dict[str, dict[str, float]] = {}
        for line in open(ROOT / BATTERY.format(ds=ds), encoding="utf-8"):
            r = json.loads(line)
            s = compute_signals(r)
            if s is not None:
                sig_by_id[str(r["id"])] = s
        data[ds] = (rows, sig_by_id)
        n_hit = sum(str(r.get("id")) in sig_by_id for r in rows)
        base_em = 100 * sum(int(r.get("replay_em", 0)) for r in rows) / len(rows)
        print(f"{ds}: {len(rows)} rows, battery join {n_hit}/{len(rows)}, "
              f"stored replay EM={base_em:.1f}")

    def sig_vec(ds: str, name: str) -> list[float | None]:
        rows, sig_by_id = data[ds]
        return [sig_by_id.get(str(r.get("id")), {}).get(name) for r in rows]

    # sanity: skip_probe_ent must equal battery max_ent; tau=0.3 must reproduce
    print("\n== sanity: original signal reproduces stored replay ==")
    for ds in DATASETS:
        rows, _ = data[ds]
        stored = [(r.get("direct") or {}).get("skip_probe_ent") for r in rows]
        batt = sig_vec(ds, "max_ent")
        n_close = sum(1 for a, b in zip(stored, batt)
                      if a is not None and b is not None and abs(a - b) < 1e-3)
        res = run(rows, [a if isinstance(a, (int, float)) and a >= 0 else None
                         for a in stored], 0.3)
        rep_em = 100 * sum(int(r.get("replay_em", 0)) for r in rows) / len(rows)
        print(f"{ds}: ent match {n_close}/{len(rows)}  rederived EM={res['em']:.1f} "
              f"(stored {rep_em:.1f})  flips={res['flips']}")

    # pooled rate of the current operating point
    pooled_ent = [v for ds in DATASETS for v in sig_vec(ds, "max_ent") if v is not None]
    base_rate = float(np.mean(np.array(pooled_ent) < 0.3))
    print(f"\npooled P(max_ent<0.3) = {base_rate:.3f}")

    for name in ["max_ent", "min_top5m", "max_renyi05", "tl_sum"]:
        pooled = np.array([v for ds in DATASETS for v in sig_vec(ds, name)
                           if v is not None])
        print(f"\n== swap direct signal -> {name} ==")
        print(f"{'rate':>5} {'tau':>8} | " + " | ".join(f"{ds:>14}" for ds in DATASETS)
              + " | macro")
        for rate in [0.05, 0.10, 0.15, base_rate, 0.25, 0.30, 0.40, 0.50]:
            tau = float(np.quantile(pooled, rate))
            cells, ems = [], []
            for ds in DATASETS:
                rows, _ = data[ds]
                res = run(rows, sig_vec(ds, name), tau)
                cells.append(f"{res['em']:5.1f} +{res['rescued']:2d}/-{res['lost']:2d}")
                ems.append(res["em"])
            star = " <- rate-matched" if abs(rate - base_rate) < 1e-9 else ""
            print(f"{rate:5.2f} {tau:8.4f} | " + " | ".join(cells)
                  + f" | {np.mean(ems):5.2f}{star}")


if __name__ == "__main__":
    main()
