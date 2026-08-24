#!/usr/bin/env python3
"""Shared loader for the thesis accuracy-first fixed-chain replay.

The thesis headline uses the Tail-Leakage step gate at ``tau=0.002305``.
This module reconstructs that policy from the tracked prompt-fix-A traces and
the tracked fixed-chain synthesis artifacts, then applies the same final
all-Tail-Leakage/type-aware verifier as the headline analysis.

No result values are embedded here: every candidate answer, correctness flag,
and routing choice is recomputed from the bundled row-level artifacts.
"""
from __future__ import annotations

from pathlib import Path

from analyze_skipped_retrieval_questions import materialize_policy
from analyze_tlsum_fixA_step_gate_final import (
    ROOT,
    load_aux,
    load_fix_rows,
    load_syntheses,
)
from seek.verifier import exact_match, token_f1


ACCURACY_FIRST_POLICY = "tau_0.002305"
ACCURACY_FIRST_TAU = 0.002305
DEFAULT_SYNTH_PATHS = (
    ROOT / "results/traces/other/tlsum_fixA_step_gate_synth_shard0.jsonl",
    ROOT / "results/traces/other/tlsum_fixA_step_gate_synth_shard1.jsonl",
)


def load_accuracy_first_rows(
    dataset: str,
    syntheses: dict[tuple[str, str], dict],
) -> list[dict]:
    """Return row-level candidates and final choices for one dataset."""
    fix_rows = load_fix_rows(dataset)
    trace, golds_by_id, direct_tl_by_id = load_aux(dataset)
    missing = [
        identity
        for identity in fix_rows
        if (dataset, identity) not in syntheses
    ]
    if missing:
        raise ValueError(
            f"{dataset}: synthesis coverage is missing {len(missing)} rows; "
            f"first missing id={missing[0]}"
        )
    rows: list[dict] = []

    for identity in sorted(fix_rows):
        fix = fix_rows[identity]
        source = trace[identity]
        golds = golds_by_id[identity]
        replay = materialize_policy(
            dataset,
            identity,
            ACCURACY_FIRST_POLICY,
            ACCURACY_FIRST_TAU,
            fix,
            syntheses,
            source,
            golds,
            direct_tl_by_id[identity],
        )
        direct_pred = source.get("direct_pred") or ""
        rows.append({
            "dataset": dataset,
            "id": identity,
            "question": fix["question"],
            "golds": golds,
            "direct": source.get("direct") or {},
            "decomposition": (
                (source.get("decomp") or {}).get("decomposition") or []
            ),
            "direct_pred": direct_pred,
            "direct_tl": float(direct_tl_by_id[identity]),
            "direct_em": int(exact_match(direct_pred, golds)),
            "direct_f1": token_f1(direct_pred, golds),
            "decomp_pred": replay["decomp_pred"],
            "decomp_em": replay["decomp_em"],
            "decomp_f1": replay["decomp_f1"],
            "steps": replay["steps"],
            "final_choice": replay["choice"],
            "final_pred": replay["final_pred"],
            "final_em": replay["final_em"],
            "final_f1": replay["final_f1"],
        })

    return rows


def load_all_accuracy_first_rows(
    synth_paths: list[Path] | tuple[Path, ...] = DEFAULT_SYNTH_PATHS,
    datasets: tuple[str, ...] = ("hotpotqa", "2wiki", "musique", "popqa"),
) -> dict[str, list[dict]]:
    """Load synthesis artifacts once and reconstruct all requested datasets."""
    syntheses = load_syntheses([Path(path) for path in synth_paths])
    return {
        dataset: load_accuracy_first_rows(dataset, syntheses)
        for dataset in datasets
    }
