#!/usr/bin/env python3
"""Compare SEEK with always-retrieve on questions where SEEK skips a step.

This analysis uses the thesis headline Tail Leakage operating point
(``tau=0.002305``).  It reuses the bundled fixed-chain synthesis artifacts and
the final all-Tail-Leakage/type-aware verifier, so its predictions are exactly
aligned with ``analyze_tlsum_fixA_step_gate_final.py``.

The primary population is question-level: a question is included when at least
one decomposed step has ``tl_sum < 0.002305`` and therefore skips retrieval in
SEEK.  The report gives the requested paired exact-match quadrants for the
final system and, as a diagnostic, for the decomposition candidate alone.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from analyze_tlsum_fixA_step_gate_final import (
    ROOT,
    final_choice,
    load_aux,
    load_fix_rows,
    load_syntheses,
    policy_steps,
)
from replay_tlsum_fixA_step_gate_synthesis import DATASETS
from seek.pipeline import _rescue_value_answer
from seek.verifier import exact_match, normalize, token_f1


TARGET_TAU = 0.002305
SEEK_POLICY = "tau_0.002305"
ALWAYS_POLICY = "always"
QUADRANTS = ("both_correct", "seek_only", "always_only", "both_wrong")
QUADRANT_LABELS = {
    "both_correct": "Both correct",
    "seek_only": "SEEK only",
    "always_only": "Always only",
    "both_wrong": "Both wrong",
}


def materialize_policy(
    ds: str,
    identity: str,
    policy_name: str,
    tau: float,
    fix: dict,
    syntheses: dict[tuple[str, str], dict],
    source: dict,
    golds: list[str],
    direct_tl: float,
) -> dict:
    """Reconstruct one policy's decomposition and final-system prediction."""
    steps = policy_steps(fix.get("new_steps") or [], tau)
    synth_row = syntheses[(ds, identity)]
    synth_index = int(synth_row["policy_to_synthesis"][policy_name])
    raw_decomp_pred = synth_row["syntheses"][synth_index]
    namespaces = [
        SimpleNamespace(sub_q=step["sub_q"], final_answer=step["final_answer"])
        for step in steps
    ]
    decomp_pred = _rescue_value_answer(
        fix["question"], raw_decomp_pred, namespaces
    )

    direct_pred = source.get("direct_pred") or ""
    choice = final_choice(
        fix["question"],
        source.get("direct") or {},
        direct_pred,
        direct_tl,
        decomp_pred,
        (source.get("decomp") or {}).get("decomposition") or [],
        steps,
    )
    final_pred = direct_pred if choice == "direct" else decomp_pred
    return {
        "steps": steps,
        "decomp_pred": decomp_pred,
        "decomp_em": int(exact_match(decomp_pred, golds)),
        "decomp_f1": token_f1(decomp_pred, golds),
        "choice": choice,
        "final_pred": final_pred,
        "final_em": int(exact_match(final_pred, golds)),
        "final_f1": token_f1(final_pred, golds),
    }


def quadrant(seek_ok: int, always_ok: int) -> str:
    if seek_ok and always_ok:
        return "both_correct"
    if seek_ok:
        return "seek_only"
    if always_ok:
        return "always_only"
    return "both_wrong"


def exact_mcnemar_p(seek_only: int, always_only: int) -> float:
    """Two-sided exact McNemar p-value (binomial form)."""
    discordant = seek_only + always_only
    if discordant == 0:
        return 1.0
    smaller = min(seek_only, always_only)
    lower_tail = sum(
        math.comb(discordant, k) for k in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def summarize_cases(cases: list[dict], total_questions: int, total_steps: int) -> dict:
    quadrants = Counter(case["final_quadrant"] for case in cases)
    decomp_quadrants = Counter(case["decomp_quadrant"] for case in cases)
    skipped_steps = sum(case["skipped_step_count"] for case in cases)
    changed_skipped_steps = sum(case["changed_skipped_step_count"] for case in cases)
    n = len(cases)
    seek_correct = quadrants["both_correct"] + quadrants["seek_only"]
    always_correct = quadrants["both_correct"] + quadrants["always_only"]
    return {
        "total_questions": total_questions,
        "affected_questions": n,
        "affected_question_rate": 100 * n / total_questions,
        "total_steps": total_steps,
        "skipped_steps": skipped_steps,
        "step_skip_rate": 100 * skipped_steps / total_steps,
        "changed_skipped_steps": changed_skipped_steps,
        "changed_skipped_step_rate": (
            100 * changed_skipped_steps / skipped_steps if skipped_steps else 0.0
        ),
        "all_skipped_step_answers_same_questions": sum(
            case["changed_skipped_step_count"] == 0 for case in cases
        ),
        "decomp_prediction_same_questions": sum(
            case["decomp_prediction_same"] for case in cases
        ),
        "final_prediction_same_questions": sum(
            case["final_prediction_same"] for case in cases
        ),
        "verifier_choice_same_questions": sum(
            case["verifier_choice_same"] for case in cases
        ),
        "quadrants": {name: quadrants[name] for name in QUADRANTS},
        "quadrant_rates": {
            name: 100 * quadrants[name] / n if n else 0.0 for name in QUADRANTS
        },
        "decomp_quadrants": {name: decomp_quadrants[name] for name in QUADRANTS},
        "decomp_quadrant_rates": {
            name: 100 * decomp_quadrants[name] / n if n else 0.0
            for name in QUADRANTS
        },
        "seek_em": 100 * seek_correct / n if n else 0.0,
        "always_em": 100 * always_correct / n if n else 0.0,
        "seek_minus_always_em": (
            100 * (seek_correct - always_correct) / n if n else 0.0
        ),
        "mcnemar_exact_p": exact_mcnemar_p(
            quadrants["seek_only"], quadrants["always_only"]
        ),
    }


def format_count_rate(summary: dict, name: str, decomp: bool = False) -> str:
    counts_key = "decomp_quadrants" if decomp else "quadrants"
    rates_key = "decomp_quadrant_rates" if decomp else "quadrant_rates"
    return f"{summary[counts_key][name]} ({summary[rates_key][name]:.2f}%)"


def comparison_table(summaries: dict[str, dict], decomp: bool = False) -> list[str]:
    lines = [
        "| Dataset | Affected questions | Both correct | SEEK only | Always only | Both wrong |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for ds in (*DATASETS, "pooled"):
        summary = summaries[ds]
        affected = (
            f"{summary['affected_questions']} / {summary['total_questions']} "
            f"({summary['affected_question_rate']:.2f}%)"
        )
        cells = " | ".join(
            format_count_rate(summary, name, decomp=decomp) for name in QUADRANTS
        )
        lines.append(f"| {ds} | {affected} | {cells} |")
    return lines


def build_report(summaries: dict[str, dict], cases_path: Path) -> str:
    pooled = summaries["pooled"]
    lines = [
        "# Accuracy-first skipped-retrieval question comparison",
        "",
        "## Protocol",
        "",
        f"- SEEK uses the thesis headline Tail Leakage step gate: `tau={TARGET_TAU}`; "
        "Always Retrieve uses `tau=0`.",
        "- Population: only questions where SEEK skips at least one decomposition step. "
        "A question with several steps is counted once in the four-way table.",
        "- Predictions come from the bundled controlled fixed-chain replay, followed by the "
        "same all-Tail-Leakage/type-aware final verifier as the thesis headline.",
        "- Correctness is alias-aware exact match (EM). Later resolved sub-questions are fixed "
        "to the source trace, so this is not a separate live end-to-end rerun.",
        "",
        "## Final-system paired comparison (primary)",
        "",
    ]
    lines.extend(comparison_table(summaries))
    lines.extend([
        "",
        "Percentages in the four outcome columns use affected questions as the denominator.",
        "",
        "## Accuracy on affected questions",
        "",
        "| Dataset | SEEK EM | Always EM | SEEK - Always | Exact McNemar p |",
        "|---|---:|---:|---:|---:|",
    ])
    for ds in (*DATASETS, "pooled"):
        summary = summaries[ds]
        lines.append(
            f"| {ds} | {summary['seek_em']:.2f} | {summary['always_em']:.2f} | "
            f"{summary['seek_minus_always_em']:+.2f} pp | "
            f"{summary['mcnemar_exact_p']:.6g} |"
        )

    lines.extend([
        "",
        "## Decomposition-candidate diagnostic",
        "",
        "This removes final direct/decomposition routing from the comparison and asks whether "
        "the changed step-answer chain alone changes exact-match correctness.",
        "",
    ])
    lines.extend(comparison_table(summaries, decomp=True))

    lines.extend([
        "",
        "## Retrieval omission diagnostics",
        "",
        "| Dataset | Skipped steps / all steps | Skipped answers changed by retrieval | "
        "Same decomp prediction | Same final prediction |",
        "|---|---:|---:|---:|---:|",
    ])
    for ds in (*DATASETS, "pooled"):
        summary = summaries[ds]
        n = summary["affected_questions"]
        skipped = summary["skipped_steps"]
        lines.append(
            f"| {ds} | {skipped} / {summary['total_steps']} "
            f"({summary['step_skip_rate']:.2f}%) | "
            f"{summary['changed_skipped_steps']} / {skipped} "
            f"({summary['changed_skipped_step_rate']:.2f}%) | "
            f"{summary['decomp_prediction_same_questions']} / {n} "
            f"({100 * summary['decomp_prediction_same_questions'] / n:.2f}%) | "
            f"{summary['final_prediction_same_questions']} / {n} "
            f"({100 * summary['final_prediction_same_questions'] / n:.2f}%) |"
        )

    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "- `Both correct`: skipping did not lose final EM correctness on that question.",
        "- `SEEK only`: skipping was beneficial for final EM.",
        "- `Always only`: direct evidence that the omission was harmful for final EM.",
        "- `Both wrong`: inconclusive about retrieval necessity; neither policy solved the question.",
        "",
        "Accordingly, `Both wrong` should not be labelled unnecessary retrieval merely because "
        "Always Retrieve also fails.",
        "",
        "## Artifacts",
        "",
        f"The row-level comparison is stored in `{cases_path}`. Each row contains the question, "
        "gold aliases, skipped step details, both decomposition/final predictions, verifier "
        "choices, EM/F1, and both quadrant labels.",
        "",
        f"Pooled check: {pooled['skipped_steps']} / {pooled['total_steps']} steps are skipped "
        f"({pooled['step_skip_rate']:.2f}%), matching the headline 6.16% retrieval reduction.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synth",
        nargs="+",
        type=Path,
        default=[
            ROOT / "results/traces/other/tlsum_fixA_step_gate_synth_shard0.jsonl",
            ROOT / "results/traces/other/tlsum_fixA_step_gate_synth_shard1.jsonl",
        ],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/reproduced/skipped_retrieval_question_comparison.md",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "results/reproduced/skipped_retrieval_question_comparison.json",
    )
    parser.add_argument(
        "--cases-out",
        type=Path,
        default=ROOT / "results/reproduced/skipped_retrieval_question_cases.jsonl",
    )
    args = parser.parse_args()

    syntheses = load_syntheses(args.synth)
    cases: list[dict] = []
    dataset_totals: dict[str, tuple[int, int]] = {}

    for ds in DATASETS:
        fix_rows = load_fix_rows(ds)
        trace, golds_by_id, direct_tl = load_aux(ds)
        total_steps = sum(len(row.get("new_steps") or []) for row in fix_rows.values())
        dataset_totals[ds] = (len(fix_rows), total_steps)

        for identity in sorted(fix_rows):
            fix = fix_rows[identity]
            stored_steps = fix.get("new_steps") or []
            skipped = [step for step in stored_steps if float(step["tl_sum"]) < TARGET_TAU]
            if not skipped:
                continue

            source = trace[identity]
            golds = golds_by_id[identity]
            seek = materialize_policy(
                ds, identity, SEEK_POLICY, TARGET_TAU, fix, syntheses,
                source, golds, direct_tl[identity],
            )
            always = materialize_policy(
                ds, identity, ALWAYS_POLICY, 0.0, fix, syntheses,
                source, golds, direct_tl[identity],
            )
            if sum(step["retrieved"] for step in seek["steps"]) != len(stored_steps) - len(skipped):
                raise AssertionError(f"{ds}/{identity}: SEEK skip count mismatch")
            if not all(step["retrieved"] for step in always["steps"]):
                raise AssertionError(f"{ds}/{identity}: Always policy skipped a step")

            skipped_details = []
            for step in skipped:
                tentative = step.get("tentative") or ""
                retrieved_answer = step.get("post_pred") or ""
                skipped_details.append({
                    "step_id": step["step_id"],
                    "sub_q": step["sub_q"],
                    "tl_sum": float(step["tl_sum"]),
                    "seek_answer_without_retrieval": tentative,
                    "always_answer_with_retrieval": retrieved_answer,
                    "answer_changed": normalize(tentative) != normalize(retrieved_answer),
                })

            cases.append({
                "dataset": ds,
                "id": identity,
                "question": fix["question"],
                "golds": golds,
                "total_step_count": len(stored_steps),
                "skipped_step_count": len(skipped_details),
                "changed_skipped_step_count": sum(
                    step["answer_changed"] for step in skipped_details
                ),
                "skipped_steps": skipped_details,
                "seek": {
                    key: seek[key] for key in (
                        "decomp_pred", "decomp_em", "decomp_f1", "choice",
                        "final_pred", "final_em", "final_f1",
                    )
                },
                "always_retrieve": {
                    key: always[key] for key in (
                        "decomp_pred", "decomp_em", "decomp_f1", "choice",
                        "final_pred", "final_em", "final_f1",
                    )
                },
                "decomp_prediction_same": (
                    normalize(seek["decomp_pred"]) == normalize(always["decomp_pred"])
                ),
                "final_prediction_same": (
                    normalize(seek["final_pred"]) == normalize(always["final_pred"])
                ),
                "verifier_choice_same": seek["choice"] == always["choice"],
                "decomp_quadrant": quadrant(seek["decomp_em"], always["decomp_em"]),
                "final_quadrant": quadrant(seek["final_em"], always["final_em"]),
            })

    summaries: dict[str, dict] = {}
    for ds in DATASETS:
        ds_cases = [case for case in cases if case["dataset"] == ds]
        summaries[ds] = summarize_cases(ds_cases, *dataset_totals[ds])
    summaries["pooled"] = summarize_cases(
        cases,
        sum(total[0] for total in dataset_totals.values()),
        sum(total[1] for total in dataset_totals.values()),
    )

    payload = {
        "protocol": {
            "population": "questions with at least one skipped decomposition step",
            "seek_policy": SEEK_POLICY,
            "seek_tau": TARGET_TAU,
            "always_policy": ALWAYS_POLICY,
            "replay": "fixed-chain",
            "metric": "alias-aware exact match",
        },
        "summaries": summaries,
    }
    for path in (args.out, args.json_out, args.cases_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_report(summaries, args.cases_out), encoding="utf-8")
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.cases_out.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    pooled = summaries["pooled"]
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.cases_out}")
    print(
        f"affected={pooled['affected_questions']}/{pooled['total_questions']}; "
        f"skipped_steps={pooled['skipped_steps']}/{pooled['total_steps']} "
        f"({pooled['step_skip_rate']:.2f}%)"
    )
    print(
        "quadrants=" + ", ".join(
            f"{name}:{pooled['quadrants'][name]}" for name in QUADRANTS
        )
    )


if __name__ == "__main__":
    main()
