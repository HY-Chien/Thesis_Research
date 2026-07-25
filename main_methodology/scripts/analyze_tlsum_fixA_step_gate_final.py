#!/usr/bin/env python3
"""Score the fixed-chain tl_sum step-gate synthesis under the final verifier.

Operational confidence inputs are exclusively:

* step retrieval gate: ``new_steps[*].tl_sum``;
* direct verifier confidence: direct-answer ``tl_sum`` with tau=0.0005;
* decomposition verifier confidence: max retrieved ``post_tl_sum`` with
  tau=0.0158.

The verifier helpers retain historical parameter/field names, but this script
populates those generic slots only with the tl_sum values listed above.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_confidence_battery import compute_signals  # noqa: E402
from replay_typeaware_verifier import choose_typeaware, load_jsonl  # noqa: E402
from replay_tlsum_fixA_step_gate_synthesis import (  # noqa: E402
    DATASETS,
    DEFAULT_POLICIES,
)
from rerun_prompt_fix import TRACE  # noqa: E402
from seek.pipeline import _rescue_value_answer  # noqa: E402
from seek.verifier import choose_answer, exact_match, token_f1  # noqa: E402


DIRECT_TL_TAU = 0.0005
POST_TL_TAU = 0.0158
TYPEAWARE_MARGIN = 2.5


def load_fix_rows(ds: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for shard in (0, 1):
        path = ROOT / f"results/tlsum_fixA_{ds}_shard{shard}.jsonl"
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def load_syntheses(paths: list[Path]) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in paths:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            key = (row["ds"], str(row["id"]))
            if key in rows:
                raise ValueError(f"duplicate synthesis row: {key}")
            rows[key] = row
    return rows


def load_aux(ds: str) -> tuple[dict[str, dict], dict[str, list[str]], dict[str, float]]:
    trace = {
        str(row["id"]): row
        for row in load_jsonl(ROOT / TRACE.format(ds=ds))
    }
    golds: dict[str, list[str]] = {}
    direct_tl: dict[str, float] = {}
    path = ROOT / f"results/conf_battery_{ds}.jsonl"
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        identity = str(row["id"])
        golds[identity] = [row["gold"]] + (row.get("aliases") or [])
        signals = compute_signals(row) if row.get("tokens") else None
        if signals is None:
            raise ValueError(f"{ds}/{identity}: missing direct tl_sum tokens")
        direct_tl[identity] = float(signals["tl_sum"])
    return trace, golds, direct_tl


def policy_steps(stored_steps: list[dict], tau: float) -> list[dict]:
    replayed: list[dict] = []
    for step in stored_steps:
        retrieved = float(step["tl_sum"]) >= tau
        replayed.append({
            "step_id": step["step_id"],
            "raw_sub_q": step["raw_sub_q"],
            # Fixed-chain replay: this is the deployed trace's resolved query.
            "sub_q": step["sub_q"],
            "tentative": step.get("tentative") or "",
            "final_answer": (
                step.get("post_pred") or ""
                if retrieved
                else step.get("tentative") or ""
            ),
            "retrieved": retrieved,
            "tl_sum": float(step["tl_sum"]),
            "post_tl_sum": float(step["post_tl_sum"]),
        })
    return replayed


def final_choice(
    question: str,
    direct_record: dict,
    direct_pred: str,
    direct_tl: float,
    decomp_pred: str,
    decomposition: list,
    steps: list[dict],
) -> str:
    # choose_answer/choose_typeaware use historical generic confidence slots.
    # They are deliberately populated with tl_sum only in this replay.
    direct = dict(direct_record or {})
    direct["skip_probe_ent"] = direct_tl
    decomp_steps = [{
        "step_id": step["step_id"],
        "raw_sub_q": step["raw_sub_q"],
        "sub_q": step["sub_q"],
        "tentative": step["tentative"],
        "final_answer": step["final_answer"],
        "retrieved": step["retrieved"],
        "post_signal_value": step["post_tl_sum"] if step["retrieved"] else -1.0,
    } for step in steps]
    decomp_record = {
        "question": question,
        "pred": decomp_pred,
        "skipped_decompose": False,
        "decomposition": decomposition,
        "steps": decomp_steps,
    }
    verification = choose_answer(
        question,
        direct,
        decomp_record,
        fallback="auto",
        direct_entropy_tau=DIRECT_TL_TAU,
        decomp_post_entropy_tau=POST_TL_TAU,
        higher_is_uncertain=True,
    )
    verifier_row = {
        "question": question,
        "choice": verification.choice,
        "reason": verification.reason,
        "direct_pred": direct_pred,
        "decomp_pred": decomp_pred,
        "direct": direct,
        "decomp": decomp_record,
    }
    choice, _ = choose_typeaware(
        verifier_row,
        direct_entropy_tau=DIRECT_TL_TAU,
        decomp_post_tau=POST_TL_TAU,
        margin=TYPEAWARE_MARGIN,
    )
    return choice


def evaluate_policy(
    ds: str,
    name: str,
    tau: float,
    fix_rows: dict[str, dict],
    syntheses: dict[tuple[str, str], dict],
    trace: dict[str, dict],
    golds: dict[str, list[str]],
    direct_tl: dict[str, float],
) -> tuple[dict, int]:
    em_sum = f1_sum = decomp_em_sum = decomp_f1_sum = 0.0
    direct_em_sum = oracle_em_sum = 0.0
    retrieved = total_steps = direct_selected = 0
    deployed_mismatch = 0
    for identity, fix in fix_rows.items():
        source = trace[identity]
        gold = golds[identity]
        stored_steps = fix.get("new_steps") or []
        steps = policy_steps(stored_steps, tau)
        synth_row = syntheses[(ds, identity)]
        synth_index = int(synth_row["policy_to_synthesis"][name])
        raw_pred = synth_row["syntheses"][synth_index]
        namespaces = [
            SimpleNamespace(sub_q=step["sub_q"], final_answer=step["final_answer"])
            for step in steps
        ]
        decomp_pred = _rescue_value_answer(fix["question"], raw_pred, namespaces)

        if name == "deployed" and decomp_pred != (fix.get("tl") or {}).get("decomp_pred"):
            deployed_mismatch += 1

        direct_pred = source.get("direct_pred") or ""
        choice = final_choice(
            fix["question"],
            source.get("direct") or {},
            direct_pred,
            direct_tl[identity],
            decomp_pred,
            (source.get("decomp") or {}).get("decomposition") or [],
            steps,
        )
        pred = direct_pred if choice == "direct" else decomp_pred
        direct_ok = int(exact_match(direct_pred, gold))
        decomp_ok = int(exact_match(decomp_pred, gold))
        em_sum += int(exact_match(pred, gold))
        f1_sum += token_f1(pred, gold)
        direct_em_sum += direct_ok
        decomp_em_sum += decomp_ok
        decomp_f1_sum += token_f1(decomp_pred, gold)
        oracle_em_sum += int(direct_ok or decomp_ok)
        direct_selected += choice == "direct"
        retrieved += sum(step["retrieved"] for step in steps)
        total_steps += len(steps)

    n = len(fix_rows)
    return ({
        "n": n,
        "em": 100 * em_sum / n,
        "f1": 100 * f1_sum / n,
        "direct_em": 100 * direct_em_sum / n,
        "decomp_em": 100 * decomp_em_sum / n,
        "decomp_f1": 100 * decomp_f1_sum / n,
        "oracle_em": 100 * oracle_em_sum / n,
        "retrieved_steps": retrieved,
        "total_steps": total_steps,
        "retrieval_rate": 100 * retrieved / total_steps,
        "direct_selected": direct_selected,
        "direct_rate": 100 * direct_selected / n,
    }, deployed_mismatch)


def format_cell(row: dict) -> str:
    return f"{row['em']:.2f}/{row['f1']:.2f}/{row['retrieval_rate']:.2f}%"


def build_report(results: dict[str, dict[str, dict]], mismatches: dict[str, int]) -> str:
    lines = [
        "# Final-trace tl_sum step-gate ablation (fixed-chain replay)",
        "",
        "Inputs: `tlsum_fixA_{ds}_shard{0,1}.jsonl` only for alternate step-answer vectors. "
        "For each threshold, a step uses `post_pred` iff `tl_sum >= tau`; otherwise it uses "
        "`tentative`. Each unique answer chain is resynthesized with the same adapter-disabled "
        "Llama-3-8B synthesis prompt as the deployed run.",
        "",
        "**Fixed-chain definition.** Later `sub_q` strings are the already-resolved strings in the "
        "deployed tau=0.008461 trace. If an earlier answer changes under another threshold, later "
        "queries are not regenerated or re-resolved. Scores away from the deployed threshold are "
        "therefore controlled fixed-chain replays, not live end-to-end reruns.",
        "",
        "Downstream routing is the adopted all-`tl_sum` verifier: direct tau=0.0005 and "
        "post-decomposition tau=0.0158. No entropy score participates in gating or routing. "
        "EM/F1 are alias-aware. Retrieval rate is the percentage of decomposed steps that retrieve.",
        "",
        "## Sanity check",
        "",
    ]
    for ds in DATASETS:
        lines.append(f"- {ds}: deployed synthesis mismatches = {mismatches[ds]} / 1000")

    lines.extend([
        "",
        "## Never / deployed / always",
        "",
        "Each cell is final-system EM/F1/step-retrieval-rate.",
        "",
        "| Dataset | Never | Deployed tau=0.008461 | Always | Gate EM - Never EM |",
        "|---|---:|---:|---:|---:|",
    ])
    for ds in DATASETS:
        never = results[ds]["never"]
        deployed = results[ds]["deployed"]
        always = results[ds]["always"]
        lines.append(
            f"| {ds} | {format_cell(never)} | {format_cell(deployed)} | "
            f"{format_cell(always)} | {deployed['em'] - never['em']:+.2f} |"
        )

    macro_threeway = {}
    for policy in ("never", "deployed", "always"):
        policy_rows = [results[ds][policy] for ds in DATASETS]
        macro_threeway[policy] = {
            "em": float(np.mean([row["em"] for row in policy_rows])),
            "f1": float(np.mean([row["f1"] for row in policy_rows])),
            "retrieval_rate": float(np.mean([row["retrieval_rate"] for row in policy_rows])),
        }
    lines.append(
        f"| Macro | {format_cell(macro_threeway['never'])} | "
        f"{format_cell(macro_threeway['deployed'])} | "
        f"{format_cell(macro_threeway['always'])} | "
        f"{macro_threeway['deployed']['em'] - macro_threeway['never']['em']:+.2f} |"
    )

    lines.extend([
        "",
        "## Threshold sensitivity",
        "",
        "| tau | HotpotQA | 2Wiki | MuSiQue | PopQA | Macro EM/F1 | Macro retr. | Pooled-step retr. |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, tau in DEFAULT_POLICIES:
        if name in {"always", "never"}:
            continue
        ds_rows = [results[ds][name] for ds in DATASETS]
        macro_em = float(np.mean([row["em"] for row in ds_rows]))
        macro_f1 = float(np.mean([row["f1"] for row in ds_rows]))
        macro_retr = float(np.mean([row["retrieval_rate"] for row in ds_rows]))
        pooled_retr = 100 * sum(row["retrieved_steps"] for row in ds_rows) / sum(
            row["total_steps"] for row in ds_rows
        )
        cells = " | ".join(format_cell(row) for row in ds_rows)
        lines.append(
            f"| {tau:.6g} | {cells} | {macro_em:.2f}/{macro_f1:.2f} | "
            f"{macro_retr:.2f}% | {pooled_retr:.2f}% |"
        )

    lines.extend([
        "",
        "## Machine-readable details",
        "",
        "The companion JSON stores final-system and decomposition-candidate EM/F1, "
        "retrieved/total step counts, and direct-selection counts for every dataset/policy.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=ROOT / "results/tlsum_fixA_step_gate_final.md",
    )
    parser.add_argument(
        "--json-out", type=Path,
        default=ROOT / "results/tlsum_fixA_step_gate_final.json",
    )
    args = parser.parse_args()

    syntheses = load_syntheses(args.synth)
    results: dict[str, dict[str, dict]] = {}
    mismatches: dict[str, int] = {}
    expected_rows = 0
    for ds in DATASETS:
        fix_rows = load_fix_rows(ds)
        trace, golds, direct_tl = load_aux(ds)
        expected_rows += len(fix_rows)
        results[ds] = {}
        deployed_mismatch = None
        for name, tau in DEFAULT_POLICIES:
            scored, mismatch = evaluate_policy(
                ds, name, tau, fix_rows, syntheses, trace, golds, direct_tl
            )
            results[ds][name] = scored
            if name == "deployed":
                deployed_mismatch = mismatch
        mismatches[ds] = int(deployed_mismatch or 0)

    if len(syntheses) != expected_rows:
        raise ValueError(
            f"synthesis coverage mismatch: got {len(syntheses)}, expected {expected_rows}"
        )

    payload = {
        "protocol": {
            "replay": "fixed-chain",
            "step_gate_signal": "tl_sum",
            "step_gate_rule": "retrieve iff tl_sum >= tau",
            "deployed_step_tau": 0.008461,
            "direct_verifier_signal": "tl_sum",
            "direct_verifier_tau": DIRECT_TL_TAU,
            "post_verifier_signal": "post_tl_sum",
            "post_verifier_tau": POST_TL_TAU,
            "entropy_operational": False,
        },
        "deployed_synthesis_mismatches": mismatches,
        "results": results,
    }
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.out.write_text(build_report(results, mismatches) + "\n", encoding="utf-8")
    print(args.out.read_text(encoding="utf-8"))
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
