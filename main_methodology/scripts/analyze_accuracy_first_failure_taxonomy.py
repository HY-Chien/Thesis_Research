#!/usr/bin/env python3
"""Recompute decomposition-path failure diagnostics at tau=0.002305.

The accuracy-first result is a fixed-chain replay: retrieved and no-retrieval
step answers are selected at the requested threshold and the resulting answer
chain is resynthesized, while later resolved sub-question strings remain fixed
to the saved live trace.  This script joins those per-question syntheses with
the aligned original-query and decomposed-query coverage traces.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_final_tlsum_failure_taxonomy import (  # noqa: E402
    intermediate_echo,
    subq1_restates_question,
)
from analyze_final_tlsum_failure_taxonomy_aligned import (  # noqa: E402
    CATEGORIES,
    category,
)
from analyze_tlsum_fixA_step_gate_final import (  # noqa: E402
    load_aux,
    load_fix_rows,
    load_syntheses,
    policy_steps,
)
from seek.pipeline import _rescue_value_answer  # noqa: E402
from seek.verifier import exact_match  # noqa: E402


DATASETS = ("hotpotqa", "2wiki", "musique")
POLICY = "tau_0.002305"
TAU = 0.002305
DEFAULT_SYNTHESIS_PATHS = (
    ROOT / "results/traces/other/tlsum_fixA_step_gate_synth_shard0.jsonl",
    ROOT / "results/traces/other/tlsum_fixA_step_gate_synth_shard1.jsonl",
)


def load_coverage(ds: str) -> tuple[dict[str, int], dict[str, int], dict[str, list[str]]]:
    trace_dir = ROOT / "results/traces" / ds
    original: dict[str, int] = {}
    with (trace_dir / f"{ds}_m3_wiki_detail.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            original[str(row["id"])] = int(row["cov"]["10"]["A"]["full"])

    decomposed: dict[str, int] = {}
    queries: dict[str, list[str]] = {}
    path = trace_dir / f"{ds}_final_tlsum_bunion_coverage.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["k"]) != 10:
                raise ValueError(f"expected coverage@10 in {path}")
            identity = str(row["id"])
            decomposed[identity] = int(row["Bunion"]["full"])
            queries[identity] = row["queries"]
    return original, decomposed, queries


def build_policy_rows(ds: str, syntheses: dict[tuple[str, str], dict]) -> list[dict]:
    fix_rows = load_fix_rows(ds)
    trace, golds, _ = load_aux(ds)
    rows: list[dict] = []
    for identity, fix in fix_rows.items():
        source = trace[identity]
        steps = policy_steps(fix.get("new_steps") or [], TAU)
        synth_row = syntheses[(ds, identity)]
        synth_index = int(synth_row["policy_to_synthesis"][POLICY])
        raw_pred = synth_row["syntheses"][synth_index]
        namespaces = [
            SimpleNamespace(sub_q=step["sub_q"], final_answer=step["final_answer"])
            for step in steps
        ]
        decomp_pred = _rescue_value_answer(fix["question"], raw_pred, namespaces)
        direct_pred = source.get("direct_pred") or ""
        gold = golds[identity]
        rows.append({
            "id": identity,
            "ds": ds,
            "question": fix["question"],
            "golds": gold,
            "steps": steps,
            "direct_pred": direct_pred,
            "decomp_pred": decomp_pred,
            "direct_em": int(exact_match(direct_pred, gold)),
            "decomp_em": int(exact_match(decomp_pred, gold)),
        })
    if len(rows) != 1000:
        raise ValueError(f"{ds}: expected 1000 rows, got {len(rows)}")
    return rows


def pct(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.1f}%)"


def analyse_dataset(ds: str, syntheses: dict[tuple[str, str], dict]) -> dict:
    rows = build_policy_rows(ds, syntheses)
    original_cov, decomposed_cov, stored_queries = load_coverage(ds)
    if len(original_cov) != len(rows) or len(decomposed_cov) != len(rows):
        raise ValueError(
            f"{ds}: coverage size mismatch rows={len(rows)} "
            f"original={len(original_cov)} decomposed={len(decomposed_cov)}"
        )
    for row in rows:
        expected = [step.get("sub_q") or step.get("raw_sub_q") or "" for step in row["steps"]]
        expected = [query for query in expected if query] or [row["question"]]
        if stored_queries[row["id"]] != expected:
            raise ValueError(f"{ds}/{row['id']}: coverage query mismatch")

    decomp_wrong = [row for row in rows if not row["decomp_em"]]
    counts = collections.Counter(
        category(row, original_cov, decomposed_cov) for row in decomp_wrong
    )
    if sum(counts.values()) != len(decomp_wrong):
        raise AssertionError(f"{ds}: taxonomy does not partition decomposition errors")

    flags = {
        "intermediate_answer_echo": sum(intermediate_echo(row) for row in decomp_wrong),
        "first_subquestion_restates_original": sum(
            subq1_restates_question(row["question"], row["steps"])
            for row in decomp_wrong
        ),
        "no_steps_retrieve": sum(
            bool(row["steps"]) and not any(step["retrieved"] for step in row["steps"])
            for row in decomp_wrong
        ),
        "mixed_retrieval": sum(
            bool(row["steps"])
            and any(step["retrieved"] for step in row["steps"])
            and not all(step["retrieved"] for step in row["steps"])
            for row in decomp_wrong
        ),
        "all_steps_retrieve": sum(
            bool(row["steps"]) and all(step["retrieved"] for step in row["steps"])
            for row in decomp_wrong
        ),
    }
    return {
        "n": len(rows),
        "decomp_errors": len(decomp_wrong),
        "decomp_em": 100 * (len(rows) - len(decomp_wrong)) / len(rows),
        "categories": {name: counts[name] for name in CATEGORIES},
        "flags": flags,
    }


def render(results: dict[str, dict]) -> str:
    lines = [
        "# Accuracy-first decomposition-path failure taxonomy",
        "",
        "Configuration: step-level Tail Leakage gate at `tau=0.002305`. "
        "The analysis uses the final fixed-chain resynthesis outputs; later resolved "
        "sub-question strings remain fixed to the saved live trace.",
        "",
        "Supporting-document coverage is title coverage@10 for the original question "
        "and the union of the fixed resolved sub-questions.",
        "",
        "## Primary taxonomy",
        "",
        "| Category | HotpotQA | 2Wiki | MuSiQue |",
        "|---|---:|---:|---:|",
    ]
    for name in CATEGORIES:
        lines.append("| " + name + " | " + " | ".join(
            pct(results[ds]["categories"][name], results[ds]["decomp_errors"])
            for ds in DATASETS
        ) + " |")

    lines += [
        "",
        "## Supplementary flags",
        "",
        "Structural flags may overlap. The final three retrieval-state rows form a partition.",
        "",
        "| Flag | HotpotQA | 2Wiki | MuSiQue |",
        "|---|---:|---:|---:|",
    ]
    flag_names = (
        "intermediate_answer_echo",
        "first_subquestion_restates_original",
        "no_steps_retrieve",
        "mixed_retrieval",
        "all_steps_retrieve",
    )
    for name in flag_names:
        lines.append("| " + name + " | " + " | ".join(
            pct(results[ds]["flags"][name], results[ds]["decomp_errors"])
            for ds in DATASETS
        ) + " |")

    lines += [
        "",
        "## Sanity checks",
        "",
    ]
    for ds in DATASETS:
        row = results[ds]
        lines.append(
            f"- {ds}: decomposition EM={row['decomp_em']:.2f}; "
            f"errors={row['decomp_errors']}/{row['n']}."
        )
    lines += [
        "",
        "## Provenance",
        "",
        "- Step answers: `results/traces/{ds}/tlsum_fixA_{ds}_shard{0,1}.jsonl`.",
        "- Fixed-chain syntheses: `results/traces/other/tlsum_fixA_step_gate_synth_shard{0,1}.jsonl`.",
        "- Original-query coverage: `results/traces/{ds}/{ds}_m3_wiki_detail.jsonl`.",
        "- Decomposed-query coverage: `results/traces/{ds}/{ds}_final_tlsum_bunion_coverage.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/reproduced/decomp_failure_taxonomy_tau0p002305.md",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=ROOT / "results/reproduced/decomp_failure_taxonomy_tau0p002305.json",
    )
    args = parser.parse_args()

    syntheses = load_syntheses(list(DEFAULT_SYNTHESIS_PATHS))
    results = {ds: analyse_dataset(ds, syntheses) for ds in DATASETS}
    payload = {
        "protocol": {
            "step_gate_signal": "tl_sum",
            "step_gate_tau": TAU,
            "replay": "fixed-chain",
            "coverage_k": 10,
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results) + "\n", encoding="utf-8")
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.out.read_text(encoding="utf-8"))
    print(f"wrote {args.out}")
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
