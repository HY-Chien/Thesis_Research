"""Integrate final tl_sum traces with freshly rerun B-union coverage.

This is the authoritative four-category decomposition failure taxonomy.  It
uses exact final prompt-fix-A resolved sub-questions from
``results/{ds}_final_tlsum_bunion_coverage.jsonl`` and never reads the legacy
B-union coverage field.

Output: results/decomp_failure_taxonomy_final_tlsum_aligned.md
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "decomp_failure_taxonomy_final_tlsum_aligned.md"
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_final_tlsum_failure_taxonomy import (  # noqa: E402
    DATASETS,
    build,
    load_original_coverage,
    matches_any_gold,
)


CATEGORIES = (
    "retrieval_gap_both",
    "answer_despite_coverage",
    "synthesis_failure",
    "constraint_loss",
)


def load_final_bunion(ds: str) -> tuple[dict[str, int], dict[str, list[str]]]:
    coverage: dict[str, int] = {}
    queries: dict[str, list[str]] = {}
    path = RESULTS / f"{ds}_final_tlsum_bunion_coverage.jsonl"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if int(rec["k"]) != 10:
                raise ValueError(f"expected k=10 in {path}")
            rid = str(rec["id"])
            coverage[rid] = int(rec["Bunion"]["full"])
            queries[rid] = rec["queries"]
    return coverage, queries


def category(row: dict, a_cov: dict[str, int], b_cov: dict[str, int]) -> str:
    if any(matches_any_gold(step.get("final_answer") or "", row["golds"])
           for step in row["steps"]):
        return "synthesis_failure"
    rid = row["id"]
    if rid not in a_cov or rid not in b_cov:
        raise KeyError(f"coverage join missing: {row['ds']} {rid}")
    if a_cov[rid] == 1 and b_cov[rid] == 0:
        return "constraint_loss"
    if b_cov[rid] == 1:
        return "answer_despite_coverage"
    return "retrieval_gap_both"


def pct(count: int, total: int) -> str:
    return f"{count} ({100 * count / total:.1f}%)"


def main() -> None:
    lines = [
        "# Final all-tl_sum decomposition failure taxonomy (aligned coverage)",
        "",
        "Final configuration: prompt-fix A, step `tl_sum >= 0.008461`, "
        "direct verifier `tau_d=0.0005`, post-decomposition verifier "
        "`tau_p=0.0158`. B-union supporting-title coverage@10 is rerun from "
        "the exact final resolved sub-question strings and stores those "
        "strings in the source JSONL.",
        "",
    ]
    summary: dict[str, tuple[int, collections.Counter]] = {}

    for ds in DATASETS:
        rows = build(ds)
        a_cov = load_original_coverage(ds)
        b_cov, queries = load_final_bunion(ds)
        if len(a_cov) != len(rows) or len(b_cov) != len(rows):
            raise ValueError(
                f"coverage size mismatch for {ds}: rows={len(rows)} "
                f"A={len(a_cov)} B={len(b_cov)}"
            )
        for row in rows:
            expected = [step.get("sub_q") or step.get("raw_sub_q") or ""
                        for step in row["steps"]]
            expected = [query for query in expected if query] or [row["question"]]
            if queries[row["id"]] != expected:
                raise ValueError(f"query alignment mismatch: {ds} {row['id']}")

        both_correct = [r for r in rows if r["direct_em"] and r["decomp_em"]]
        direct_only = [r for r in rows if r["direct_em"] and not r["decomp_em"]]
        decomp_only = [r for r in rows if not r["direct_em"] and r["decomp_em"]]
        both_wrong = [r for r in rows if not r["direct_em"] and not r["decomp_em"]]
        direct_wrong = decomp_only + both_wrong
        decomp_wrong = direct_only + both_wrong

        by_group = {
            "direct_only": collections.Counter(
                category(row, a_cov, b_cov) for row in direct_only
            ),
            "both_wrong": collections.Counter(
                category(row, a_cov, b_cov) for row in both_wrong
            ),
        }
        total = by_group["direct_only"] + by_group["both_wrong"]
        if sum(total.values()) != len(decomp_wrong):
            raise AssertionError(f"taxonomy does not partition {ds}")

        lines += [
            f"## {ds} (n={len(rows)})",
            "",
            f"Direct candidate wrong: {pct(len(direct_wrong), len(rows))}; "
            f"decomposition candidate wrong: {pct(len(decomp_wrong), len(rows))}.",
            "",
            "| correctness group | count | share |",
            "|---|---:|---:|",
            f"| both correct | {len(both_correct)} | "
            f"{100 * len(both_correct) / len(rows):.1f}% |",
            f"| direct only | {len(direct_only)} | "
            f"{100 * len(direct_only) / len(rows):.1f}% |",
            f"| decompose only | {len(decomp_only)} | "
            f"{100 * len(decomp_only) / len(rows):.1f}% |",
            f"| both wrong | {len(both_wrong)} | "
            f"{100 * len(both_wrong) / len(rows):.1f}% |",
            "",
            "| primary category | direct-only | both-wrong | total | "
            "share of decomp errors |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in CATEGORIES:
            count = total[name]
            lines.append(
                f"| {name} | {by_group['direct_only'][name]} | "
                f"{by_group['both_wrong'][name]} | {count} | "
                f"{100 * count / len(decomp_wrong):.1f}% |"
            )

        # The coverage cross-tab is a useful integrity check and diagnostic.
        lines += [
            "",
            "| correctness group | n | A original full@10 | "
            "B-union final full@10 |",
            "|---|---:|---:|---:|",
        ]
        for label, group in (
            ("both correct", both_correct),
            ("direct only", direct_only),
            ("decompose only", decomp_only),
            ("both wrong", both_wrong),
        ):
            a_count = sum(a_cov[row["id"]] for row in group)
            b_count = sum(b_cov[row["id"]] for row in group)
            lines.append(
                f"| {label} | {len(group)} | {pct(a_count, len(group))} | "
                f"{pct(b_count, len(group))} |"
            )
        lines.append("")
        summary[ds] = (len(decomp_wrong), total)

    lines += [
        "## Thesis-ready table",
        "",
        "| Category | HotpotQA | 2Wiki | MuSiQue |",
        "|---|---:|---:|---:|",
    ]
    for name in CATEGORIES:
        lines.append(f"| {name} | " + " | ".join(
            pct(summary[ds][1][name], summary[ds][0]) for ds in DATASETS
        ) + " |")
    lines += [
        "",
        "## Provenance",
        "",
        "- Final candidates and final resolved step answers: "
        "`results/tlsum_fixA_{ds}_shard{0,1}.jsonl`, joined by "
        "`scripts/replay_verifier_tlsum_fixA.py`.",
        "- Original-query A coverage: `results/{ds}_m3_wiki_detail.jsonl`, "
        "using only the unchanged A/full@10 field.",
        "- Fresh final B-union coverage and exact query strings: "
        "`results/{ds}_final_tlsum_bunion_coverage.jsonl`, generated by "
        "`scripts/recompute_final_tlsum_bunion_coverage.py`.",
        "- Taxonomy integration and alignment assertions: "
        "`scripts/analyze_final_tlsum_failure_taxonomy_aligned.py`.",
        "",
        "The prior B-union coverage and prior percentages are not used.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
