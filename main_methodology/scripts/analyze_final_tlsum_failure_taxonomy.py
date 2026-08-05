"""Failure analysis for the final all-tl_sum prompt-fix-A traces.

This analysis deliberately does not reuse the legacy decomposed-query
``B-union`` coverage labels.  Those labels were generated from an earlier
set of resolved sub-questions.  Prompt-fix A can change an intermediate
answer and therefore the substituted text of every later sub-question.

The report contains only quantities that can be reproduced exactly from the
final traces, plus original-question (A) full-coverage@10.  A retrieval is
independent of decomposition and remains aligned because the original
question and corpus query are unchanged.

Output: results/decomp_failure_taxonomy_final_tlsum.md
"""

from __future__ import annotations

import collections
import json
import re
import string
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "decomp_failure_taxonomy_final_tlsum.md"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_verifier_tlsum_fixA import build, decide  # noqa: E402


DATASETS = ["hotpotqa", "2wiki", "musique"]
TAU_DIRECT = 0.0005
TAU_POST = 0.0158

STOP = set(
    "what which who whom whose where when why how is are was were did does do "
    "the a an of in on at by for from to with and or as that this it his her "
    "its their there be been s".split()
)


def norm(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def f1(pred: str, gold: str) -> float:
    pred_toks = norm(pred).split()
    gold_toks = norm(gold).split()
    overlap = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    n_common = sum(overlap.values())
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def matches_any_gold(answer: str, golds: list[str]) -> bool:
    """Retain the legacy taxonomy's exact-or-F1>=.5 step-match rule.

    Unlike the legacy report, test all aliases used by final evaluation.
    """
    return any(norm(answer) == norm(g) or f1(answer, g) >= 0.5 for g in golds)


def content_tokens(text: str) -> list[str]:
    return [tok for tok in norm(text).split() if tok not in STOP]


def subq1_restates_question(question: str, steps: list[dict]) -> bool:
    if len(steps) < 2:
        return False
    q_toks = content_tokens(question)
    if not q_toks:
        return False
    subq_toks = set(content_tokens(steps[0].get("sub_q") or
                                   steps[0].get("raw_sub_q") or ""))
    recall = sum(tok in subq_toks for tok in q_toks) / len(q_toks)
    return recall >= 0.6


def intermediate_echo(row: dict) -> bool:
    """Final synthesis equals an earlier, non-final step answer."""
    answers = [step.get("final_answer") or "" for step in row["steps"]]
    if len(answers) < 2:
        return False
    pred = norm(row["decomp_pred"])
    final_step = norm(answers[-1])
    return any(pred == norm(ans) and norm(ans) != final_step
               for ans in answers[:-1] if ans)


def load_original_coverage(ds: str) -> dict[str, int]:
    """Load only A full@10; B-union is intentionally not consumed."""
    path = RESULTS / f"{ds}_m3_wiki_detail.jsonl"
    out: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[str(rec["id"])] = int(rec["cov"]["10"]["A"]["full"])
    return out


def pct(count: int, total: int, digits: int = 1) -> str:
    return f"{count} ({100 * count / total:.{digits}f}%)" if total else "0 (-)"


def main() -> None:
    lines = [
        "# Final tl_sum decomposition-path failure analysis",
        "",
        "Configuration: prompt-fix A; step retrieval `tl_sum >= 0.008461`; "
        "direct verifier `tl_sum` at `tau_d=0.0005`; post-decomposition "
        "verifier `tl_sum` at `tau_p=0.0158`.",
        "",
        "A step-level synthesis opportunity uses the pre-existing taxonomy "
        "rule: at least one final step answer has exact match or token F1 >= "
        "0.5 against any gold answer or alias. Categories and flags below are "
        "computed from the final traces.",
        "",
    ]

    cross_rows: dict[str, dict] = {}
    for ds in DATASETS:
        rows = build(ds)
        a_cov = load_original_coverage(ds)
        groups = {key: [] for key in ("both_correct", "direct_only",
                                      "decomp_only", "both_wrong")}
        final_wrong = []
        for row in rows:
            key = {
                (1, 1): "both_correct",
                (1, 0): "direct_only",
                (0, 1): "decomp_only",
                (0, 0): "both_wrong",
            }[(row["direct_em"], row["decomp_em"])]
            groups[key].append(row)
            choice = decide(row, "both", TAU_DIRECT, TAU_POST)
            row["final_choice_tlsum"] = choice
            final_em = row["direct_em"] if choice == "direct" else row["decomp_em"]
            if not final_em:
                final_wrong.append(row)

        decomp_wrong = groups["direct_only"] + groups["both_wrong"]
        synth = [r for r in decomp_wrong
                 if any(matches_any_gold(step.get("final_answer") or "", r["golds"])
                        for step in r["steps"])]
        no_step_match = [r for r in decomp_wrong if r not in synth]
        flags = {
            "intermediate_echo": sum(intermediate_echo(r) for r in decomp_wrong),
            "subq1_restates_question": sum(
                subq1_restates_question(r["question"], r["steps"])
                for r in decomp_wrong
            ),
            "zero_retrieved_steps": sum(
                bool(r["steps"]) and not any(s.get("retrieved") for s in r["steps"])
                for r in decomp_wrong
            ),
            "all_steps_retrieved": sum(
                bool(r["steps"]) and all(s.get("retrieved") for s in r["steps"])
                for r in decomp_wrong
            ),
            "no_decomposition_steps": sum(not r["steps"] for r in decomp_wrong),
        }
        flags["mixed_retrieval"] = (
            len(decomp_wrong) - flags["zero_retrieved_steps"]
            - flags["all_steps_retrieved"] - flags["no_decomposition_steps"]
        )

        joined = [r for r in decomp_wrong if r["id"] in a_cov]
        a_full = sum(a_cov[r["id"]] for r in joined)
        dir_capture = sum(r["final_choice_tlsum"] == "direct"
                          for r in groups["direct_only"])
        dec_capture = sum(r["final_choice_tlsum"] == "decomp"
                          for r in groups["decomp_only"])
        router_miss_direct = len(groups["direct_only"]) - dir_capture
        router_miss_decomp = len(groups["decomp_only"]) - dec_capture

        lines += [
            f"## {ds} (n={len(rows)})",
            "",
            "| candidate correctness group | count | share |",
            "|---|---:|---:|",
        ]
        for key in ("both_correct", "direct_only", "decomp_only", "both_wrong"):
            lines.append(f"| {key} | {len(groups[key])} | "
                         f"{100 * len(groups[key]) / len(rows):.1f}% |")
        lines += [
            "",
            f"Final all-tl_sum router: direct-only capture "
            f"{pct(dir_capture, len(groups['direct_only']))}; decomp-only "
            f"capture {pct(dec_capture, len(groups['decomp_only']))}; final "
            f"wrong {len(final_wrong)}/{len(rows)} "
            f"({100 * len(final_wrong) / len(rows):.1f}%).",
            "",
            "| source among final wrong answers | count | share |",
            "|---|---:|---:|",
            f"| both candidates wrong | {len(groups['both_wrong'])} | "
            f"{100 * len(groups['both_wrong']) / len(final_wrong):.1f}% |",
            f"| router misses a direct-only win | {router_miss_direct} | "
            f"{100 * router_miss_direct / len(final_wrong):.1f}% |",
            f"| router misses a decomp-only win | {router_miss_decomp} | "
            f"{100 * router_miss_decomp / len(final_wrong):.1f}% |",
            "",
            f"Decomposition candidate errors: n={len(decomp_wrong)}.",
            "",
            "| exact final-trace category | count | share of decomp errors |",
            "|---|---:|---:|",
            f"| synthesis opportunity: a step answer matches gold/alias | "
            f"{len(synth)} | {100 * len(synth) / len(decomp_wrong):.1f}% |",
            f"| no step answer matches gold/alias | {len(no_step_match)} | "
            f"{100 * len(no_step_match) / len(decomp_wrong):.1f}% |",
            "",
            "| orthogonal final-trace flag | count | share of decomp errors |",
            "|---|---:|---:|",
        ]
        for flag in ("intermediate_echo", "subq1_restates_question",
                     "zero_retrieved_steps", "mixed_retrieval",
                     "all_steps_retrieved", "no_decomposition_steps"):
            lines.append(f"| {flag} | {flags[flag]} | "
                         f"{100 * flags[flag] / len(decomp_wrong):.1f}% |")
        lines += [
            "",
            "Original-query retrieval coverage (the reusable A arm only): "
            f"full supporting-title coverage@10 in {a_full}/{len(joined)} "
            f"({100 * a_full / len(joined):.1f}%) decomposition errors.",
            "",
        ]
        cross_rows[ds] = {
            "n": len(decomp_wrong), "synth": len(synth),
            "no_step": len(no_step_match), "a_full": a_full,
            "a_join": len(joined), **flags,
        }

    lines += [
        "## Cross-dataset table",
        "",
        "| exact final-trace category | HotpotQA | 2Wiki | MuSiQue |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("synth", "Step answer matches gold/alias, synthesis wrong"),
        ("no_step", "No step answer matches gold/alias"),
        ("intermediate_echo", "Flag: intermediate-answer echo"),
        ("subq1_restates_question", "Flag: sub-q1 restates question"),
        ("zero_retrieved_steps", "Gate state: zero steps retrieved"),
        ("mixed_retrieval", "Gate state: mixed"),
        ("all_steps_retrieved", "Gate state: all steps retrieved"),
    ):
        cells = [pct(cross_rows[ds][key], cross_rows[ds]["n"]) for ds in DATASETS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| Original-query A full coverage@10 | " + " | ".join(
        pct(cross_rows[ds]["a_full"], cross_rows[ds]["a_join"])
        for ds in DATASETS
    ) + " |")

    lines += [
        "",
        "## What cannot be retained from the legacy taxonomy",
        "",
        "The legacy labels `retrieval_gap_both`, `answer_despite_cov`, and "
        "`constraint_loss` require full-coverage@10 for the decomposed "
        "B-union queries. The saved detail files contain only coverage bits, "
        "not the query strings, and were generated before the final "
        "prompt-fix-A chain. Because later resolved sub-questions substitute "
        "earlier final answers, they can differ in the final trace. Those "
        "three labels therefore must not be copied into the final tl_sum "
        "table. They require a new B-union retrieval run over the final "
        "resolved sub-questions.",
        "",
        "Original-query A coverage is retained because the query is the "
        "unchanged original question and is independent of the decomposition "
        "answers and confidence metric.",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
