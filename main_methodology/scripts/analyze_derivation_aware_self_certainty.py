#!/usr/bin/env python3
"""Derivation-aware self-certainty probe for QA.

This is a lightweight diagnostic experiment, not a replacement for the main
SEEK metric. It uses existing dual-path outputs that contain both a direct
answer and a decomposed reasoning trace.

The score is intentionally not answer-type rule based. For each direct answer,
we ask whether the answer looks more like an intermediate answer in the
decomposition trace or like the final decomposed answer.

Definitions:
  intermediate_sim = max token-F1(direct_pred, non-final step answers)
  final_sim        = max token-F1(direct_pred, final step / decomp pred)
  role_risk        = intermediate_sim - final_sim

If a direct answer is low entropy but role_risk is high, it is a candidate
"confident but stopped at the wrong derivation role" error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_FILES = {
    "2WikiMultiHopQA": "main_methodology/results/2wiki_1000_verify_v2_pathaware_directctx.jsonl",
    "HotpotQA": "main_methodology/results/hotpotqa_1000_verify_v2_pathaware_directctx.jsonl",
    "MuSiQue": "main_methodology/results/musique_1000_verify_v2_pathaware_directctx.jsonl",
    "PopQA": "main_methodology/results/popqa_1000_verify_v2_pathaware_directctx.jsonl",
}


def norm(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(a: Any, b: Any) -> float:
    a_toks = norm(a).split()
    b_toks = norm(b).split()
    if not a_toks or not b_toks:
        return 0.0
    common = Counter(a_toks) & Counter(b_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(a_toks)
    recall = overlap / len(b_toks)
    return 2 * precision * recall / (precision + recall)


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Mann-Whitney AUROC. Higher score should mean more likely label=1."""
    pairs = [(s, y) for s, y in zip(scores, labels) if math.isfinite(s)]
    n_pos = sum(y == 1 for _, y in pairs)
    n_neg = sum(y == 0 for _, y in pairs)
    if n_pos == 0 or n_neg == 0:
        return None

    pairs.sort(key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(y == 1 for _, y in pairs[i:j])
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def zscores(values: list[float]) -> list[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [0.0 for _ in values]
    mu = mean(finite)
    var = mean((v - mu) ** 2 for v in finite)
    sd = math.sqrt(var) or 1.0
    return [(v - mu) / sd if math.isfinite(v) else 0.0 for v in values]


def step_answers(step: dict[str, Any]) -> list[str]:
    vals = []
    for key in ("tentative", "final_answer", "answer", "pred"):
        val = step.get(key)
        if isinstance(val, str) and norm(val):
            vals.append(val)
    return vals


def best_match(pred: str, candidates: list[str]) -> tuple[float, str]:
    best_score = 0.0
    best_text = ""
    for cand in candidates:
        score = token_f1(pred, cand)
        if score > best_score:
            best_score = score
            best_text = cand
    return best_score, best_text


def analyze_file(dataset: str, path: Path) -> dict[str, Any]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            direct = row.get("direct") or {}
            decomp = row.get("decomp") or {}
            pred = direct.get("pred") or ""
            if not norm(pred):
                continue

            steps = decomp.get("steps") or row.get("steps") or []
            if not isinstance(steps, list) or not steps:
                continue

            final_candidates = []
            decomp_pred = decomp.get("pred") or row.get("decomp_pred") or ""
            if norm(decomp_pred):
                final_candidates.append(decomp_pred)
            final_candidates.extend(step_answers(steps[-1]))

            intermediate_candidates = []
            for step in steps[:-1]:
                intermediate_candidates.extend(step_answers(step))

            inter_sim, inter_text = best_match(pred, intermediate_candidates)
            final_sim, final_text = best_match(pred, final_candidates)
            role_risk = inter_sim - final_sim

            ent = direct.get("skip_probe_ent")
            try:
                direct_entropy = float(ent)
            except (TypeError, ValueError):
                direct_entropy = float("nan")

            direct_em = row.get("direct_em")
            if direct_em is None:
                direct_em = int(norm(pred) == norm(row.get("gold", "")))
            wrong = 1 - int(bool(direct_em))

            records.append(
                {
                    "dataset": dataset,
                    "id": row.get("id"),
                    "question": row.get("question", ""),
                    "gold": row.get("gold", ""),
                    "direct_pred": pred,
                    "decomp_pred": decomp_pred,
                    "direct_entropy": direct_entropy,
                    "wrong": wrong,
                    "intermediate_sim": inter_sim,
                    "intermediate_match": inter_text,
                    "final_sim": final_sim,
                    "final_match": final_text,
                    "role_risk": role_risk,
                }
            )

    ent_z = zscores([r["direct_entropy"] for r in records])
    role_z = zscores([r["role_risk"] for r in records])
    for r, ez, rz in zip(records, ent_z, role_z):
        r["dasc_risk"] = ez + rz

    labels = [r["wrong"] for r in records]
    low_entropy_cutoff = sorted(r["direct_entropy"] for r in records if math.isfinite(r["direct_entropy"]))[
        max(0, int(0.2 * len(records)) - 1)
    ]
    low_entropy = [r for r in records if r["direct_entropy"] <= low_entropy_cutoff]
    low_wrong = [r for r in low_entropy if r["wrong"]]
    flagged = [
        r
        for r in low_wrong
        if r["intermediate_sim"] >= 0.8 and r["final_sim"] < 0.5
    ]

    return {
        "dataset": dataset,
        "path": str(path),
        "records": records,
        "n": len(records),
        "direct_acc": 1 - mean(labels) if labels else float("nan"),
        "entropy_auroc": auroc([r["direct_entropy"] for r in records], labels),
        "role_auroc": auroc([r["role_risk"] for r in records], labels),
        "dasc_auroc": auroc([r["dasc_risk"] for r in records], labels),
        "low_entropy_cutoff": low_entropy_cutoff,
        "low_entropy_n": len(low_entropy),
        "low_entropy_wrong": len(low_wrong),
        "flagged_low_entropy_wrong": len(flagged),
        "examples": sorted(flagged, key=lambda r: (-r["role_risk"], r["direct_entropy"]))[:5],
    }


def fmt_auc(val: float | None) -> str:
    return "NA" if val is None else f"{val:.3f}"


def write_report(results: list[dict[str, Any]], out: Path) -> None:
    pooled_records = [r for result in results for r in result["records"]]
    labels = [r["wrong"] for r in pooled_records]
    pooled = {
        "n": len(pooled_records),
        "direct_acc": 1 - mean(labels) if labels else float("nan"),
        "entropy_auroc": auroc([r["direct_entropy"] for r in pooled_records], labels),
        "role_auroc": auroc([r["role_risk"] for r in pooled_records], labels),
        "dasc_auroc": auroc([r["dasc_risk"] for r in pooled_records], labels),
        "low_entropy_n": sum(r["low_entropy_n"] for r in results),
        "low_entropy_wrong": sum(r["low_entropy_wrong"] for r in results),
        "flagged_low_entropy_wrong": sum(r["flagged_low_entropy_wrong"] for r in results),
    }

    lines = [
        "# Derivation-Aware Self-Certainty Probe",
        "",
        "This is a small diagnostic experiment for a QA adaptation of self-certainty.",
        "It does not use hand-written answer-type rules. Instead, it checks whether",
        "a direct answer is closer to an intermediate answer in the decomposition",
        "trace or to the final decomposed answer.",
        "",
        "Score:",
        "",
        "```text",
        "role_risk = max_F1(direct_pred, intermediate step answers)",
        "            - max_F1(direct_pred, final step answer / decomp prediction)",
        "DASC risk = z(direct entropy) + z(role_risk)",
        "```",
        "",
        "Higher score means higher risk of a wrong direct answer. The low-entropy",
        "flagged column focuses on confident direct answers whose prediction looks",
        "like an intermediate derivation result.",
        "",
        "## Relation to Self-Certainty",
        "",
        "The original Self-Certainty paper proposes using model-internal certainty",
        "to rank sampled responses instead of relying only on majority voting:",
        "https://arxiv.org/abs/2502.18581.",
        "",
        "This probe keeps the same motivation but adapts it to QA. A QA answer",
        "should not only be high-certainty; it should be high-certainty as the",
        "final answer to the original question, not merely as an intermediate",
        "answer in the reasoning chain. Therefore, this first experiment adds",
        "a derivation-role margin on top of entropy/self-certainty.",
        "",
        "## Summary",
        "",
        "| Dataset | n | direct acc | AUROC entropy | AUROC role_risk | AUROC DASC | low-ent wrong | flagged low-ent wrong |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['n']} | {100*r['direct_acc']:.1f} | "
            f"{fmt_auc(r['entropy_auroc'])} | {fmt_auc(r['role_auroc'])} | "
            f"{fmt_auc(r['dasc_auroc'])} | {r['low_entropy_wrong']} / {r['low_entropy_n']} | "
            f"{r['flagged_low_entropy_wrong']} |"
        )
    lines.append(
        f"| ALL | {pooled['n']} | {100*pooled['direct_acc']:.1f} | "
        f"{fmt_auc(pooled['entropy_auroc'])} | {fmt_auc(pooled['role_auroc'])} | "
        f"{fmt_auc(pooled['dasc_auroc'])} | {pooled['low_entropy_wrong']} / {pooled['low_entropy_n']} | "
        f"{pooled['flagged_low_entropy_wrong']} |"
    )

    lines.extend(
        [
            "",
            "## Flagged Examples",
            "",
            "| Dataset | Question | Gold | Direct pred | Intermediate match | Decomp final | entropy | role_risk |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for result in results:
        for ex in result["examples"]:
            q = str(ex["question"]).replace("|", "\\|")
            gold = str(ex["gold"]).replace("|", "\\|")
            pred = str(ex["direct_pred"]).replace("|", "\\|")
            inter = str(ex["intermediate_match"]).replace("|", "\\|")
            final = str(ex["decomp_pred"] or ex["final_match"]).replace("|", "\\|")
            lines.append(
                f"| {ex['dataset']} | {q} | {gold} | {pred} | {inter} | {final} | "
                f"{ex['direct_entropy']:.3f} | {ex['role_risk']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `role_risk` is a derivation-role signal, not an answer-type rule.",
            "- It catches cases where the model confidently outputs a bridge or evidence value.",
            "- It will miss alias/semantic cases where lexical F1 cannot connect related forms,",
            "  such as `American` vs `United States`.",
            "- A full QA self-certainty version should replace token-F1 matching with",
            "  model-scored certainty of candidate answers under the original question and",
            "  each decomposition step.",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="main_methodology/results/derivation_aware_self_certainty_probe.md")
    ap.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="DATASET:PATH",
        help="Override input files. Can be passed multiple times.",
    )
    args = ap.parse_args()

    files = DEFAULT_FILES.copy()
    for spec in args.file:
        dataset, path = spec.split(":", 1)
        files[dataset] = path

    results = []
    for dataset, rel_path in files.items():
        path = Path(rel_path)
        if not path.exists():
            raise FileNotFoundError(path)
        results.append(analyze_file(dataset, path))

    out = Path(args.out)
    write_report(results, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
