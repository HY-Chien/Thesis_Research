#!/usr/bin/env python3
"""Model-scored derivation-aware self-certainty probe.

This is the next step after analyze_derivation_aware_self_certainty.py.
Instead of using token-F1 to decide whether a direct answer looks like an
intermediate answer, this script scores the same candidate answer under
different QA prompts with the model's own output distribution.

For a fixed candidate answer y:
  self_certainty(y | q) = mean_i KL(U || p_model(. | q, y_<i))

Higher values mean the model distribution is sharper while emitting y.
The QA role risk is:
  sc_role_risk = max self_certainty(y | intermediate sub-question)
                 - max self_certainty(y | original/final question)

If the score is high, the direct answer is more "self-certain" as an
intermediate derivation answer than as the final answer.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.retriever import PerQuestionBM25, format_context  # noqa: E402

BASE = "/workspace/S113065528/RAG-DDR/models/Llama3-8B-Instruct"
DEFAULT_ADAPTER = "main_methodology/checkpoints/llama3-8b-decomposer-musique-2wiki-popqa"
ANSWER_INSTRUCTION = (
    "Answer the question with the answer phrase only. "
    "No explanation, no preamble, no full sentence -- just the answer."
)

DEFAULT_FILES = {
    "2WikiMultiHopQA": "main_methodology/results/2wiki_1000_verify_v2_pathaware_directctx.jsonl",
    "HotpotQA": "main_methodology/results/hotpotqa_1000_verify_v2_pathaware_directctx.jsonl",
    "MuSiQue": "main_methodology/results/musique_1000_verify_v2_pathaware_directctx.jsonl",
    "PopQA": "main_methodology/results/popqa_1000_verify_v2_pathaware_directctx.jsonl",
}

DEV_FILES = {
    "2WikiMultiHopQA": "main_methodology/datasets/2wiki/raw/dev_first1000.jsonl",
    "HotpotQA": "main_methodology/datasets/hotpotqa/raw/dev_first1000.jsonl",
    "MuSiQue": "main_methodology/datasets/musique/raw/dev_first1000.jsonl",
    "PopQA": "main_methodology/datasets/popqa/raw/dev.jsonl",
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


def fmt_auc(val: float | None) -> str:
    return "NA" if val is None else f"{val:.3f}"


def score_prompt(tok: Any, question: str, context: str | None = None) -> str:
    body = question
    if context:
        body = f"Context:\n{context}\n\nQuestion: {question}"
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def candidate_scores(
    model: Any, tok: Any, question: str, answer: str, context: str | None = None
) -> dict[str, float]:
    """Return model-internal scores for fixed candidate answer under a question."""
    prompt = score_prompt(tok, question, context=context)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full_ids = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_len:
        return {"self_certainty": 0.0, "avg_logprob": float("-inf"), "mean_entropy": 0.0}

    out = model(full_ids)
    logits = out.logits[:, prompt_len - 1 : full_ids.shape[1] - 1, :]
    target = full_ids[:, prompt_len:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()

    vocab = log_probs.shape[-1]
    self_certainty = -((log_probs + math.log(vocab)).mean(dim=-1)).mean().item()
    avg_logprob = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean().item()
    mean_entropy = torch.special.entr(probs).sum(dim=-1).mean().item()
    return {
        "self_certainty": self_certainty,
        "avg_logprob": avg_logprob,
        "mean_entropy": mean_entropy,
    }


def load_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            direct = row.get("direct") or {}
            decomp = row.get("decomp") or {}
            pred = direct.get("pred") or ""
            ent = direct.get("skip_probe_ent")
            try:
                direct_entropy = float(ent)
            except (TypeError, ValueError):
                continue
            steps = decomp.get("steps") or []
            if not norm(pred) or not isinstance(steps, list) or len(steps) < 2:
                continue
            direct_em = row.get("direct_em")
            if direct_em is None:
                direct_em = int(norm(pred) == norm(row.get("gold", "")))
            rows.append(
                {
                    "dataset": dataset,
                    "id": row.get("id"),
                    "question": row.get("question", ""),
                    "gold": row.get("gold", ""),
                    "direct_pred": pred,
                    "direct_entropy": direct_entropy,
                    "wrong": 1 - int(bool(direct_em)),
                    "decomp_pred": decomp.get("pred") or "",
                    "steps": steps,
                }
            )
    return rows


def attach_paragraphs(rows: list[dict[str, Any]], dataset: str) -> None:
    dev_path = Path(DEV_FILES[dataset])
    by_id = {}
    by_question = {}
    with dev_path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            if raw.get("id") is not None:
                by_id[str(raw.get("id"))] = raw
            by_question[raw.get("question", "")] = raw
    for row in rows:
        raw = by_id.get(str(row.get("id"))) or by_question.get(row.get("question", ""))
        row["paragraphs"] = (raw or {}).get("paragraphs") or []


def select_rows(rows: list[dict[str, Any]], per_dataset: int) -> list[dict[str, Any]]:
    """Balanced low-entropy sample: half wrong, half correct when possible."""
    wrong = sorted([r for r in rows if r["wrong"]], key=lambda r: r["direct_entropy"])
    correct = sorted([r for r in rows if not r["wrong"]], key=lambda r: r["direct_entropy"])
    half = per_dataset // 2
    selected = wrong[:half] + correct[: per_dataset - min(half, len(wrong))]
    if len(selected) < per_dataset:
        chosen = {id(r) for r in selected}
        filler = [r for r in sorted(rows, key=lambda r: r["direct_entropy"]) if id(r) not in chosen]
        selected.extend(filler[: per_dataset - len(selected)])
    return sorted(selected, key=lambda r: (r["dataset"], r["direct_entropy"]))


def step_questions(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    steps = row["steps"]
    intermediate = []
    for step in steps[:-1]:
        q = step.get("sub_q") or step.get("raw_sub_q")
        if q:
            intermediate.append(str(q))

    final = [row["question"]]
    last_q = steps[-1].get("sub_q") or steps[-1].get("raw_sub_q")
    if last_q:
        final.append(str(last_q))
    return intermediate, final


def f1_role_risk(row: dict[str, Any]) -> float:
    pred = row["direct_pred"]
    inter_answers = []
    for step in row["steps"][:-1]:
        for key in ("tentative", "final_answer", "answer", "pred"):
            val = step.get(key)
            if isinstance(val, str) and norm(val):
                inter_answers.append(val)
    final_answers = [row["decomp_pred"]]
    for key in ("tentative", "final_answer", "answer", "pred"):
        val = row["steps"][-1].get(key)
        if isinstance(val, str) and norm(val):
            final_answers.append(val)
    inter = max([token_f1(pred, a) for a in inter_answers] or [0.0])
    final = max([token_f1(pred, a) for a in final_answers] or [0.0])
    return inter - final


def context_for(row: dict[str, Any], question: str, top_k: int) -> str | None:
    paragraphs = row.get("paragraphs") or []
    if not paragraphs:
        return None
    retr = PerQuestionBM25(paragraphs)
    return format_context(retr.topk(question, k=top_k))


def run_probe(
    model: Any, tok: Any, rows: list[dict[str, Any]], with_context: bool, top_k: int
) -> list[dict[str, Any]]:
    cache: dict[tuple[str, str, str], dict[str, float]] = {}
    scored = []
    for idx, row in enumerate(rows, 1):
        pred = row["direct_pred"]
        inter_qs, final_qs = step_questions(row)
        all_qs = [("intermediate", q) for q in inter_qs] + [("final", q) for q in final_qs]

        inter_scores = []
        final_scores = []
        inter_lp_scores = []
        final_lp_scores = []
        for role, q in all_qs:
            ctx = context_for(row, q, top_k) if with_context else None
            key = (q, pred, ctx or "")
            if key not in cache:
                cache[key] = candidate_scores(model, tok, q, pred, context=ctx)
            score = cache[key]["self_certainty"]
            lp_score = cache[key]["avg_logprob"]
            if role == "intermediate":
                inter_scores.append((score, q))
                inter_lp_scores.append((lp_score, q))
            else:
                final_scores.append((score, q))
                final_lp_scores.append((lp_score, q))

        best_inter, best_inter_q = max(inter_scores, default=(0.0, ""))
        best_final, best_final_q = max(final_scores, default=(0.0, ""))
        best_inter_lp, best_inter_lp_q = max(inter_lp_scores, default=(float("-inf"), ""))
        best_final_lp, best_final_lp_q = max(final_lp_scores, default=(float("-inf"), ""))
        out = dict(row)
        out.update(
            {
                "best_inter_sc": best_inter,
                "best_inter_question": best_inter_q,
                "best_final_sc": best_final,
                "best_final_question": best_final_q,
                "sc_role_risk": best_inter - best_final,
                "best_inter_logprob": best_inter_lp,
                "best_inter_logprob_question": best_inter_lp_q,
                "best_final_logprob": best_final_lp,
                "best_final_logprob_question": best_final_lp_q,
                "logprob_role_risk": best_inter_lp - best_final_lp,
                "f1_role_risk": f1_role_risk(row),
            }
        )
        scored.append(out)
        if idx % 20 == 0:
            print(f"  scored {idx}/{len(rows)}", flush=True)
    return scored


def add_combined_scores(rows: list[dict[str, Any]]) -> None:
    ent_z = zscores([r["direct_entropy"] for r in rows])
    sc_z = zscores([r["sc_role_risk"] for r in rows])
    lp_z = zscores([r["logprob_role_risk"] for r in rows])
    f1_z = zscores([r["f1_role_risk"] for r in rows])
    for r, ez, sz, lz, fz in zip(rows, ent_z, sc_z, lp_z, f1_z):
        r["model_dasc_risk"] = ez + sz
        r["logprob_dasc_risk"] = ez + lz
        r["f1_dasc_risk"] = ez + fz


def write_report(
    rows: list[dict[str, Any]], out: Path, per_dataset: int, with_context: bool
) -> None:
    add_combined_scores(rows)

    lines = [
        "# Model-Scored DASC Probe",
        "",
        "This is a small QA adaptation of Self-Certainty on low-entropy direct-answer cases.",
        "Unlike the first DASC-lite probe, this version does not use token-F1 to define the",
        "derivation-role score. It teacher-forces the direct prediction under the original",
        "question, final sub-question, and intermediate sub-questions, then measures how",
        "sharp the model's token distribution is while emitting the same candidate answer.",
        "",
        "```text",
        "self_certainty(y | q) = mean_i KL(U || p_model(. | q, y_<i))",
        "sc_role_risk = max self_certainty(y | intermediate sub-question)",
        "               - max self_certainty(y | original/final question)",
        "logprob_role_risk = max log p(y | intermediate sub-question)",
        "                    - max log p(y | original/final question)",
        "model_DASC = z(direct_entropy) + z(sc_role_risk)",
        "logprob_DASC = z(direct_entropy) + z(logprob_role_risk)",
        "```",
        "",
        f"Selection: up to {per_dataset} low-entropy direct cases per dataset, balanced",
        "between official direct-wrong and direct-correct cases when possible.",
        f"Context-aware scoring: {with_context}.",
        "",
        "## Summary",
        "",
        "| Dataset | n | wrong | AUROC entropy | AUROC F1-role | AUROC SC-role | AUROC logprob-role | AUROC F1-DASC | AUROC SC-DASC | AUROC logprob-DASC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    datasets = sorted({r["dataset"] for r in rows})
    pooled_labels = [r["wrong"] for r in rows]
    for dataset in datasets:
        subset = [r for r in rows if r["dataset"] == dataset]
        labels = [r["wrong"] for r in subset]
        lines.append(
            f"| {dataset} | {len(subset)} | {sum(labels)} | "
            f"{fmt_auc(auroc([r['direct_entropy'] for r in subset], labels))} | "
            f"{fmt_auc(auroc([r['f1_role_risk'] for r in subset], labels))} | "
            f"{fmt_auc(auroc([r['sc_role_risk'] for r in subset], labels))} | "
            f"{fmt_auc(auroc([r['logprob_role_risk'] for r in subset], labels))} | "
            f"{fmt_auc(auroc([r['f1_dasc_risk'] for r in subset], labels))} | "
            f"{fmt_auc(auroc([r['model_dasc_risk'] for r in subset], labels))} | "
            f"{fmt_auc(auroc([r['logprob_dasc_risk'] for r in subset], labels))} |"
        )
    lines.append(
        f"| ALL | {len(rows)} | {sum(pooled_labels)} | "
        f"{fmt_auc(auroc([r['direct_entropy'] for r in rows], pooled_labels))} | "
        f"{fmt_auc(auroc([r['f1_role_risk'] for r in rows], pooled_labels))} | "
        f"{fmt_auc(auroc([r['sc_role_risk'] for r in rows], pooled_labels))} | "
        f"{fmt_auc(auroc([r['logprob_role_risk'] for r in rows], pooled_labels))} | "
        f"{fmt_auc(auroc([r['f1_dasc_risk'] for r in rows], pooled_labels))} | "
        f"{fmt_auc(auroc([r['model_dasc_risk'] for r in rows], pooled_labels))} | "
        f"{fmt_auc(auroc([r['logprob_dasc_risk'] for r in rows], pooled_labels))} |"
    )

    top = sorted([r for r in rows if r["wrong"]], key=lambda r: -r["sc_role_risk"])[:16]
    lines.extend(
        [
            "",
            "## Top Self-Certainty Role-Risk Wrong Examples",
            "",
            "| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | sc_role_risk |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for r in top:
        def clean(x: Any) -> str:
            return str(x).replace("|", "\\|")

        lines.append(
            f"| {r['dataset']} | {clean(r['question'])} | {clean(r['gold'])} | "
            f"{clean(r['direct_pred'])} | {clean(r['best_inter_question'])} | "
            f"{clean(r['best_final_question'])} | {r['direct_entropy']:.3f} | "
            f"{r['sc_role_risk']:.3f} |"
        )

    top_lp = sorted([r for r in rows if r["wrong"]], key=lambda r: -r["logprob_role_risk"])[:16]
    lines.extend(
        [
            "",
            "## Top Logprob Role-Risk Wrong Examples",
            "",
            "| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | logprob_role_risk |",
            "|---|---|---|---|---|---|---:|---:|",
        ]
    )
    for r in top_lp:
        def clean(x: Any) -> str:
            return str(x).replace("|", "\\|")

        lines.append(
            f"| {r['dataset']} | {clean(r['question'])} | {clean(r['gold'])} | "
            f"{clean(r['direct_pred'])} | {clean(r['best_inter_logprob_question'])} | "
            f"{clean(r['best_final_logprob_question'])} | {r['direct_entropy']:.3f} | "
            f"{r['logprob_role_risk']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is still a small diagnostic run, not the final metric.",
            "- The model-scored role signal uses no gold answer, but it depends on the",
            "  quality of the decomposition trace.",
            "- Pure self-certainty measures distribution sharpness, not necessarily answer",
            "  compatibility. The log-probability role score is included as a QA-specific",
            "  companion signal.",
            "- Some comparison or yes/no questions may still need an explicit final-answer",
            "  synthesis score, because the final answer is not always a sub-question answer.",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--per-dataset", type=int, default=40)
    ap.add_argument("--out", default="main_methodology/results/model_scored_dasc_probe.md")
    ap.add_argument("--file", action="append", default=[], metavar="DATASET:PATH")
    ap.add_argument("--with-context", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    files = DEFAULT_FILES.copy()
    for spec in args.file:
        dataset, path = spec.split(":", 1)
        files[dataset] = path

    rows = []
    for dataset, rel_path in files.items():
        loaded = load_rows(Path(rel_path), dataset)
        if args.with_context:
            attach_paragraphs(loaded, dataset)
        rows.extend(select_rows(loaded, args.per_dataset))
        print(f"selected {min(args.per_dataset, len(loaded))} from {dataset}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    scored = run_probe(model, tok, rows, with_context=args.with_context, top_k=args.top_k)
    write_report(scored, Path(args.out), args.per_dataset, with_context=args.with_context)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
