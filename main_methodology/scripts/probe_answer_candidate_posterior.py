#!/usr/bin/env python3
"""Probe answer-candidate posterior as a direct/decompose verifier signal.

This script tests a model-based alternative to token entropy. Instead of asking
how uncertain the model was while generating one answer, it teacher-forces the
saved direct and decomposed candidate answers under the same original question
prompt and compares their length-normalized log probabilities.

At inference time this score needs only:
  - the original question
  - optional retrieved context
  - the saved direct candidate answer
  - the saved decomposed candidate answer

Gold labels are used only here to evaluate whether the score would have helped
the final routing decision.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

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
    "2WikiMultiHopQA": "main_methodology/results/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "HotpotQA": "main_methodology/results/hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "MuSiQue": "main_methodology/results/musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    "PopQA": "main_methodology/results/popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
}

DEV_FILES = {
    "2WikiMultiHopQA": "main_methodology/datasets/2wiki/raw/dev_first1000.jsonl",
    "HotpotQA": "main_methodology/datasets/hotpotqa/raw/dev_first1000.jsonl",
    "MuSiQue": "main_methodology/datasets/musique/raw/dev_first1000.jsonl",
    "PopQA": "main_methodology/datasets/popqa/raw/dev.jsonl",
}

ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def token_f1(pred: Any, gold: Any) -> float:
    pred_toks = normalize(pred).split()
    gold_toks = normalize(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_toks)
    recall = overlap / len(gold_toks)
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


def fmt_auc(val: float | None) -> str:
    return "NA" if val is None else f"{val:.3f}"


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def safe_float(value: Any, default: float = math.inf) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def current_choice(row: dict[str, Any]) -> str:
    return row.get("replay_choice") or row.get("choice") or "decomp"


def current_em(row: dict[str, Any]) -> int:
    if "replay_em" in row:
        return int(row.get("replay_em") or 0)
    return int(row.get("em") or 0)


def direct_entropy(row: dict[str, Any]) -> float:
    return safe_float((row.get("direct") or {}).get("skip_probe_ent"))


def direct_pred(row: dict[str, Any]) -> str:
    return row.get("direct_pred") or (row.get("direct") or {}).get("pred") or ""


def decomp_pred(row: dict[str, Any]) -> str:
    return row.get("decomp_pred") or (row.get("decomp") or {}).get("pred") or ""


def choose_pred(row: dict[str, Any], choice: str) -> str:
    return direct_pred(row) if choice == "direct" else decomp_pred(row)


def choose_em(row: dict[str, Any], choice: str) -> int:
    key = "direct_em" if choice == "direct" else "decomp_em"
    return int(row.get(key) or 0)


def score_prompt(tok: Any, question: str, context: str | None = None) -> str:
    body = question
    if context:
        body = f"Context:\n{context}\n\nQuestion: {question}"
    msgs = [{"role": "user", "content": f"{ANSWER_INSTRUCTION}\n\n{body}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def candidate_scores(
    model: Any,
    tok: Any,
    question: str,
    answer: str,
    context: str | None = None,
) -> dict[str, float]:
    prompt = score_prompt(tok, question, context=context)
    prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    full_ids = tok(prompt + answer, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    answer_len = full_ids.shape[1] - prompt_len
    if answer_len <= 0:
        return {
            "avg_logprob": float("-inf"),
            "sum_logprob": float("-inf"),
            "self_certainty": 0.0,
            "mean_entropy": 0.0,
            "answer_tokens": 0,
        }

    out = model(full_ids)
    logits = out.logits[:, prompt_len - 1 : full_ids.shape[1] - 1, :]
    target = full_ids[:, prompt_len:]
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    vocab = log_probs.shape[-1]
    self_certainty = -((log_probs + math.log(vocab)).mean(dim=-1)).mean().item()
    mean_entropy = torch.special.entr(probs).sum(dim=-1).mean().item()
    sum_lp = token_lp.sum().item()
    return {
        "avg_logprob": token_lp.mean().item(),
        "sum_logprob": sum_lp,
        "self_certainty": self_certainty,
        "mean_entropy": mean_entropy,
        "answer_tokens": int(answer_len),
    }


def attach_paragraphs(rows: list[dict[str, Any]], dataset: str) -> None:
    dev_path = Path(DEV_FILES[dataset])
    by_id: dict[str, dict[str, Any]] = {}
    by_question: dict[str, dict[str, Any]] = {}
    with dev_path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            if raw.get("id") is not None:
                by_id[str(raw.get("id"))] = raw
            by_question[raw.get("question", "")] = raw
    for row in rows:
        raw = by_id.get(str(row.get("id"))) or by_question.get(row.get("question", ""))
        row["paragraphs"] = (raw or {}).get("paragraphs") or []


def original_context(row: dict[str, Any], top_k: int) -> str | None:
    paragraphs = row.get("paragraphs") or []
    if not paragraphs:
        return None
    retriever = PerQuestionBM25(paragraphs)
    return format_context(retriever.topk(row.get("question") or "", k=top_k))


def select_rows(rows: list[dict[str, Any]], per_dataset: int, mode: str) -> list[dict[str, Any]]:
    if per_dataset <= 0 or len(rows) <= per_dataset:
        return rows
    if mode == "first":
        return rows[:per_dataset]
    if mode == "low_entropy":
        return sorted(rows, key=direct_entropy)[:per_dataset]
    if mode == "routing":
        direct_only = [r for r in rows if int(r.get("direct_em") or 0) and not int(r.get("decomp_em") or 0)]
        decomp_only = [r for r in rows if int(r.get("decomp_em") or 0) and not int(r.get("direct_em") or 0)]
        both_correct = [r for r in rows if int(r.get("direct_em") or 0) and int(r.get("decomp_em") or 0)]
        both_wrong = [r for r in rows if not int(r.get("direct_em") or 0) and not int(r.get("decomp_em") or 0)]
        buckets = [direct_only, decomp_only, both_correct, both_wrong]
        out: list[dict[str, Any]] = []
        quota = max(1, per_dataset // len(buckets))
        for bucket in buckets:
            out.extend(bucket[:quota])
        seen = {id(r) for r in out}
        for row in rows:
            if len(out) >= per_dataset:
                break
            if id(row) not in seen:
                out.append(row)
        return out[:per_dataset]
    raise ValueError(f"unknown sample mode: {mode}")


def load_rows(files: dict[str, str], per_dataset: int, sample_mode: str, with_context: bool) -> dict[str, list[dict[str, Any]]]:
    rows_by_dataset = {}
    for dataset, rel_path in files.items():
        rows = load_jsonl(Path(rel_path))
        rows = [
            row for row in rows
            if normalize(direct_pred(row)) and normalize(decomp_pred(row))
        ]
        if with_context:
            attach_paragraphs(rows, dataset)
        rows_by_dataset[dataset] = select_rows(rows, per_dataset, sample_mode)
        print(f"selected {len(rows_by_dataset[dataset])}/{len(rows)} from {dataset}", flush=True)
    return rows_by_dataset


def score_rows(
    model: Any,
    tok: Any,
    rows_by_dataset: dict[str, list[dict[str, Any]]],
    *,
    with_context: bool,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    cache: dict[tuple[str, str, str], dict[str, float]] = {}
    scored_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for dataset, rows in rows_by_dataset.items():
        scored = []
        for idx, row in enumerate(rows, 1):
            question = row.get("question") or ""
            ctx = original_context(row, top_k) if with_context else None
            d_answer = direct_pred(row)
            c_answer = decomp_pred(row)
            d_key = (question, d_answer, ctx or "")
            c_key = (question, c_answer, ctx or "")
            if d_key not in cache:
                cache[d_key] = candidate_scores(model, tok, question, d_answer, context=ctx)
            if c_key not in cache:
                cache[c_key] = candidate_scores(model, tok, question, c_answer, context=ctx)

            d_score = cache[d_key]
            c_score = cache[c_key]
            avg_margin = c_score["avg_logprob"] - d_score["avg_logprob"]
            sum_margin = c_score["sum_logprob"] - d_score["sum_logprob"]
            sc_margin = c_score["self_certainty"] - d_score["self_certainty"]
            p_decomp = sigmoid(avg_margin)
            out = dict(row)
            out.update(
                {
                    "dataset": dataset,
                    "acp_direct_avg_logprob": d_score["avg_logprob"],
                    "acp_decomp_avg_logprob": c_score["avg_logprob"],
                    "acp_avg_margin": avg_margin,
                    "acp_sum_margin": sum_margin,
                    "acp_sc_margin": sc_margin,
                    "acp_p_decomp": p_decomp,
                    "acp_direct_tokens": d_score["answer_tokens"],
                    "acp_decomp_tokens": c_score["answer_tokens"],
                }
            )
            scored.append(out)
            if idx % 25 == 0:
                print(f"  {dataset}: scored {idx}/{len(rows)}", flush=True)
        scored_by_dataset[dataset] = scored
    return scored_by_dataset


@dataclass(frozen=True)
class EvalResult:
    em: float
    f1: float
    direct_rate: float
    changed: int
    rescued: int
    lost: int


def evaluate(rows: list[dict[str, Any]], chooser: Callable[[dict[str, Any]], str]) -> EvalResult:
    em_sum = 0
    f1_sum = 0.0
    direct_count = 0
    changed = 0
    rescued = 0
    lost = 0
    for row in rows:
        choice = chooser(row)
        em = choose_em(row, choice)
        base = current_em(row)
        em_sum += em
        f1_sum += token_f1(choose_pred(row, choice), row.get("gold") or "")
        direct_count += choice == "direct"
        changed += choice != current_choice(row)
        rescued += int(not base and em)
        lost += int(base and not em)
    n = len(rows) or 1
    return EvalResult(
        em=100 * em_sum / n,
        f1=100 * f1_sum / n,
        direct_rate=100 * direct_count / n,
        changed=changed,
        rescued=rescued,
        lost=lost,
    )


def current_policy(row: dict[str, Any]) -> str:
    return current_choice(row)


def acp_argmax(row: dict[str, Any]) -> str:
    if normalize(direct_pred(row)) == normalize(decomp_pred(row)):
        return current_choice(row)
    return "decomp" if row["acp_avg_margin"] > 0 else "direct"


def acp_margin(threshold: float) -> Callable[[dict[str, Any]], str]:
    def choose(row: dict[str, Any]) -> str:
        if normalize(direct_pred(row)) == normalize(decomp_pred(row)):
            return current_choice(row)
        margin = row["acp_avg_margin"]
        if margin >= threshold:
            return "decomp"
        if margin <= -threshold:
            return "direct"
        return current_choice(row)

    return choose


def acp_decomp_guard(threshold: float) -> Callable[[dict[str, Any]], str]:
    """Only veto current direct choices when the candidate posterior favors decomp."""

    def choose(row: dict[str, Any]) -> str:
        if current_choice(row) == "direct" and row["acp_avg_margin"] >= threshold:
            return "decomp"
        return current_choice(row)

    return choose


def acp_direct_rescue(threshold: float) -> Callable[[dict[str, Any]], str]:
    """Only rescue current decomp choices when the candidate posterior favors direct."""

    def choose(row: dict[str, Any]) -> str:
        if current_choice(row) == "decomp" and row["acp_avg_margin"] <= -threshold:
            return "direct"
        return current_choice(row)

    return choose


def render_eval_table(
    scored_by_dataset: dict[str, list[dict[str, Any]]],
    policies: list[tuple[str, Callable[[dict[str, Any]], str]]],
) -> str:
    lines = [
        "| Policy | Dataset | EM | F1 | Direct % | Changed | Rescue/Lost |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    macro: dict[str, list[EvalResult]] = {name: [] for name, _ in policies}
    for policy_name, chooser in policies:
        for dataset, rows in scored_by_dataset.items():
            result = evaluate(rows, chooser)
            macro[policy_name].append(result)
            lines.append(
                f"| {policy_name} | {dataset} | {result.em:.2f} | {result.f1:.2f} | "
                f"{result.direct_rate:.1f} | {result.changed} | {result.rescued}/{result.lost} |"
            )

    lines.extend(["", "| Policy | Macro EM | Macro F1 | Macro Direct % |", "|---|---:|---:|---:|"])
    for policy_name, _ in policies:
        results = macro[policy_name]
        lines.append(
            f"| {policy_name} | "
            f"{mean([r.em for r in results]):.2f} | "
            f"{mean([r.f1 for r in results]):.2f} | "
            f"{mean([r.direct_rate for r in results]):.1f} |"
        )
    return "\n".join(lines)


def render_diagnostic_table(scored_by_dataset: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "| Dataset | n | current wrong | AUROC entropy | AUROC ACP margin | direct-only ACP chooses direct | decomp-only ACP chooses decomp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    all_rows = []
    for dataset, rows in scored_by_dataset.items():
        all_rows.extend(rows)
        labels = [1 - current_em(row) for row in rows]
        direct_only = [row for row in rows if int(row.get("direct_em") or 0) and not int(row.get("decomp_em") or 0)]
        decomp_only = [row for row in rows if int(row.get("decomp_em") or 0) and not int(row.get("direct_em") or 0)]
        direct_hits = sum(1 for row in direct_only if row["acp_avg_margin"] < 0)
        decomp_hits = sum(1 for row in decomp_only if row["acp_avg_margin"] > 0)
        lines.append(
            f"| {dataset} | {len(rows)} | {sum(labels)} | "
            f"{fmt_auc(auroc([direct_entropy(row) for row in rows], labels))} | "
            f"{fmt_auc(auroc([abs(row['acp_avg_margin']) for row in rows], [0 for _ in rows])) if False else fmt_auc(auroc([row['acp_avg_margin'] for row in rows], labels))} | "
            f"{direct_hits}/{len(direct_only)} | {decomp_hits}/{len(decomp_only)} |"
        )

    labels = [1 - current_em(row) for row in all_rows]
    direct_only = [row for row in all_rows if int(row.get("direct_em") or 0) and not int(row.get("decomp_em") or 0)]
    decomp_only = [row for row in all_rows if int(row.get("decomp_em") or 0) and not int(row.get("direct_em") or 0)]
    direct_hits = sum(1 for row in direct_only if row["acp_avg_margin"] < 0)
    decomp_hits = sum(1 for row in decomp_only if row["acp_avg_margin"] > 0)
    lines.append(
        f"| ALL | {len(all_rows)} | {sum(labels)} | "
        f"{fmt_auc(auroc([direct_entropy(row) for row in all_rows], labels))} | "
        f"{fmt_auc(auroc([row['acp_avg_margin'] for row in all_rows], labels))} | "
        f"{direct_hits}/{len(direct_only)} | {decomp_hits}/{len(decomp_only)} |"
    )
    return "\n".join(lines)


def clean_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_examples(scored_by_dataset: dict[str, list[dict[str, Any]]]) -> str:
    rows = [row for rows in scored_by_dataset.values() for row in rows]
    changed_good = [
        row for row in rows
        if choose_em(row, acp_argmax(row)) and not current_em(row)
    ]
    changed_bad = [
        row for row in rows
        if current_em(row) and not choose_em(row, acp_argmax(row))
    ]

    def table(title: str, subset: list[dict[str, Any]]) -> list[str]:
        out = [
            f"## {title}",
            "",
            "| Dataset | Question | Gold | Direct | Decomp | Current | ACP | margin |",
            "|---|---|---|---|---|---|---|---:|",
        ]
        for row in sorted(subset, key=lambda r: abs(r["acp_avg_margin"]), reverse=True)[:12]:
            out.append(
                f"| {row['dataset']} | {clean_cell(row.get('question'))} | {clean_cell(row.get('gold'))} | "
                f"{clean_cell(direct_pred(row))} | {clean_cell(decomp_pred(row))} | "
                f"{current_choice(row)} | {acp_argmax(row)} | {row['acp_avg_margin']:.3f} |"
            )
        return out

    return "\n".join(
        table("ACP Argmax Rescues", changed_good)
        + [""]
        + table("ACP Argmax Regressions", changed_bad)
    )


def write_report(
    scored_by_dataset: dict[str, list[dict[str, Any]]],
    out: Path,
    *,
    per_dataset: int,
    sample_mode: str,
    with_context: bool,
) -> None:
    policies: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
        ("current", current_policy),
        ("acp_argmax", acp_argmax),
        ("acp_margin_0.05", acp_margin(0.05)),
        ("acp_margin_0.10", acp_margin(0.10)),
        ("acp_margin_0.20", acp_margin(0.20)),
        ("acp_direct_rescue_0.10", acp_direct_rescue(0.10)),
        ("acp_decomp_guard_0.10", acp_decomp_guard(0.10)),
    ]

    lines = [
        "# Answer Candidate Posterior Probe",
        "",
        "This probe tests a model-based verifier signal for choosing between saved",
        "direct and decomposed answers. It does not use the gold answer when scoring.",
        "",
        "```text",
        "score(a) = mean_token_log p_model(a | original question, optional context)",
        "ACP margin = score(decomp_pred) - score(direct_pred)",
        "p(decomp | candidates) = softmax(score(decomp), score(direct))",
        "```",
        "",
        f"Rows per dataset: {'all' if per_dataset <= 0 else per_dataset}.",
        f"Sample mode: {sample_mode}.",
        f"Context-aware scoring: {with_context}.",
        "",
        "## Diagnostic Signal",
        "",
        render_diagnostic_table(scored_by_dataset),
        "",
        "## Verifier Replay",
        "",
        render_eval_table(scored_by_dataset, policies),
        "",
        render_examples(scored_by_dataset),
        "",
        "## Interpretation Notes",
        "",
        "- ACP is not another rule over surface forms. It is a candidate ranking score",
        "  induced by the model's conditional distribution over the two available",
        "  answer candidates.",
        "- ACP can replace only the direct-vs-decompose selection layer, not the",
        "  decomposition generator or retriever.",
        "- If `acp_argmax` improves routing-sensitive samples but hurts the full replay,",
        "  ACP is better used as a soft feature inside the verifier rather than as a",
        "  hard replacement.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def write_scored_jsonl(scored_by_dataset: dict[str, list[dict[str, Any]]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rows in scored_by_dataset.values():
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--per-dataset", type=int, default=100)
    ap.add_argument("--sample-mode", choices=["first", "low_entropy", "routing"], default="first")
    ap.add_argument("--with-context", action="store_true")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--out", default="main_methodology/results/answer_candidate_posterior_probe.md")
    ap.add_argument("--scored-out", default="")
    ap.add_argument("--file", action="append", default=[], metavar="DATASET:PATH")
    args = ap.parse_args()

    files = DEFAULT_FILES.copy()
    for spec in args.file:
        dataset, path = spec.split(":", 1)
        files[dataset] = path

    rows_by_dataset = load_rows(
        files,
        per_dataset=args.per_dataset,
        sample_mode=args.sample_mode,
        with_context=args.with_context,
    )

    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    scored_by_dataset = score_rows(
        model,
        tok,
        rows_by_dataset,
        with_context=args.with_context,
        top_k=args.top_k,
    )
    write_report(
        scored_by_dataset,
        Path(args.out),
        per_dataset=args.per_dataset,
        sample_mode=args.sample_mode,
        with_context=args.with_context,
    )
    if args.scored_out:
        write_scored_jsonl(scored_by_dataset, Path(args.scored_out))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
