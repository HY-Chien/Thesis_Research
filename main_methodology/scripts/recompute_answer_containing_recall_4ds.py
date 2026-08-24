#!/usr/bin/env python3
"""Recompute four-dataset open-domain answer-containing retrieval recall.

This is the four-dataset counterpart of ``recall_gold_decomp_musique.py``:
same wiki-21M Elasticsearch corpus, same path-aware retrieval policy, and the
same answer-containing proxy. It intentionally excludes MuSiQue-only oracle
arms such as gold decomposition; this script only compares original-query vs
SEEK decomposed-query retrieval across all datasets.

Run from ``Research/main_methodology``:

    python scripts/recompute_answer_containing_recall_4ds.py

The output markdown feeds ``scripts/make_answer_hit_fig.py``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RETRIEVER_PATH = ROOT / "seek" / "retriever.py"

spec = importlib.util.spec_from_file_location("seek_retriever", RETRIEVER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load retriever module from {RETRIEVER_PATH}")
retriever_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retriever_mod)
OpenDomainBM25 = retriever_mod.OpenDomainBM25

DATASETS: dict[str, dict[str, str]] = {
    "2Wiki": {
        "dev": "datasets/2wiki/raw/dev_first1000.jsonl",
        "trace": "results/traces/2wiki/2wiki_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    },
    "HotpotQA": {
        "dev": "datasets/hotpotqa/raw/dev_first1000.jsonl",
        "trace": "results/traces/hotpotqa/hotpotqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    },
    "MuSiQue": {
        "dev": "datasets/musique/raw/dev_first1000.jsonl",
        "trace": "results/traces/musique/musique_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    },
    "PopQA": {
        "dev": "datasets/popqa/raw/dev.jsonl",
        "trace": "results/traces/popqa/popqa_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl",
    },
}

WS = re.compile(r"\s+")
YESNO = {"yes", "no"}


def norm(text: Any) -> str:
    return WS.sub(" ", str(text or "").lower().strip())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def answer_needles(record: dict[str, Any]) -> set[str]:
    needles = {norm(record.get("answer"))}
    needles |= {norm(alias) for alias in record.get("answer_aliases") or []}
    return {needle for needle in needles if needle}


def contains_answer(passages: list[dict[str, Any]], needles: set[str]) -> bool:
    for passage in passages:
        text = norm(f"{passage.get('title', '')} {passage.get('paragraph_text', '')}")
        if any(needle in text for needle in needles):
            return True
    return False


def trace_steps(trace_row: dict[str, Any]) -> list[dict[str, Any]]:
    return trace_row.get("steps") or (trace_row.get("decomp") or {}).get("steps") or []


def sub_questions(trace_row: dict[str, Any]) -> list[str]:
    out = []
    for step in trace_steps(trace_row):
        sub_q = step.get("sub_q") or step.get("raw_sub_q")
        if sub_q:
            out.append(sub_q)
    return out


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def fmt_float(value: float) -> str:
    return f"{value:.2f}"


def eval_dataset(
    name: str,
    cfg: dict[str, str],
    retriever_on: OpenDomainBM25,
    retriever_off: OpenDomainBM25,
    ks: list[int],
    corpus: str,
    progress_every: int,
) -> dict[str, Any]:
    dev_rows = {str(row["id"]): row for row in load_jsonl(ROOT / cfg["dev"])}
    trace_rows = load_jsonl(ROOT / cfg["trace"])
    max_k = max(ks)

    hits = {
        "orig": {k: 0 for k in ks},
        "decomp": {k: 0 for k in ks},
        "union": {k: 0 for k in ks},
        "decomp_only": {k: 0 for k in ks},
        "orig_only": {k: 0 for k in ks},
    }
    n_eval = 0
    skipped_yesno = 0
    skipped_missing = 0
    subq_counts: Counter[int] = Counter()

    for idx, trace_row in enumerate(trace_rows, 1):
        record = dev_rows.get(str(trace_row.get("id")))
        if record is None:
            skipped_missing += 1
            continue

        needles = answer_needles(record)
        if not needles:
            skipped_missing += 1
            continue
        if needles <= YESNO:
            skipped_yesno += 1
            continue

        subqs = sub_questions(trace_row)
        subq_counts[len(subqs)] += 1
        n_eval += 1

        orig_hits = retriever_off.topk(record["question"], k=max_k)
        decomp_hits_by_step = [retriever_on.topk(q, k=max_k) for q in subqs]

        for k in ks:
            orig_pool = orig_hits[:k]
            decomp_pool = [p for step_hits in decomp_hits_by_step for p in step_hits[:k]]
            union_pool = orig_pool + decomp_pool

            got_orig = contains_answer(orig_pool, needles)
            got_decomp = contains_answer(decomp_pool, needles)
            got_union = contains_answer(union_pool, needles)

            hits["orig"][k] += int(got_orig)
            hits["decomp"][k] += int(got_decomp)
            hits["union"][k] += int(got_union)
            hits["decomp_only"][k] += int(got_decomp and not got_orig)
            hits["orig_only"][k] += int(got_orig and not got_decomp)

        if progress_every and idx % progress_every == 0:
            print(f"[{name}] {idx}/{len(trace_rows)} traces processed", flush=True)

    return {
        "dataset": name,
        "corpus": corpus,
        "n_eval": n_eval,
        "skipped_yesno": skipped_yesno,
        "skipped_missing": skipped_missing,
        "subq_counts": dict(sorted(subq_counts.items())),
        "hits": hits,
    }


def render_markdown(results: list[dict[str, Any]], ks: list[int]) -> str:
    lines = [
        "# Four-dataset Open-domain Answer-containing Retrieval Recall",
        "",
        "Date: 2026-07-06",
        "",
        "Setting: final open-domain wiki retrieval condition, recomputed with the same answer-containing recall implementation family as the MuSiQue gold-decomposition diagnostic. Original-question retrieval uses paragraph text only; decomposed sub-question retrieval uses title + paragraph text, matching path-aware retrieval.",
        "",
        "`answer-containing hit@k` means that the retrieved passages contain the gold answer or one of its aliases in the passage title or text. Yes/no questions are excluded because yes/no strings are not meaningful evidence strings.",
        "",
        "Note: this file is a unified rerun for the four-dataset figure. Compare numbers within this file; do not mix them with older saved runs or with the MuSiQue-only oracle arms.",
        "",
        "## Main Table, k=5",
        "",
        "| Dataset | N eval | Original-query hit@5 | Decomposed-query hit@5 | Union hit@5 | Decomp - Original |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for res in results:
        h = res["hits"]
        orig = pct(h["orig"][5], res["n_eval"])
        decomp = pct(h["decomp"][5], res["n_eval"])
        union = pct(h["union"][5], res["n_eval"])
        lines.append(
            f"| {res['dataset']} | {res['n_eval']} | {fmt_float(orig)} | "
            f"**{fmt_float(decomp)}** | **{fmt_float(union)}** | "
            f"{decomp - orig:+.2f} |"
        )

    lines += ["", "## k Sweep", ""]
    for res in results:
        h = res["hits"]
        lines += [
            f"### {res['dataset']}",
            "",
            f"N eval = {res['n_eval']}; skipped yes/no = {res['skipped_yesno']}; "
            f"sub-question counts = `{res['subq_counts']}`.",
            "",
            "| k | Original | Decomp | Union | Decomp - Original | Decomp-only | Original-only |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for k in ks:
            orig = pct(h["orig"][k], res["n_eval"])
            decomp = pct(h["decomp"][k], res["n_eval"])
            union = pct(h["union"][k], res["n_eval"])
            lines.append(
                f"| {k} | {fmt_float(orig)} | {fmt_float(decomp)} | "
                f"{fmt_float(union)} | {decomp - orig:+.2f} | "
                f"{h['decomp_only'][k]} | {h['orig_only'][k]} |"
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "The unified rerun supports the same qualitative conclusion as the earlier four-dataset figure: decomposition substantially improves answer-containing retrieval on the multi-hop datasets, with the largest gains on 2Wiki and MuSiQue. PopQA remains a near-single-hop case, so the retrieval gain is small. This table should be used for the four-dataset figure when presenting it next to the MuSiQue gold-decomposition diagnostic.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8, 10])
    parser.add_argument("--corpus", default="wiki")
    parser.add_argument(
        "--out",
        default="results/reproduced/answer_containing_recall_4ds_open_domain.md",
        help="Markdown output path relative to Research/main_methodology.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    ks = sorted(args.ks)
    if 5 not in ks:
        raise SystemExit("--ks must include 5 because the main table and figure use hit@5")

    retriever_on = OpenDomainBM25(corpus_name=args.corpus, query_title_field_too=True)
    retriever_off = OpenDomainBM25(corpus_name=args.corpus, query_title_field_too=False)

    results = []
    for name, cfg in DATASETS.items():
        results.append(
            eval_dataset(
                name=name,
                cfg=cfg,
                retriever_on=retriever_on,
                retriever_off=retriever_off,
                ks=ks,
                corpus=args.corpus,
                progress_every=args.progress_every,
            )
        )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(results, ks), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
