#!/usr/bin/env python3
"""Create SEEK's four compact, order-preserving evaluation subsets.

The selection follows DeepRAG: take the first 1,000 rows from each official
split without shuffling. Only question identifiers and gold answers are
written; retrieval passages come from the external ``wiki`` index at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


OUTPUT_NAMES = {
    "hotpotqa": "hotpotqa.jsonl",
    "twowiki": "2wikimultihopqa.jsonl",
    "musique": "musique.jsonl",
    "popqa": "popqa.jsonl",
}


def load_json_array(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON array: {path}")
    return rows


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected an object at {path}:{line_number}")
            yield row


def identifier(row: dict[str, Any]) -> str:
    for key in ("_id", "id", "qid"):
        if row.get(key) is not None:
            return str(row[key])
    raise ValueError("source row is missing _id/id/qid")


def require_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"source row requires a string '{key}'")
    return value


def hotpot_rows(path: Path) -> Iterable[dict[str, Any]]:
    for row in load_json_array(path):
        yield {
            "qid": identifier(row),
            "question": require_text(row, "question"),
            "answer": require_text(row, "answer"),
            "answer_id": None,
        }


def twowiki_rows(path: Path) -> Iterable[dict[str, Any]]:
    for row in load_json_array(path):
        yield {
            "qid": identifier(row),
            "question": require_text(row, "question"),
            "answer": require_text(row, "answer"),
            "answer_id": row.get("answer_id"),
        }


def musique_rows(path: Path) -> Iterable[dict[str, Any]]:
    for row in load_jsonl(path):
        aliases = row.get("answer_aliases") or []
        if not isinstance(aliases, list):
            raise ValueError("MuSiQue 'answer_aliases' must be a list")
        yield {
            "qid": identifier(row),
            "question": require_text(row, "question"),
            "answer": [require_text(row, "answer"), *aliases],
            "answer_id": None,
        }


def popqa_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                answers = json.loads(row["possible_answers"])
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError("PopQA requires JSON 'possible_answers'") from exc
            if not isinstance(answers, list) or not answers:
                raise ValueError("PopQA 'possible_answers' must be a non-empty list")
            yield {
                "qid": identifier(row),
                "question": require_text(row, "question"),
                "answer": answers,
                "answer_id": None,
            }


def write_subset(
    rows: Iterable[dict[str, Any]],
    output_path: Path,
    limit: int,
    force: bool,
) -> None:
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")

    selected = list(islice(rows, limit))
    if len(selected) != limit:
        raise ValueError(
            f"{output_path.name}: requested {limit} rows, found {len(selected)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{output_path}: {len(selected)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hotpotqa", required=True, type=Path,
                        help="official HotpotQA distractor dev JSON array")
    parser.add_argument("--twowiki", required=True, type=Path,
                        help="official 2WikiMultiHopQA dev JSON array")
    parser.add_argument("--musique", required=True, type=Path,
                        help="official MuSiQue-Ans dev JSONL")
    parser.add_argument("--popqa", required=True, type=Path,
                        help="official PopQA TSV")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    sources = {
        "hotpotqa": hotpot_rows(args.hotpotqa),
        "twowiki": twowiki_rows(args.twowiki),
        "musique": musique_rows(args.musique),
        "popqa": popqa_rows(args.popqa),
    }
    for dataset, rows in sources.items():
        write_subset(
            rows,
            args.output_dir / OUTPUT_NAMES[dataset],
            args.limit,
            args.force,
        )


if __name__ == "__main__":
    main()
