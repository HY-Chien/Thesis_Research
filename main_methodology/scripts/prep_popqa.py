"""Convert deeprag_compare PopQA top1000 → SEER-compatible jsonl with BM25-retrieved paragraphs.

For each question we pre-fetch top-K passages from the local `wiki` (psgs_w100, ~21M docs)
Elasticsearch index. SEER's PerQuestionBM25 will then re-rank these per sub-question.
This matches DeepRAG's "global Wikipedia BM25" protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from elasticsearch import Elasticsearch
from tqdm import tqdm


def search(es: Elasticsearch, q: str, k: int = 30):
    body = {
        "size": k,
        "query": {
            "bool": {
                "should": [
                    {"match": {"paragraph_text": q}},
                    {"match": {"title": q}},
                ]
            }
        },
    }
    res = es.search(index="wiki", body=body)
    return res["hits"]["hits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/S113065528/Research/deeprag_compare/data/top1000/popqa.jsonl")
    ap.add_argument("--out", default="/home/S113065528/Research/main_methodology/datasets/popqa/raw/dev.jsonl")
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    es = Elasticsearch(["http://localhost:9200"], request_timeout=60)
    assert es.ping(), "ES not reachable on localhost:9200"

    rows = [json.loads(l) for l in open(args.src)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_empty = 0
    with open(args.out, "w") as f:
        for r in tqdm(rows, desc="bm25"):
            q = r["question"]
            golds = r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            hits = search(es, q, args.k)
            paragraphs = [
                {
                    "idx": i,
                    "title": h["_source"].get("title", ""),
                    "paragraph_text": h["_source"].get("paragraph_text", ""),
                    "is_supporting": False,
                }
                for i, h in enumerate(hits)
            ]
            if not paragraphs:
                n_empty += 1
            out = {
                "id": r["qid"],
                "question": q,
                "answer": golds[0],
                "answer_aliases": golds[1:],
                "paragraphs": paragraphs,
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows → {args.out}  (empty retrievals: {n_empty})")


if __name__ == "__main__":
    main()
