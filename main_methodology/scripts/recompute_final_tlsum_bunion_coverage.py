"""Recompute B-union full supporting-title coverage for final tl_sum traces.

Unlike the legacy m3 detail files, each output row records the exact resolved
sub-question strings used for retrieval, making alignment auditable.
Only B-union@10 is queried here; unchanged original-question A coverage is
read from the existing m3 detail file by the downstream taxonomy script.

Output: results/{ds}_final_tlsum_bunion_coverage.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from replay_verifier_tlsum_fixA import build  # noqa: E402
from seek.retriever import OpenDomainBM25  # noqa: E402


DEV = {
    "hotpotqa": ROOT / "datasets/hotpotqa/raw/dev_first1000.jsonl",
    "2wiki": ROOT / "datasets/2wiki/raw/dev_first1000.jsonl",
    "musique": ROOT / "datasets/musique/raw/dev_first1000.jsonl",
}
WS = re.compile(r"\s+")


def norm_title(title: str) -> str:
    return WS.sub(" ", (title or "").lower().replace("_", " ").strip())


def supporting_titles(record: dict) -> set[str]:
    return {
        norm_title(paragraph["title"])
        for paragraph in record.get("paragraphs", [])
        if paragraph.get("is_supporting") and paragraph.get("title")
    }


def retrieved_titles(hits: list[dict]) -> set[str]:
    return {norm_title(hit.get("title") or "") for hit in hits
            if hit.get("title")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", choices=sorted(DEV), required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dev = {
        str(rec["id"]): rec
        for rec in (
            json.loads(line)
            for line in DEV[args.ds].read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    rows = build(args.ds)
    retriever = OpenDomainBM25(corpus_name="wiki")
    output = RESULTS / f"{args.ds}_final_tlsum_bunion_coverage.jsonl"
    done: set[str] = set()
    if args.resume and output.exists():
        done = {
            str(json.loads(line)["id"])
            for line in output.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    query_cache: dict[str, set[str]] = {}

    with output.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for index, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            gold = supporting_titles(dev[row["id"]])
            queries = [step.get("sub_q") or step.get("raw_sub_q") or ""
                       for step in row["steps"]]
            queries = [query for query in queries if query] or [row["question"]]
            union: set[str] = set()
            for query in queries:
                if query not in query_cache:
                    query_cache[query] = retrieved_titles(
                        retriever.topk(query, k=args.k, title_field=True)
                    )
                union |= query_cache[query]
            hit = len(gold & union)
            record = {
                "id": row["id"],
                "queries": queries,
                "gold": sorted(gold),
                "k": args.k,
                "Bunion": {"hit": hit, "full": int(hit == len(gold))},
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            if index % 100 == 0:
                print(f"{args.ds}: {index}/{len(rows)}", flush=True)

    print(f"wrote {output}")


if __name__ == "__main__":
    main()
