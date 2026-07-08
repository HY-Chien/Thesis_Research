"""Download HotpotQA (distractor) and convert to MuSiQue-aligned raw format.

Run:
    conda activate adaptive_rag
    python /home/S113065528/Research/main_methodology/scripts/prepare_hotpotqa.py

Output:
    main_methodology/datasets/hotpotqa/raw/{train,dev}.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
OUT_ROOT = Path("/home/S113065528/Research/main_methodology/datasets/hotpotqa")
SPLIT_MAP = {"train": "train", "validation": "dev"}


def to_musique_like(ex: dict) -> dict:
    titles = ex["context"]["title"]
    sents = ex["context"]["sentences"]
    sup_titles = set(ex["supporting_facts"]["title"])

    paragraphs = []
    for i, (title, sent_list) in enumerate(zip(titles, sents)):
        paragraphs.append({
            "idx": i,
            "title": title,
            "paragraph_text": " ".join(sent_list).strip(),
            "is_supporting": title in sup_titles,
        })

    supporting_facts = [
        {"title": t, "sent_id": s}
        for t, s in zip(ex["supporting_facts"]["title"], ex["supporting_facts"]["sent_id"])
    ]

    return {
        "id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type"),
        "level": ex.get("level"),
        "paragraphs": paragraphs,
        "supporting_facts": supporting_facts,
    }


def main() -> None:
    print(f"Loading {REPO} ({CONFIG}) ...")
    ds = load_dataset(REPO, CONFIG, trust_remote_code=True)
    print(f"Splits: {list(ds.keys())}")

    for hf_split, stem in SPLIT_MAP.items():
        split = ds[hf_split]
        out_path = OUT_ROOT / "raw" / f"{stem}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for ex in split:
                f.write(json.dumps(to_musique_like(ex), ensure_ascii=False) + "\n")
                n += 1
        print(f"  {hf_split:>10s} -> raw/{stem}.jsonl ({n} rows)")

    print(f"\nDone. Files under: {OUT_ROOT}")


if __name__ == "__main__":
    main()
