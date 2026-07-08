"""Download 2WikiMultiHopQA and convert to MuSiQue-aligned raw format.

Source: xanhho/2WikiMultihopQA (parquet mirror of the original Alab-NII release).

Run:
    conda activate adaptive_rag
    python /home/S113065528/Research/main_methodology/scripts/prepare_2wiki.py

Output:
    main_methodology/datasets/2wiki/raw/{train,dev}.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO = "xanhho/2WikiMultihopQA"
OUT_ROOT = Path("/home/S113065528/Research/main_methodology/datasets/2wiki")
SPLIT_MAP = {"train": "train", "dev": "dev"}  # 2Wiki HF split is named "dev"


def _maybe_json(v):
    """Parquet mirror stores context/supporting_facts/evidences as JSON strings."""
    return json.loads(v) if isinstance(v, str) else v


def to_musique_like(ex: dict) -> dict:
    # context: [[title, [sentences...]], ...]
    context = _maybe_json(ex["context"])
    sup = _maybe_json(ex["supporting_facts"])      # [[title, sent_id], ...]
    ev = _maybe_json(ex.get("evidences"))          # [[fact, relation, entity], ...]

    sup_titles = {row[0] for row in sup}

    paragraphs = []
    for i, (title, sent_list) in enumerate(context):
        paragraphs.append({
            "idx": i,
            "title": title,
            "paragraph_text": " ".join(sent_list).strip(),
            "is_supporting": title in sup_titles,
        })

    supporting_facts = [{"title": t, "sent_id": s} for t, s in sup]

    evidences = None
    if ev:
        evidences = [{"fact": f, "relation": r, "entity": e} for f, r, e in ev]

    return {
        "id": ex["_id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "type": ex.get("type"),
        "paragraphs": paragraphs,
        "supporting_facts": supporting_facts,
        "evidences": evidences,
    }


def main() -> None:
    print(f"Loading {REPO} via direct parquet (bypassing buggy custom loader) ...")
    base = f"hf://datasets/{REPO}"
    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"{base}/train.parquet",
            "dev": f"{base}/dev.parquet",
            "test": f"{base}/test.parquet",
        },
    )
    print(f"Splits: {list(ds.keys())}")

    for hf_split, stem in SPLIT_MAP.items():
        if hf_split not in ds:
            print(f"  [skip] split '{hf_split}' missing")
            continue
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
