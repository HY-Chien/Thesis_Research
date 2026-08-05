"""Download dgslibisey/MuSiQue and emit raw + SFT-ready jsonl splits.

Run from any directory:
    python scripts/prepare_musique.py

Output layout:
    main_methodology/datasets/musique/
        raw/{train,dev}.jsonl      # all native fields, one row per line
        sft/{train,dev}.jsonl      # {"id", "input", "output"} for decomposer SFT
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

REPO = "dgslibisey/MuSiQue"
ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "datasets" / "musique"

# HF split name -> our filename stem
SPLIT_MAP = {"train": "train", "validation": "dev"}


def format_decomposition(decomp: list[dict]) -> str:
    """Turn question_decomposition list into the SFT target string.

    Example output:
        #1: The Collegian >> owned by
        #2: When was #1 founded?
    """
    lines = []
    for i, step in enumerate(decomp, start=1):
        lines.append(f"#{i}: {step['question']}")
    return "\n".join(lines)


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    print(f"Loading {REPO} via datasets.load_dataset ...")
    ds = load_dataset(REPO)
    print(f"Splits: {list(ds.keys())}")

    for hf_split, stem in SPLIT_MAP.items():
        if hf_split not in ds:
            print(f"  [skip] split '{hf_split}' not present")
            continue
        split = ds[hf_split]

        raw_path = OUT_ROOT / "raw" / f"{stem}.jsonl"
        sft_path = OUT_ROOT / "sft" / f"{stem}.jsonl"

        def raw_rows():
            for ex in split:
                yield ex

        def sft_rows():
            for ex in split:
                yield {
                    "id": ex["id"],
                    "input": ex["question"],
                    "output": format_decomposition(ex["question_decomposition"]),
                }

        n_raw = write_jsonl(raw_path, raw_rows())
        n_sft = write_jsonl(sft_path, sft_rows())
        print(f"  {hf_split:>10s} -> raw/{stem}.jsonl ({n_raw} rows), sft/{stem}.jsonl ({n_sft} rows)")

    print("\nDone. Files written under:", OUT_ROOT)


if __name__ == "__main__":
    main()
