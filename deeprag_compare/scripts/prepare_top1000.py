"""
Prepare top-1000 dev questions matching DeepRAG / DRAGIN convention.

DeepRAG's scripts/inference/run.sh uses "top 1000 questions from dev split" for
HotpotQA, 2WikiMultiHopQA, MuSiQue. We slice the raw dev files identically so
the comparison is on the same question set as Table 2 of the DeepRAG paper.

Output format (one JSON per line):
  { "qid": str, "question": str, "answer": str | list[str],
    "answer_id": str | null }
"""

import json
import os

RAW = "/home/S113065528/Adaptive-RAG/raw_data"
OUT = "/home/S113065528/Research/deeprag_compare/data/top1000"
os.makedirs(OUT, exist_ok=True)
N = 1000


def hotpotqa():
    src = os.path.join(RAW, "hotpotqa", "hotpot_dev_distractor_v1.json")
    with open(src) as f:
        data = json.load(f)
    out = []
    for ex in data[:N]:
        out.append({
            "qid": ex["_id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_id": None,
        })
    dst = os.path.join(OUT, "hotpotqa.jsonl")
    with open(dst, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"HotpotQA  -> {dst}  ({len(out)} questions)")


def wikimultihopqa():
    src = os.path.join(RAW, "2wikimultihopqa", "dev.json")
    with open(src) as f:
        data = json.load(f)
    out = []
    for ex in data[:N]:
        out.append({
            "qid": ex["_id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "answer_id": ex.get("answer_id"),
        })
    dst = os.path.join(OUT, "2wikimultihopqa.jsonl")
    with open(dst, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"2WikiMHQA -> {dst}  ({len(out)} questions)")

    # id aliases (needed for gold-answer expansion in eval)
    alias_src = os.path.join(RAW, "2wikimultihopqa", "id_aliases.json")
    alias_dst = os.path.join(OUT, "2wikimultihopqa_id_aliases.jsonl")
    if os.path.exists(alias_src):
        with open(alias_src) as f_in, open(alias_dst, "w") as f_out:
            for line in f_in:
                f_out.write(line)
        print(f"           aliases -> {alias_dst}")


def musique():
    src = os.path.join(RAW, "musique", "musique_ans_v1.0_dev.jsonl")
    out = []
    with open(src) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            ans = [ex["answer"]] + ex.get("answer_aliases", [])
            out.append({
                "qid": ex["id"],
                "question": ex["question"],
                "answer": ans,
                "answer_id": None,
            })
            if len(out) >= N:
                break
    dst = os.path.join(OUT, "musique.jsonl")
    with open(dst, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"MuSiQue   -> {dst}  ({len(out)} questions)")


def popqa():
    import json as _json
    try:
        import datasets as hf_datasets
    except ImportError:
        print("pip install datasets  (needed for PopQA)")
        return
    d = hf_datasets.load_dataset("akariasai/PopQA", split="test")
    out = []
    for ex in d:
        answers = _json.loads(ex["possible_answers"])
        out.append({
            "qid": str(ex["id"]),
            "question": ex["question"],
            "answer": answers,
            "answer_id": None,
        })
        if len(out) >= N:
            break
    dst = os.path.join(OUT, "popqa.jsonl")
    with open(dst, "w") as f:
        for e in out:
            f.write(_json.dumps(e, ensure_ascii=False) + "\n")
    print(f"PopQA     -> {dst}  ({len(out)} questions)")


if __name__ == "__main__":
    hotpotqa()
    wikimultihopqa()
    musique()
    popqa()
