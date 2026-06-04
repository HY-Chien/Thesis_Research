"""Score an OSS decomposition file against MuSiQue gold.

Metrics
-------
- parse_rate: rows with decomposition_error == None
- count_match: generated step count == gold step count
- final_answer EM/F1: last generated step's answer vs gold `answer` (+aliases)
- step_answer EM: per gold step (count-matched rows), generated[i].answer vs gold[i].answer
- psi_acc: per gold step (count-matched rows), generated[i].paragraph_support_idx
  == gold[i].paragraph_support_idx

Usage:
    python eval_decomposition.py <decomposition.jsonl>
"""
from __future__ import annotations

import json
import re
import string
import sys
from collections import Counter


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def em(pred: str, golds: list[str]) -> bool:
    p = normalize(pred)
    return any(p == normalize(g) for g in golds)


def f1(pred: str, golds: list[str]) -> float:
    p = normalize(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize(g).split()
        if not p or not gt:
            best = max(best, float(p == gt))
            continue
        common = Counter(p) & Counter(gt)
        n = sum(common.values())
        if n == 0:
            continue
        prec, rec = n / len(p), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def main() -> None:
    path = sys.argv[1]
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)

    n_ok = sum(1 for r in rows if r["decomposition_error"] is None)
    n_count_match = 0
    fa_em = fa_f1 = 0.0
    step_total = step_em = psi_total = psi_hit = 0

    for r in rows:
        d = r["generated_decomposition"]
        if not d:
            continue
        gold = r["question_decomposition"]
        golds_final = [r["answer"]] + (r.get("answer_aliases") or [])

        # final answer: last generated step
        last_ans = d[-1].get("answer") or ""
        fa_em += em(last_ans, golds_final)
        fa_f1 += f1(last_ans, golds_final)

        if len(d) == len(gold):
            n_count_match += 1
            for gs, ps in zip(gold, d):
                step_total += 1
                if em(ps.get("answer") or "", [gs["answer"]]):
                    step_em += 1
                gpsi = gs.get("paragraph_support_idx")
                ppsi = ps.get("paragraph_support_idx")
                if gpsi is not None:
                    psi_total += 1
                    if ppsi == gpsi:
                        psi_hit += 1

    print(f"file: {path}")
    print(f"rows: {n}")
    print(f"parse_rate:     {n_ok}/{n} = {n_ok/n:.1%}")
    print(f"count_match:    {n_count_match}/{n_ok} = {n_count_match/max(n_ok,1):.1%}  (of parsed)")
    print(f"final_answer EM: {fa_em/max(n_ok,1):.1%}   F1: {fa_f1/max(n_ok,1):.1%}  (of parsed)")
    print(f"step_answer EM: {step_em}/{step_total} = {step_em/max(step_total,1):.1%}  (count-matched rows)")
    print(f"psi_acc:        {psi_hit}/{psi_total} = {psi_hit/max(psi_total,1):.1%}  (count-matched rows)")


if __name__ == "__main__":
    main()
