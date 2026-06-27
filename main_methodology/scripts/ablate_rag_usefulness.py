"""Ablation: is retrieval-after-decomposition useful, and is the entropy gate a
useful retrieve-or-not signal?

Answers two questions the thesis must defend:

  Q1  拆解後 RAG 有用?   decompose+retrieve  vs  decompose+NO-retrieve(never-RAG)
  Q2  信心指標判斷要不要 RAG 有用?
        (a) gate(tau) vs always-RAG : same EM, fewer retrievals? (efficiency)
        (b) does entropy DISCRIMINATE the steps/questions that need retrieval?
            AUROC(question max-entropy -> never-RAG answer is WRONG) and
            retrieval-rescue rate stratified by entropy quartile.

Usage:
  python scripts/ablate_rag_usefulness.py \
      --never results/musique200_C_tauInf.jsonl \
      --gate  results/musique200_C_maxent0.3.jsonl \
      --always results/musique200_C_tau0.0.jsonl \
      --name "MuSiQue-200 (distractor)"

never/always optional; with only --gate you still get its EM + retr-rate.
AUROC needs --never (its per-question entropy + correctness are the labels)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seek.verifier import exact_match, token_f1  # noqa: E402


def _golds(row: dict) -> list[str]:
    g = row.get("gold")
    if g is None:
        g = row.get("answer") or row.get("answers") or ""
    return g if isinstance(g, list) else [g]


def _step_entropy(step: dict) -> float:
    """Robust to schema drift: newer files use max_ent/signal_value, the early
    tauInf dump used a single 'entropy' key. We want the gating signal
    (token max-entropy)."""
    for k in ("max_ent", "signal_value", "entropy"):
        v = step.get(k)
        if v is not None and v >= 0:
            return float(v)
    return 0.0


def load(path: str | None):
    if not path:
        return None
    return [json.loads(l) for l in open(path)]


def _em(row: dict) -> int:
    """Prefer the run-time stored em (matches headline tables / uses aliases);
    fall back to recomputing from gold."""
    v = row.get("em")
    if v is not None:
        return int(v)
    return int(exact_match(row.get("pred", ""), _golds(row)))


def basic(rows: list[dict]) -> dict:
    n = len(rows)
    em = sum(_em(r) for r in rows) / n
    f1 = sum(token_f1(r.get("pred", ""), _golds(r)) for r in rows) / n
    tot = ret = 0
    for r in rows:
        for s in (r.get("steps") or []):
            tot += 1
            if s.get("retrieved"):
                ret += 1
    return dict(n=n, em=em * 100, f1=f1 * 100,
                retr=(ret / tot * 100 if tot else 0.0), steps=tot)


def auroc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney AUROC; label 1 = positive (here: never-RAG WRONG)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for ng in neg:
            if p > ng:
                wins += 1
            elif p == ng:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def discrimination(never: list[dict], retr: list[dict], retr_name: str):
    """Per-question: entropy (from never-RAG run, = parametric signal) vs whether
    never-RAG is wrong; then how often the retrieval run RESCUES the wrong ones,
    split by entropy quartile."""
    by_id_r = {(r.get("id") or r.get("question")): r for r in retr}
    qent, never_wrong, rescued = [], [], []
    for r in never:
        key = r.get("id") or r.get("question")
        if key not in by_id_r:
            continue
        steps = r.get("steps") or []
        if not steps:
            continue
        ent = max(_step_entropy(s) for s in steps)
        nw = 0 if _em(r) else 1
        rc = _em(by_id_r[key])
        qent.append(ent)
        never_wrong.append(nw)
        rescued.append(rc)
    a = auroc(qent, never_wrong)
    # quartile rescue: among never-WRONG questions, fraction the retrieval run fixes
    paired = sorted(zip(qent, never_wrong, rescued), key=lambda x: x[0])
    qs = len(paired) // 4 or 1
    print(f"  AUROC(entropy -> never-RAG wrong)        = {a:.3f}   (>0.5 = entropy flags the right questions)")
    print(f"  retrieval source = {retr_name}")
    print(f"  {'entropy quartile':22} {'n_wrong':>8} {'rescued':>8} {'rescue%':>8} {'mean_ent':>9}")
    for qi in range(4):
        chunk = paired[qi * qs: (qi + 1) * qs] if qi < 3 else paired[qi * qs:]
        wrong = [(e, r) for e, nw, r in chunk if nw == 1]
        nwrong = len(wrong)
        resc = sum(r for _, r in wrong)
        ment = sum(e for e, _, _ in chunk) / len(chunk) if chunk else 0
        pct = resc / nwrong * 100 if nwrong else 0
        print(f"  Q{qi+1} ({'low' if qi==0 else 'high' if qi==3 else 'mid':>4})            {nwrong:8d} {resc:8d} {pct:8.1f} {ment:9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--never")
    ap.add_argument("--gate")
    ap.add_argument("--always")
    ap.add_argument("--name", default="(unnamed)")
    args = ap.parse_args()

    never, gate, always = load(args.never), load(args.gate), load(args.always)
    print(f"\n=== {args.name} ===")
    print(f"{'condition':28} {'n':>4} {'EM':>6} {'F1':>6} {'retr%':>7}")
    for label, rows in [("never-RAG (decompose only)", never),
                        ("entropy-gate (tau)", gate),
                        ("always-RAG (every step)", always)]:
        if rows is None:
            continue
        b = basic(rows)
        print(f"{label:28} {b['n']:4d} {b['em']:6.1f} {b['f1']:6.1f} {b['retr']:7.1f}")

    if never and gate:
        print("\n-- Q2(b) entropy discrimination (gate as retrieval source) --")
        discrimination(never, gate, "gate")
    elif never and always:
        print("\n-- Q2(b) entropy discrimination (always-RAG as retrieval source) --")
        discrimination(never, always, "always-RAG")


if __name__ == "__main__":
    main()
