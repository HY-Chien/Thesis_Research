"""
Exp7c: Qualitative case pool on CoT results (exp7).

Same 4 groups as exp5 (A/B/C/D), but now each case includes the CoT reasoning
chain, so you can annotate whether SC's confidence aligns with the quality of
the reasoning steps (not just the final entity).

Groups:
  A — Low SC + Incorrect   (SC correctly flags bad reasoning / hallucination)
  B — High SC + Incorrect  (confident wrong reasoning — SC fails)
  C — Low SC + Correct     (SC over-triggers on correct answers)
  D — SC/PPL disagreement  (largest rank gap)
"""

import argparse
import json
from pathlib import Path

ROOT = Path("/home/S113065528/Adaptive-RAG/experiment/self_certainty")

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="2wiki")
parser.add_argument("--n", type=int, default=15)
args = parser.parse_args()

RESULTS_IN = ROOT / "results" / f"exp7_cot_{args.dataset}_results.json"
OUT_JSON = ROOT / "results" / f"exp7c_cot_cases_{args.dataset}.json"
OUT_MD = ROOT / "results" / f"exp7c_cot_cases_{args.dataset}.md"

GROUP_DESC = {
    "A_low_sc_incorrect": "Low SC + Incorrect — SC correctly flags bad CoT (RAG should trigger)",
    "B_high_sc_incorrect": "High SC + Incorrect — Confident wrong reasoning (SC fails)",
    "C_low_sc_correct": "Low SC + Correct — SC over-triggers despite correct answer",
    "D_sc_ppl_disagreement": "SC/PPL disagreement — largest rank gap between the signals",
}


def rank_map(items, key, ascending=True):
    s = sorted(items, key=lambda r: r[key], reverse=not ascending)
    return {r["id"]: i for i, r in enumerate(s)}


def pick_groups(results, n):
    n_total = len(results)
    sorted_sc = sorted(results, key=lambda r: r["sc"])
    low_ids = {r["id"] for r in sorted_sc[: int(0.4 * n_total)]}
    high_ids = {r["id"] for r in sorted_sc[-int(0.4 * n_total):]}

    A = sorted([r for r in results if r["id"] in low_ids and not r["is_correct"]],
               key=lambda r: r["sc"])[:n]
    B = sorted([r for r in results if r["id"] in high_ids and not r["is_correct"]],
               key=lambda r: -r["sc"])[:n]
    C = sorted([r for r in results if r["id"] in low_ids and r["is_correct"]],
               key=lambda r: r["sc"])[:n]

    sc_r = rank_map(results, "sc", True)       # high rank = high SC = confident
    ppl_r = rank_map(results, "ppl", False)    # high rank = low PPL = confident
    for r in results:
        r["_d"] = abs(sc_r[r["id"]] - ppl_r[r["id"]])
    D = sorted(results, key=lambda r: -r["_d"])[:n]
    for r in results:
        r.pop("_d", None)

    return {"A_low_sc_incorrect": A, "B_high_sc_incorrect": B,
            "C_low_sc_correct": C, "D_sc_ppl_disagreement": D}


def trim(r):
    return {"id": r["id"], "question": r["question"],
            "ground_truths": r["ground_truths"],
            "cot": r["cot"], "final_answer": r["final_answer"],
            "sc": round(r["sc"], 4), "ppl": round(r["ppl"], 4),
            "n_tokens": r["n_tokens"], "is_correct": r["is_correct"]}


def to_markdown(groups):
    lines = [f"# Exp7c — CoT Qualitative Cases ({args.dataset.upper()})\n",
             "> Annotation fields: **reasoning_quality** (sound / flawed / fabricated), "
             "**failure_mode** (wrong-fact / wrong-linking / refusal / hallucination / other), "
             "**answer_length**, **notes**.\n"]
    for key, items in groups.items():
        lines.append(f"\n## Group {key[0]} — {GROUP_DESC[key]}\n")
        for i, r in enumerate(items, 1):
            correct = "✓" if r["is_correct"] else "✗"
            gold = " / ".join(r["ground_truths"])
            cot = r["cot"].strip()
            if len(cot) > 1500:
                cot = cot[:1500] + " …[truncated]"
            lines.append(f"### {key[0]}-{i:02d}  [{correct}]  SC={r['sc']:.2f}  "
                         f"PPL={r['ppl']:.3f}  n_tok={r['n_tokens']}")
            lines.append(f"- **Q**     : {r['question']}")
            lines.append(f"- **Gold**  : {gold}")
            lines.append(f"- **Final** : {r['final_answer']}")
            lines.append("- **CoT**   :")
            lines.append("  > " + cot.replace("\n", "\n  > "))
            lines.append("- **annot** : reasoning_quality=___  failure_mode=___  "
                         "answer_length=___")
            lines.append("- **notes** : ___")
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    with open(RESULTS_IN) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} from {RESULTS_IN}")

    groups = pick_groups(results, args.n)
    trimmed = {k: [trim(r) for r in v] for k, v in groups.items()}

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(to_markdown(trimmed))
    print(f"Wrote {OUT_JSON}\nWrote {OUT_MD}")
    for k, v in trimmed.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
