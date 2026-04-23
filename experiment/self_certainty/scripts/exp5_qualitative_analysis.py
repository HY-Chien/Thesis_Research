"""
Experiment 5: Qualitative Analysis of SC as RAG-triggering signal.

Loads exp4 results (500 NQ questions with SC/PPL/is_correct) and partitions
them into 4 case groups for manual annotation:

  Group A — Low SC + Incorrect   (SC correctly flags wrong answer → RAG should trigger)
  Group B — High SC + Incorrect  (hallucination, SC fails to flag)
  Group C — Low SC + Correct     (SC over-triggers → wasted retrieval)
  Group D — SC/PPL disagreement  (rank gap between the two signals is largest)

Outputs:
  - results/exp5_qualitative_cases.json         : all 4 groups as JSON
  - results/exp5_qualitative_cases.md           : markdown table for manual annotation
"""

import json
from pathlib import Path

ROOT = Path("/home/S113065528/Adaptive-RAG/experiment/self_certainty")
RESULTS_IN = ROOT / "results" / "exp4_nq_results.json"
OUT_JSON = ROOT / "results" / "exp5_qualitative_cases.json"
OUT_MD = ROOT / "results" / "exp5_qualitative_cases.md"

N_PER_GROUP = 20


def load_results():
    with open(RESULTS_IN) as f:
        return json.load(f)


def rank_map(items, key, ascending=True):
    """Return {id: rank} where rank 0 = smallest (if ascending)."""
    sorted_items = sorted(items, key=lambda r: r[key], reverse=not ascending)
    return {r["id"]: i for i, r in enumerate(sorted_items)}


def pick_groups(results, n=N_PER_GROUP):
    # SC low = least confident; PPL high = least confident.
    # For "low SC" cutoff use bottom 40% by SC; "high SC" top 40%.
    n_total = len(results)
    sorted_by_sc = sorted(results, key=lambda r: r["sc"])
    low_sc_ids = {r["id"] for r in sorted_by_sc[: int(0.4 * n_total)]}
    high_sc_ids = {r["id"] for r in sorted_by_sc[-int(0.4 * n_total) :]}

    # Group A: low SC & incorrect — sort by SC ascending (most confident-that-it's-wrong first)
    group_a = sorted(
        [r for r in results if r["id"] in low_sc_ids and not r["is_correct"]],
        key=lambda r: r["sc"],
    )[:n]

    # Group B: high SC & incorrect — sort by SC descending (highest confidence hallucinations)
    group_b = sorted(
        [r for r in results if r["id"] in high_sc_ids and not r["is_correct"]],
        key=lambda r: -r["sc"],
    )[:n]

    # Group C: low SC & correct — sort by SC ascending
    group_c = sorted(
        [r for r in results if r["id"] in low_sc_ids and r["is_correct"]],
        key=lambda r: r["sc"],
    )[:n]

    # Group D: largest SC/PPL rank disagreement.
    # Higher SC = more confident (rank high = confident). Lower PPL = more confident.
    sc_rank = rank_map(results, "sc", ascending=True)         # high rank = high SC = confident
    ppl_rank = rank_map(results, "ppl", ascending=False)      # high rank = low PPL = confident
    for r in results:
        r["_disagree"] = abs(sc_rank[r["id"]] - ppl_rank[r["id"]])
    group_d = sorted(results, key=lambda r: -r["_disagree"])[:n]
    for r in results:
        r.pop("_disagree", None)

    return {
        "A_low_sc_incorrect": group_a,
        "B_high_sc_incorrect": group_b,
        "C_low_sc_correct": group_c,
        "D_sc_ppl_disagreement": group_d,
    }


def trim(r):
    return {
        "id": r["id"],
        "question": r["question"],
        "ground_truths": r["ground_truths"],
        "answer": r["answer"],
        "sc": round(r["sc"], 4),
        "ppl": round(r["ppl"], 4),
        "n_tokens": r["n_tokens"],
        "is_correct": r["is_correct"],
    }


GROUP_DESC = {
    "A_low_sc_incorrect": "Low SC + Incorrect — SC correctly flags hallucination (RAG should trigger)",
    "B_high_sc_incorrect": "High SC + Incorrect — Confident hallucination (SC fails)",
    "C_low_sc_correct": "Low SC + Correct — SC over-triggers (retrieval would be wasted)",
    "D_sc_ppl_disagreement": "SC/PPL disagreement — largest rank gap between the two signals",
}


def to_markdown(groups):
    lines = [
        "# Exp5 Qualitative Cases\n",
        "> See `exp5_annotation_guide.md` for what to fill in each annotation field.\n",
    ]
    for key, items in groups.items():
        lines.append(f"\n## Group {key[0]} — {GROUP_DESC[key]}\n")
        for i, r in enumerate(items, 1):
            ans = r["answer"].replace("\n", " ")
            gold = " / ".join(r["ground_truths"])
            correct = "✓" if r["is_correct"] else "✗"
            lines.append(f"### {key[0]}-{i:02d}  [{correct}]  SC={r['sc']:.2f}  PPL={r['ppl']:.3f}  n_tok={r['n_tokens']}")
            lines.append(f"- **Q**     : {r['question']}")
            lines.append(f"- **Gold**  : {gold}")
            lines.append(f"- **Pred**  : {ans}")
            lines.append("- **annot** : answer_length=___  question_type=___  entity_type=___")
            lines.append("- **notes** : ___")
            lines.append("")
    return "\n".join(lines) + "\n"


ANNOTATION_GUIDE = """# Exp5 Annotation Guide

Fill in the four fields under each case in `exp5_qualitative_cases.md`. Goal:
find *patterns* that explain when SC works and when it fails.

## 1. `answer_length`
How long is the model's predicted answer (not the question).

| value  | criterion                                | example |
|--------|------------------------------------------|---------|
| short  | 1–4 tokens, pure entity / number / date  | "Grover Cleveland", "1994" |
| medium | 5–20 tokens, one sentence                | "The song was written by Buck Moore in 2003." |
| long   | >20 tokens, multi-sentence / CoT-like    | refusal, explanation, list |

**Why it matters**: SC averages over all tokens including filler ("the", "was");
short answers give PPL an advantage because correctness concentrates on entity
tokens. If Group B (high-SC hallucinations) is dominated by `short`, that
confirms SC's weakness on factoid QA.

## 2. `question_type`
What kind of reasoning the question requires.

| value      | criterion                                             |
|------------|-------------------------------------------------------|
| factoid    | single fact lookup ("capital of X", "who wrote Y")    |
| temporal   | asks for a date / year / "when"                       |
| numeric    | asks for a count / number / statistic                 |
| list       | asks for multiple items ("name all members of …")     |
| multi-hop  | needs chaining 2+ facts ("president when X happened") |
| definition | "what is X", "meaning of Y"                           |
| other      | anything else                                         |

**Why it matters**: SC paper's wins come from long-CoT math/code. If SC wins
on `multi-hop` and loses on `factoid`, we have a clean story: SC helps when
the answer involves reasoning tokens, not when it's a single entity lookup.

## 3. `entity_type`
Type of the *gold* answer (what the model should produce).

| value    | examples                               |
|----------|----------------------------------------|
| person   | "Grover Cleveland"                     |
| place    | "Paris", "Mount Everest"               |
| date     | "1994", "July 4, 1776"                 |
| number   | "42", "3.14"                           |
| title    | song / book / movie names              |
| org      | companies, teams, institutions         |
| concept  | abstract nouns, definitions            |
| other    |                                        |

**Why it matters**: rare entity types (obscure titles, non-English names) tend
to get high PPL *and* low SC — so both signals agree. Common types
(dates, numbers) are where disagreement lives. Group D cases should skew here.

## 4. `notes`
Free-form. Record anything that would explain the SC/correctness combination:

- "refusal answer" (model says 'I don't know') — inflates SC artificially?
- "gold answer is actually wrong / ambiguous"
- "hallucinated a plausible but wrong entity"
- "model copied entity from question"
- "answer is correct but phrased differently than gold"
- "CoT reasoning, answer at the end"

## Target observations (what we're looking for)

When you finish, try to answer:

1. **Group B (high-SC hallucinations)**: what `answer_length` / `entity_type` dominates?
   → tells us when SC is dangerous.
2. **Group C (low-SC correct)**: are these mostly short factoid answers where
   the model actually knew the fact? → tells us SC has false positives on easy questions.
3. **Group D (SC/PPL disagreement)**: does SC or PPL match correctness better?
   Is there a question_type pattern?
4. **Overall**: does the data support "SC is better on long / multi-step answers,
   PPL is better on short factoid"? That's the thesis framing.
"""


def write_guide(path):
    with open(path, "w") as f:
        f.write(ANNOTATION_GUIDE)
    print(f"Wrote {path}")


def main():
    results = load_results()
    print(f"Loaded {len(results)} results from {RESULTS_IN}")

    groups = pick_groups(results)
    trimmed = {k: [trim(r) for r in v] for k, v in groups.items()}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(trimmed, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT_JSON}")

    with open(OUT_MD, "w") as f:
        f.write(to_markdown(trimmed))
    print(f"Wrote {OUT_MD}")

    write_guide(ROOT / "results" / "exp5_annotation_guide.md")

    print("\nGroup sizes:")
    for k, v in trimmed.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
