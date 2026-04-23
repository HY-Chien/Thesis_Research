# Exp5 Annotation Guide

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
