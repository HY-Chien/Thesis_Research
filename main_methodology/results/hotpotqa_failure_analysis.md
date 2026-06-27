# HotpotQA Failure Analysis

Date: 2026-06-23

Scope: HotpotQA dev first 1000, honest wiki-21M pool, final path-aware verifier.

Primary files:

- `results/hotpotqa_1000_verify_v2_pathaware.jsonl`
- `datasets/hotpotqa/raw/dev_first1000.jsonl`
- `results/hotpotqa_m3_wiki_detail.jsonl` for original-question retrieval with title off
- `results/hotpotqa_m3_wiki_titlefield_detail.jsonl` for decomposition retrieval with title on

## Executive Summary

HotpotQA is low because decomposition is not consistently helpful on this dataset.
In the final path-aware setting, the direct path reaches 30.3 EM, while the
decomposition path reaches only 25.0 EM. The verifier improves the final score
to 31.8 EM, but the oracle over direct/decompose choices is 40.1 EM, close to
the DeepRAG paper number of 40.7 EM.

The main pattern is different from 2Wiki and MuSiQue. HotpotQA has more cases
where direct is correct and decomposition is wrong than the reverse:

| Group | Count | Meaning |
|---|---:|---|
| both correct | 152 | direct and decomposition both answer correctly |
| direct only | 151 | original-question direct answer is correct, decomposition is wrong |
| decomp only | 98 | decomposition is correct, direct is wrong |
| both wrong | 599 | neither path answers correctly |

This means two things:

1. The current verifier still leaves about 7.4 EM of reachable direct-only
   headroom on the table.
2. The larger bottleneck is the 599 both-wrong cases, which require better
   retrieval, decomposition, or final synthesis, not just a better router.

## Router Diagnosis

Final verifier EM is 31.8. The oracle between direct and decomposition is 40.1.

| Error source among final wrong cases | Count | Share of final wrong |
|---|---:|---:|
| both paths wrong | 599 | 87.8% |
| router missed a direct-only case | 74 | 10.9% |
| router missed a decomp-only case | 9 | 1.3% |

Router capture:

| Group | Caught | Missed | Capture rate |
|---|---:|---:|---:|
| direct only | 77 | 74 | 51.0% |
| decomp only | 89 | 9 | 90.8% |

The verifier is conservative toward decomposition. This is good for 2Wiki and
MuSiQue, but it is a poor default for HotpotQA, where the direct path is often
stronger.

## Type Breakdown

HotpotQA bridge questions are the main source of damage.

| Type | N | both correct | direct only | decomp only | both wrong |
|---|---:|---:|---:|---:|---:|
| bridge | 807 | 94 | 125 | 58 | 530 |
| comparison | 193 | 58 | 26 | 40 | 69 |

Comparison questions are not the main reason HotpotQA collapses. Bridge
questions dominate the error count, and bridge direct-only cases are more than
twice as common as bridge decomp-only cases.

## Retrieval Coverage

For coverage, use full-coverage@10 as a generous proxy for the actual top-k=8
QA run. The direct path is compared with title-off original-question retrieval,
matching path-aware direct retrieval. The decomposition path is compared with
title-on sub-question B-union retrieval, matching path-aware decomposition.

| Group | A full@10 | B-union full@10 | Interpretation |
|---|---:|---:|---|
| both correct | 49.3% | 65.8% | retrieval usually sufficient |
| direct only | 45.7% | 41.1% | decomposition often loses useful support or fails to use it |
| decomp only | 28.6% | 62.2% | canonical decomposition win pattern |
| both wrong | 20.4% | 36.1% | retrieval is weak, and coverage alone is not enough |

Direct-only cross-tab:

| Direct A full@10 | Decomp B full@10 | Count | Likely cause |
|---|---|---:|---|
| no | no | 57 | both retrieval paths weak, direct may rely on model prior or partial evidence |
| yes | no | 32 | decomposition query loses support that the original question retrieved |
| no | yes | 25 | decomp retrieves support but answer/synthesis fails |
| yes | yes | 37 | retrieval is present, but decomposition/answering/synthesis fails |

Thus, direct-only errors are not only retrieval failures. A substantial share
has enough decomposed evidence but still produces the wrong answer.

## Representative Cases

### 1. Decomposition loses disambiguating context

ID: `5a85b2d95542997b5ce40028`

Question: Who was known by his stage name Aladin and helped organizations
improve their performance as a consultant?

Gold: `Eenasul Fateh`

Direct prediction: `Eenasul Fateh`

Decomposition prediction: `Ali`

Decomposition:

```text
#1: Aladin >> real name
#2: #1 >> employer
```

The original question retrieves `Eenasul Fateh` near the top because the query
contains both the stage name and the consultant clue. The decomposed first query
shrinks the context to `Aladin >> real name`, which retrieves ambiguous Aladin
articles and loses the intended person. This is a query decomposition failure:
the first sub-question drops the disambiguating constraint.

### 2. Over-decomposition turns an answerable question into a wrong chain

ID: `5a8b3af455429971feec467b`

Question: Blinding Edge Pictures produced which 2000 film starring Bruce Willis?

Gold: `Unbreakable`

Direct prediction: `Unbreakable`

Decomposition prediction: `Live Free or Die Hard`

Decomposition:

```text
#1: What is the name of the film that Blinding Edge Pictures produced that starred Bruce Willis?
#2: What is the release year of #1?
```

The first sub-question is already the original answer target, but the chain adds
an unnecessary second step about release year. Retrieval for the sub-question
drifts toward other Bruce Willis films, while the direct original query still
has enough constraints to find `Unbreakable`. This is an over-decomposition
case: a simple bridge-like question is harmed by forcing a chain.

### 3. Correct step answers, wrong final comparison synthesis

ID: `5a7bbb64554299042af8f7cc`

Question: Who is older, Annie Morton or Terry Richardson?

Gold: `Terry Richardson`

Direct prediction: `Terry Richardson`

Decomposition prediction: `Annie Morton`

Decomposition steps:

```text
#1: Annie Morton >> date of birth => October 8, 1970
#2: Terry Richardson >> date of birth => August 14, 1965
```

The decomposed retrieval has full gold coverage and the step answers are enough
to decide the comparison. The final synthesis should choose Terry Richardson
because 1965 is earlier than 1970, but it chooses Annie Morton. This is a final
synthesis/comparison reasoning error, not a retrieval error.

## Additional Observations

Some HotpotQA EM losses are answer-normalization or specificity issues rather
than genuine reasoning failures, for example:

- gold `American`, decomp `United States`
- gold `British`, decomp `United Kingdom`
- gold `Canary Islands, Spain`, decomp `Spain`

These do not explain the whole HotpotQA gap, but they slightly understate the
semantic quality of some decomposed answers.

Evidence-support verifier variants did not fix HotpotQA. Existing path-aware
evidence runs score 30.3 EM, below the no-evidence verifier's 31.8 EM. The
lexical evidence rule often supports the wrong decomposed answer because the
wrong entity or value appears in retrieved passages.

## Improvement Directions

### Priority 1: Better routing for HotpotQA-like questions

The current verifier catches only 77 of 151 direct-only cases. A diagnostic
threshold sweep over existing traces shows that flipping some
`auto_fallback_decomp` decisions to direct can raise HotpotQA from 31.8 EM to
about 33.4 to 33.6 EM.

However, the same tweak hurts other datasets badly. For example, flipping when
`direct_entropy < 1.2` and `decomp_max_post_entropy > 0.3` gives:

| Dataset | Current EM | Tweaked EM | Delta |
|---|---:|---:|---:|
| HotpotQA | 31.8 | 33.6 | +1.8 |
| 2Wiki | 46.9 | 39.1 | -7.8 |
| MuSiQue | 24.6 | 21.1 | -3.5 |
| PopQA | 51.8 | 43.1 | -8.7 |

So this should not replace the unified canonical verifier directly. It suggests
that HotpotQA needs either a dataset/domain-aware router, or a more precise
feature that detects HotpotQA-style over-decomposition without harming datasets
where decomposition is the main source of gains.

### Priority 2: Add a "do not decompose" option for simple HotpotQA bridge cases

Many HotpotQA bridge questions already contain enough lexical constraints in
the original question. The decomposer should be allowed to keep a single direct
query when the original question is highly specific and direct entropy is low.

Useful signals:

- low direct entropy
- high post-retrieval entropy in decomposed steps
- decomposed first step removes rare constraints from the original question
- decomposition adds an attribute step not needed by the answer type

### Priority 3: Preserve original-question constraints in sub-queries

The Aladin case shows a recurring issue: sub-questions reduce the query to a
short ambiguous entity name. A retrieval-side fix is to query with:

```text
sub_question + selected constraints from original_question
```

or to union the direct original-question top-k passages with the decomposed
step passages. This specifically targets the 32 direct-only cases where direct
retrieval has full coverage but decomposition retrieval does not.

### Priority 4: Improve final synthesis for comparison/date cases

Some cases have correct step answers but wrong final synthesis. Add deterministic
or prompt-level comparison handling for:

- older/younger from birth dates
- earlier/later from dates or years
- same nationality/type/location yes-no questions
- choose entity A or B after comparing step values

This should be tested separately because comparison questions are not the main
HotpotQA error mass, but they are clean and easy to fix.

Implementation note (2026-06-23): added a conservative deterministic
post-synthesis guard in `seek/pipeline.py::_rescue_value_answer`. It fires only
for clear date-comparison questions where the compared entities themselves
appear in the original question and at least two step answers contain parseable
dates/years. Offline replay on existing path-aware traces estimates:

| Dataset | Final EM before | Final EM after | Final F1 before | Final F1 after |
|---|---:|---:|---:|---:|
| HotpotQA | 31.8 | 32.1 | 41.2 | 41.7 |
| 2Wiki | 46.9 | 47.8 | 53.5 | 54.4 |
| MuSiQue | 24.6 | 24.6 | 33.5 | 33.5 |
| PopQA | 51.8 | 51.8 | 54.2 | 54.2 |

This is a small but clean fix: it addresses synthesis errors, not the larger
HotpotQA retrieval/decomposition bottleneck.

### Priority 5: Add answer alias normalization

A small normalization layer for country/demonym and location specificity could
recover some false EM negatives. This is not the main fix, but it would make the
reported score better match semantic correctness.

## Recommended Next Experiments

1. HotpotQA direct-context union ablation:
   run decomposition answering with each step seeing both sub-question top-k and
   original-question top-k evidence.

2. HotpotQA no-decompose router ablation:
   add a conservative direct fallback for low-direct-entropy, high-decomp-entropy
   cases and evaluate on HotpotQA, 2Wiki, MuSiQue, and PopQA to avoid breaking
   the unified configuration.

3. Comparison synthesis patch:
   add a deterministic or few-shot final synthesis rule for date/value
   comparisons and evaluate only comparison-type HotpotQA first.

4. Decomposition prompt or SFT patch:
   add negative examples where over-decomposition hurts, especially cases where
   the first sub-question is already the final answer target.

