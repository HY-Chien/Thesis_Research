# Derivation-Aware Self-Certainty Probe

This is a small diagnostic experiment for a QA adaptation of self-certainty.
It does not use hand-written answer-type rules. Instead, it checks whether
a direct answer is closer to an intermediate answer in the decomposition
trace or to the final decomposed answer.

Score:

```text
role_risk = max_F1(direct_pred, intermediate step answers)
            - max_F1(direct_pred, final step answer / decomp prediction)
DASC risk = z(direct entropy) + z(role_risk)
```

Higher score means higher risk of a wrong direct answer. The low-entropy
flagged column focuses on confident direct answers whose prediction looks
like an intermediate derivation result.

## Relation to Self-Certainty

The original Self-Certainty paper proposes using model-internal certainty
to rank sampled responses instead of relying only on majority voting:
https://arxiv.org/abs/2502.18581.

This probe keeps the same motivation but adapts it to QA. A QA answer
should not only be high-certainty; it should be high-certainty as the
final answer to the original question, not merely as an intermediate
answer in the reasoning chain. Therefore, this first experiment adds
a derivation-role margin on top of entropy/self-certainty.

## Summary

| Dataset | n | direct acc | AUROC entropy | AUROC role_risk | AUROC DASC | low-ent wrong | flagged low-ent wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 1000 | 18.0 | 0.671 | 0.860 | 0.866 | 139 / 200 | 25 |
| HotpotQA | 1000 | 30.3 | 0.743 | 0.655 | 0.765 | 79 / 200 | 5 |
| MuSiQue | 1000 | 10.2 | 0.711 | 0.818 | 0.857 | 156 / 200 | 27 |
| PopQA | 1000 | 37.8 | 0.780 | 0.707 | 0.813 | 37 / 200 | 0 |
| ALL | 4000 | 24.1 | 0.758 | 0.770 | 0.802 | 411 / 800 | 57 |

## Flagged Examples

| Dataset | Question | Gold | Direct pred | Intermediate match | Decomp final | entropy | role_risk |
|---|---|---|---|---|---|---:|---:|
| 2WikiMultiHopQA | Who is the paternal grandfather of Eochaid Ailtlethan? | Connla Cáem | Ailill Caisfiaclach | Ailill Caisfiaclach | Eochaid Ailtlethan | 0.030 | 1.00 |
| 2WikiMultiHopQA | Where was the husband of Jeanne Raunay born? | Évreux | André Beaunier | André Beaunier | Paris | 0.060 | 1.00 |
| 2WikiMultiHopQA | Do John Amyas Alexander and Albert Seward share the same nationality? | yes | British | British | Yes | 0.164 | 1.00 |
| 2WikiMultiHopQA | Which film has the director who died earlier, 11 Harrowhouse or A Little Sister Of Everybody? | A Little Sister Of Everybody | Robert Thornby | Robert Thornby | 11 Harrowhouse | 0.194 | 1.00 |
| 2WikiMultiHopQA | Which film has the director who died later, Swarg Narak or Antardhan? | Swarg Narak | Dasari Narayana Rao | Dasari Narayana Rao | Swarg Narak | 0.196 | 1.00 |
| HotpotQA |  Sir Johnston Forbes-Robertson was considered the finest at playing a character who was the king of which country ? | Prince of Denmark | Hamlet | Hamlet | Denmark | 0.008 | 1.00 |
| HotpotQA | In what city can a person find both the Nusretiye Clock Tower and the Eski Imaret Mosque be found? | Istanbul, Turkey | Istanbul | Istanbul | Beyoğlu | 0.022 | 1.00 |
| HotpotQA | What was the proper name of the husband of Lollia Paullina? | Gaius Julius Caesar Augustus Germanicus | Caligula | Caligula | Gaius Julius Caesar Augustus Germanicus | 0.032 | 1.00 |
| HotpotQA | Mamie Gummer played the role of Nancy Crozier on the TV series airing on what network? | CBS | The Good Wife | The Good Wife | CBS | 0.033 | 1.00 |
| HotpotQA | Who is still with the group that covered "Leather-Winged Bat" in 1969? | Yarrow and Stookey | Peter, Paul and Mary | Peter, Paul and Mary | Peter Yarrow | 0.020 | 0.67 |
| MuSiQue | Who did the spouse of Hagar marry after the death of Sarah? | Keturah | Abraham | Abraham | Keturah | 0.008 | 1.00 |
| MuSiQue | What is the river where Long Spruce Generating Station is located a tributary of? | Hudson's Bay | Nelson River | Nelson River | Lake Winnipeg | 0.019 | 1.00 |
| MuSiQue | Of which river is the river Kichi Sipi Bridge crosses a tributary? | Hudson's Bay | Nelson River | Nelson River | Lake Winnipeg | 0.021 | 1.00 |
| MuSiQue | What does the river that Darling Mills Creek turns into flow into? | Sydney Harbour | Parramatta River | Parramatta River | Port Jackson | 0.027 | 1.00 |
| MuSiQue | Into what does the river for which Toongabbie Creek serves as the mouth flow? | Sydney Harbour | Parramatta River | Parramatta River | Port Jackson | 0.038 | 1.00 |

## Interpretation

- `role_risk` is a derivation-role signal, not an answer-type rule.
- It catches cases where the model confidently outputs a bridge or evidence value.
- It will miss alias/semantic cases where lexical F1 cannot connect related forms,
  such as `American` vs `United States`.
- A full QA self-certainty version should replace token-F1 matching with
  model-scored certainty of candidate answers under the original question and
  each decomposition step.
