# Answer Candidate Posterior Probe

This probe tests a model-based verifier signal for choosing between saved
direct and decomposed answers. It does not use the gold answer when scoring.

```text
score(a) = mean_token_log p_model(a | original question, optional context)
ACP margin = score(decomp_pred) - score(direct_pred)
p(decomp | candidates) = softmax(score(decomp), score(direct))
```

Rows per dataset: 100.
Sample mode: first.
Context-aware scoring: True.

## Diagnostic Signal

| Dataset | n | current wrong | AUROC entropy | AUROC ACP margin | direct-only ACP chooses direct | decomp-only ACP chooses decomp |
|---|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 100 | 48 | 0.530 | 0.467 | 2/2 | 17/40 |
| HotpotQA | 100 | 57 | 0.586 | 0.388 | 10/11 | 6/13 |
| MuSiQue | 100 | 75 | 0.590 | 0.474 | 1/2 | 5/14 |
| PopQA | 100 | 57 | 0.528 | 0.459 | 0/0 | 11/24 |
| ALL | 400 | 237 | 0.582 | 0.451 | 13/15 | 39/91 |

## Verifier Replay

| Policy | Dataset | EM | F1 | Direct % | Changed | Rescue/Lost |
|---|---|---:|---:|---:|---:|---:|
| current | 2WikiMultiHopQA | 52.00 | 57.85 | 8.0 | 0 | 0/0 |
| current | HotpotQA | 43.00 | 52.23 | 24.0 | 0 | 0/0 |
| current | MuSiQue | 25.00 | 35.13 | 18.0 | 0 | 0/0 |
| current | PopQA | 43.00 | 37.12 | 7.0 | 0 | 0/0 |
| acp_argmax | 2WikiMultiHopQA | 32.00 | 42.12 | 45.0 | 39 | 1/21 |
| acp_argmax | HotpotQA | 40.00 | 50.08 | 44.0 | 30 | 4/7 |
| acp_argmax | MuSiQue | 16.00 | 28.47 | 48.0 | 46 | 1/10 |
| acp_argmax | PopQA | 32.00 | 33.70 | 35.0 | 34 | 1/12 |
| acp_margin_0.05 | 2WikiMultiHopQA | 32.00 | 42.12 | 45.0 | 39 | 1/21 |
| acp_margin_0.05 | HotpotQA | 40.00 | 50.08 | 44.0 | 30 | 4/7 |
| acp_margin_0.05 | MuSiQue | 16.00 | 28.37 | 49.0 | 45 | 1/10 |
| acp_margin_0.05 | PopQA | 32.00 | 33.80 | 35.0 | 30 | 1/12 |
| acp_margin_0.10 | 2WikiMultiHopQA | 32.00 | 42.12 | 45.0 | 39 | 1/21 |
| acp_margin_0.10 | HotpotQA | 40.00 | 50.08 | 44.0 | 30 | 4/7 |
| acp_margin_0.10 | MuSiQue | 16.00 | 28.53 | 47.0 | 43 | 1/10 |
| acp_margin_0.10 | PopQA | 33.00 | 34.13 | 34.0 | 29 | 1/11 |
| acp_margin_0.20 | 2WikiMultiHopQA | 33.00 | 42.23 | 43.0 | 37 | 1/20 |
| acp_margin_0.20 | HotpotQA | 41.00 | 50.48 | 43.0 | 29 | 4/6 |
| acp_margin_0.20 | MuSiQue | 17.00 | 29.39 | 45.0 | 41 | 1/9 |
| acp_margin_0.20 | PopQA | 34.00 | 33.73 | 31.0 | 26 | 1/10 |
| acp_direct_rescue_0.10 | 2WikiMultiHopQA | 32.00 | 42.12 | 47.0 | 39 | 1/21 |
| acp_direct_rescue_0.10 | HotpotQA | 40.00 | 49.91 | 52.0 | 28 | 3/6 |
| acp_direct_rescue_0.10 | MuSiQue | 17.00 | 28.95 | 55.0 | 37 | 1/9 |
| acp_direct_rescue_0.10 | PopQA | 32.00 | 33.63 | 37.0 | 30 | 0/11 |
| acp_decomp_guard_0.10 | 2WikiMultiHopQA | 52.00 | 57.85 | 7.0 | 1 | 0/0 |
| acp_decomp_guard_0.10 | HotpotQA | 43.00 | 52.39 | 19.0 | 5 | 1/1 |
| acp_decomp_guard_0.10 | MuSiQue | 24.00 | 34.70 | 11.0 | 7 | 0/1 |
| acp_decomp_guard_0.10 | PopQA | 44.00 | 37.62 | 6.0 | 1 | 1/0 |

| Policy | Macro EM | Macro F1 | Macro Direct % |
|---|---:|---:|---:|
| current | 40.75 | 45.58 | 14.2 |
| acp_argmax | 30.00 | 38.59 | 43.0 |
| acp_margin_0.05 | 30.00 | 38.59 | 43.2 |
| acp_margin_0.10 | 30.25 | 38.71 | 42.5 |
| acp_margin_0.20 | 31.25 | 38.96 | 40.5 |
| acp_direct_rescue_0.10 | 30.25 | 38.65 | 47.8 |
| acp_decomp_guard_0.10 | 40.75 | 45.64 | 10.8 |

## ACP Argmax Rescues

| Dataset | Question | Gold | Direct | Decomp | Current | ACP | margin |
|---|---|---|---|---|---|---|---:|
| HotpotQA | What is the name of the singer who's song was released as the lead single from the album "Confessions", and that had popular song stuck behind for eight consecutive weeks? | Usher | Usher | Huey Lewis | decomp | direct | -5.558 |
| HotpotQA | Where are Teide National Park and Garajonay National Park located? | Canary Islands, Spain | Canary Islands, Spain | Spain | decomp | direct | -5.036 |
| HotpotQA | What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni? | David Weissman | David Weissman | The Family Man | decomp | direct | -3.422 |
| 2WikiMultiHopQA | Who is the paternal grandfather of Kujō Yoritsugu? | Kujō Michiie | Kujō Michiie | Kujō Kanezane | decomp | direct | -2.841 |
| MuSiQue | What is another notable work made by the author of Miss Sara Sampson? | Emilia Galotti | Emilia Galotti | Laocoön | decomp | direct | -1.471 |
| PopQA | What is Thomas R. Whitney's occupation? | politician | politician and writer | politician | direct | decomp | 1.183 |
| HotpotQA | The 2011–12 VCU Rams men's basketball team, led by third year head coach Shaka Smart, represented Virginia Commonwealth University which was founded in what year? | 1838 | 1835 | 1838 | direct | decomp | 0.312 |

## ACP Argmax Regressions

| Dataset | Question | Gold | Direct | Decomp | Current | ACP | margin |
|---|---|---|---|---|---|---|---:|
| 2WikiMultiHopQA | Are both Nikolsk Airport and Svobodny Airport located in the same country? | yes | Russia | Yes | decomp | direct | -8.875 |
| HotpotQA | Are Freakonomics and In the Realm of the Hackers both American documentaries? | no | Yes | No | decomp | direct | -8.562 |
| 2WikiMultiHopQA | Are Simonds Catholic College and Saginaw High School (Texas) both located in the same country? | no | yes | No | decomp | direct | -7.750 |
| MuSiQue | Who is the spouse of the Rabbit Hole's producer? | Keith Urban | Nicole Kidman | Keith Urban | decomp | direct | -7.098 |
| 2WikiMultiHopQA | Are both Kaufland and Otrag located in the same country? | yes | Germany | Yes | decomp | direct | -7.000 |
| PopQA | What is Ivo Perilli's occupation? | screenwriter | footballer | screenwriter | decomp | direct | -6.847 |
| HotpotQA | Were Scott Derrickson and Ed Wood of the same nationality? | yes | American | Yes | decomp | direct | -5.375 |
| HotpotQA | Are both Cypress and Ajuga genera? | no | yes | No | decomp | direct | -5.000 |
| 2WikiMultiHopQA | Where does Karin Stoltenberg's husband work at? | United Nations | Thorvald Stoltenberg | United Nations | decomp | direct | -4.353 |
| MuSiQue | What is the network which National Cycle Route 57 is part of an example of? | national cycling route network | National cycling route network | National Cycle Network | direct | decomp | 4.266 |
| 2WikiMultiHopQA | Are Fire In Hell and The Tiger: An Old Hunter'S Tale from the same country? | yes | South Korea | Yes | decomp | direct | -4.140 |
| PopQA | What is Harvey Peltier, Sr.'s occupation? | lawyer | attorney, banker, businessman, sugar grower, oilman, champion horse breeder, politician | politician | decomp | direct | -3.950 |

## Interpretation Notes

- ACP is not another rule over surface forms. It is a candidate ranking score
  induced by the model's conditional distribution over the two available
  answer candidates.
- ACP can replace only the direct-vs-decompose selection layer, not the
  decomposition generator or retriever.
- If `acp_argmax` improves routing-sensitive samples but hurts the full replay,
  ACP is better used as a soft feature inside the verifier rather than as a
  hard replacement.
