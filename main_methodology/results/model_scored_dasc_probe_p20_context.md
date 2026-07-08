# Model-Scored DASC Probe

This is a small QA adaptation of Self-Certainty on low-entropy direct-answer cases.
Unlike the first DASC-lite probe, this version does not use token-F1 to define the
derivation-role score. It teacher-forces the direct prediction under the original
question, final sub-question, and intermediate sub-questions, then measures how
sharp the model's token distribution is while emitting the same candidate answer.

```text
self_certainty(y | q) = mean_i KL(U || p_model(. | q, y_<i))
sc_role_risk = max self_certainty(y | intermediate sub-question)
               - max self_certainty(y | original/final question)
logprob_role_risk = max log p(y | intermediate sub-question)
                    - max log p(y | original/final question)
model_DASC = z(direct_entropy) + z(sc_role_risk)
logprob_DASC = z(direct_entropy) + z(logprob_role_risk)
```

Selection: up to 20 low-entropy direct cases per dataset, balanced
between official direct-wrong and direct-correct cases when possible.
Context-aware scoring: True.

## Summary

| Dataset | n | wrong | AUROC entropy | AUROC F1-role | AUROC SC-role | AUROC logprob-role | AUROC F1-DASC | AUROC SC-DASC | AUROC logprob-DASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 20 | 10 | 0.440 | 0.930 | 0.810 | 0.770 | 0.740 | 0.630 | 0.560 |
| HotpotQA | 20 | 10 | 0.830 | 0.285 | 0.510 | 0.360 | 0.360 | 0.550 | 0.590 |
| MuSiQue | 20 | 10 | 0.520 | 0.820 | 0.870 | 0.680 | 0.710 | 0.710 | 0.600 |
| ALL | 60 | 30 | 0.548 | 0.668 | 0.720 | 0.589 | 0.620 | 0.668 | 0.589 |

## Top Self-Certainty Role-Risk Wrong Examples

| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | sc_role_risk |
|---|---|---|---|---|---|---:|---:|
| 2WikiMultiHopQA | Which country the director of film Ata Pata Laapata is from? | Indian | India | Ata Pata Laapata >> director | Rajpal Yadav >> country of citizenship | 0.010 | 2.924 |
| HotpotQA | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | reigning monarch of the United Kingdom | Buckingham Palace | Where is Waterloo Vase used as a garden ornament? | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | 0.001 | 2.326 |
| 2WikiMultiHopQA | Who is William Ii, Count Of Flanders's paternal grandmother? | Mathilde of Bourbon | Margaret II of Flanders | William II, Count of Flanders >> father | William I of Flanders >> mother | 0.019 | 1.029 |
| HotpotQA | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | Apsley George Benet Cherry-Garrard | Apsley Cherry-Garrard | The Worst Journey in the World >> author | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | 0.001 | 0.698 |
| 2WikiMultiHopQA | Who is Grimoald The Younger's paternal grandfather? | Ansegisel | Pepin of Herstal | Grimoald the Younger >> father | Who is Grimoald The Younger's paternal grandfather? | 0.003 | 0.675 |
| MuSiQue | What is the anime that has Seta Sōjirō an instance of? | OVA | Rurouni Kenshin | Seta Sōjirō >> present in work | What is the anime that has Seta Sōjirō an instance of? | 0.007 | 0.340 |
| MuSiQue | Who was the first African American student at the university Robert Khayat was educated at? | James Howard Meredith | Gene Gray | Robert Khayat >> educated at | Who was the first African American student at the university Robert Khayat was educated at? | 0.002 | 0.137 |
| 2WikiMultiHopQA | Who is Blanche Of Portugal (1259–1321)'s paternal grandmother? | Urraca of Castile | Beatrice of Castile | Blanche of Portugal >> father | Afonso III of Portugal >> mother | 0.004 | 0.088 |
| HotpotQA | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | Cynthia Ann Stephanie "Cyndi" Lauper | Cyndi Lauper | What artist released the debut album "She's So Unusual"? | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | 0.001 | 0.019 |
| 2WikiMultiHopQA | Who is the paternal grandmother of King Kang Of Zhou? | Yi Jiang | Arsinoe II | King Kang of Zhou >> father | Who is the paternal grandmother of King Kang Of Zhou? | 0.019 | -0.062 |
| MuSiQue | Who was the sibling of Nannina de' Medici? | Giuliano de' Medici | Lorenzo de' Medici | Nannina de' Medici >> sibling | Who was the sibling of Nannina de' Medici? | 0.005 | -0.163 |
| HotpotQA | Which American music star appeared as a guest in a Will & Grace episode called "Back Up, Dancer"? | Janet Damita Jo Jackson | Janet Jackson | Which American music star appeared as a guest in an episode called "Back Up, Dancer"? | Which American music star appeared as a guest in a Will & Grace episode called "Back Up, Dancer"? | 0.002 | -0.194 |
| MuSiQue | What actor had the role of the narrator of How I Met Your Mother? | Josh Radnor | Bob Saget | who narrates how i met your mother | What actor had the role of the narrator of How I Met Your Mother? | 0.005 | -0.569 |
| 2WikiMultiHopQA | Who is the father-in-law of Tsandzile Ndwandwe? | Ndvungunye | Sobhuza I | Tsandzile Ndwandwe >> spouse | Who is the father-in-law of Tsandzile Ndwandwe? | 0.021 | -0.841 |
| 2WikiMultiHopQA | Who is the paternal grandmother of Ingegerd Olofsdotter Of Sweden? | Sigrid the Haughty | Estrid of the Obotrites | Ingegerd Olofsdotter of Sweden >> father | Who is the paternal grandmother of Ingegerd Olofsdotter Of Sweden? | 0.010 | -0.857 |
| HotpotQA | What party does the House of Commons representative for Congleton belong to? | British Conservative Party | Conservative | Congleton >> member of parliament | What party does the House of Commons representative for Congleton belong to? | 0.004 | -1.370 |

## Top Logprob Role-Risk Wrong Examples

| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | logprob_role_risk |
|---|---|---|---|---|---|---:|---:|
| MuSiQue | What is the anime that has Seta Sōjirō an instance of? | OVA | Rurouni Kenshin | Seta Sōjirō >> present in work | What is the anime that has Seta Sōjirō an instance of? | 0.007 | 0.002 |
| MuSiQue | Who was the sibling of Nannina de' Medici? | Giuliano de' Medici | Lorenzo de' Medici | Nannina de' Medici >> sibling | Who was the sibling of Nannina de' Medici? | 0.005 | 0.001 |
| HotpotQA | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | Apsley George Benet Cherry-Garrard | Apsley Cherry-Garrard | The Worst Journey in the World >> author | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | 0.001 | 0.000 |
| 2WikiMultiHopQA | Who is Grimoald The Younger's paternal grandfather? | Ansegisel | Pepin of Herstal | Grimoald the Younger >> father | Who is Grimoald The Younger's paternal grandfather? | 0.003 | -0.000 |
| HotpotQA | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | Cynthia Ann Stephanie "Cyndi" Lauper | Cyndi Lauper | What artist released the debut album "She's So Unusual"? | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | 0.001 | -0.000 |
| HotpotQA | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | reigning monarch of the United Kingdom | Buckingham Palace | Where is Waterloo Vase used as a garden ornament? | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | 0.001 | -0.000 |
| HotpotQA | Which American music star appeared as a guest in a Will & Grace episode called "Back Up, Dancer"? | Janet Damita Jo Jackson | Janet Jackson | Which American music star appeared as a guest in an episode called "Back Up, Dancer"? | Which American music star appeared as a guest in a Will & Grace episode called "Back Up, Dancer"? | 0.002 | -0.011 |
| MuSiQue | Who did the spouse of Hagar marry after the death of Sarah? | Keturah | Abraham | Hagar >> spouse | Who did the spouse of Hagar marry after the death of Sarah? | 0.008 | -0.046 |
| HotpotQA | What american actress/singer born in 1956 recorded the song Something there for beauty and the beast? | Donna Paige Helmintoller | Paige O'Hara | who sings something there in beauty and the beast | What american actress/singer born in 1956 recorded the song Something there for beauty and the beast? | 0.003 | -0.049 |
| 2WikiMultiHopQA | Who is the father-in-law of Tsandzile Ndwandwe? | Ndvungunye | Sobhuza I | Tsandzile Ndwandwe >> spouse | Who is the father-in-law of Tsandzile Ndwandwe? | 0.021 | -0.137 |
| MuSiQue | What actor had the role of the narrator of How I Met Your Mother? | Josh Radnor | Bob Saget | who narrates how i met your mother | What actor had the role of the narrator of How I Met Your Mother? | 0.005 | -0.470 |
| HotpotQA | Which of Kylian Hazard's older brothers plays for the German Bundesliga club Borussia Mönchengladbach and the Belgium national team. | Thorgan Ganael Francis Hazard | Thorgan Hazard | Kylian Hazard >> sibling | Which of Kylian Hazard's older brothers plays for the German Bundesliga club Borussia Mönchengladbach and the Belgium national team. | 0.003 | -0.710 |
| 2WikiMultiHopQA | Who is William Ii, Count Of Flanders's paternal grandmother? | Mathilde of Bourbon | Margaret II of Flanders | William II, Count of Flanders >> father | William I of Flanders >> mother | 0.019 | -1.135 |
| 2WikiMultiHopQA | Where did Henry C. Lord's father graduate from? | Bowdoin College | Dartmouth College | Henry C. Lord >> father | Where did Henry C. Lord's father graduate from? | 0.013 | -1.680 |
| 2WikiMultiHopQA | Who is Blanche Of Portugal (1259–1321)'s paternal grandmother? | Urraca of Castile | Beatrice of Castile | Blanche of Portugal >> father | Who is Blanche Of Portugal (1259–1321)'s paternal grandmother? | 0.004 | -1.730 |
| MuSiQue | Who was Suzanne Farrington's mother's spouse? | Laurence Olivier | Herbert Leigh Holman | Suzanne Farrington >> mother | Who was Suzanne Farrington's mother's spouse? | 0.008 | -2.450 |

## Caveats

- This is still a small diagnostic run, not the final metric.
- The model-scored role signal uses no gold answer, but it depends on the
  quality of the decomposition trace.
- Pure self-certainty measures distribution sharpness, not necessarily answer
  compatibility. The log-probability role score is included as a QA-specific
  companion signal.
- Some comparison or yes/no questions may still need an explicit final-answer
  synthesis score, because the final answer is not always a sub-question answer.
