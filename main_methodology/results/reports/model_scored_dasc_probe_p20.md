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

## Summary

| Dataset | n | wrong | AUROC entropy | AUROC F1-role | AUROC SC-role | AUROC logprob-role | AUROC F1-DASC | AUROC SC-DASC | AUROC logprob-DASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 20 | 10 | 0.440 | 0.930 | 0.730 | 0.660 | 0.740 | 0.590 | 0.540 |
| HotpotQA | 20 | 10 | 0.830 | 0.285 | 0.540 | 0.540 | 0.360 | 0.650 | 0.600 |
| MuSiQue | 20 | 10 | 0.520 | 0.820 | 0.680 | 0.590 | 0.710 | 0.580 | 0.530 |
| ALL | 60 | 30 | 0.548 | 0.668 | 0.648 | 0.581 | 0.620 | 0.603 | 0.568 |

## Top Self-Certainty Role-Risk Wrong Examples

| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | sc_role_risk |
|---|---|---|---|---|---|---:|---:|
| HotpotQA | What american actress/singer born in 1956 recorded the song Something there for beauty and the beast? | Donna Paige Helmintoller | Paige O'Hara | Paige O'Hara and Robby Benson >> place of birth | What american actress/singer born in 1956 recorded the song Something there for beauty and the beast? | 0.003 | 4.807 |
| MuSiQue | Who was the first African American student at the university Robert Khayat was educated at? | James Howard Meredith | Gene Gray | Robert Khayat >> educated at | Who was the first African American student at the university Robert Khayat was educated at? | 0.002 | 3.317 |
| MuSiQue | What record label is the performer who released All Your Faded Things on? | Kill Rock Stars | Cold Crush Records | All Your Faded Things >> performer | What record label is the performer who released All Your Faded Things on? | 0.001 | 2.609 |
| MuSiQue | What actor had the role of the narrator of How I Met Your Mother? | Josh Radnor | Bob Saget | who narrates how i met your mother | What actor had the role of the narrator of How I Met Your Mother? | 0.005 | 1.302 |
| HotpotQA | Which American music star appeared as a guest in a Will & Grace episode called "Back Up, Dancer"? | Janet Damita Jo Jackson | Janet Jackson | Which American music star appeared as a guest in an episode called "Back Up, Dancer"? | Janet Jackson >> present in work | 0.002 | 1.002 |
| 2WikiMultiHopQA | Who is William Ii, Count Of Flanders's paternal grandmother? | Mathilde of Bourbon | Margaret II of Flanders | William II, Count of Flanders >> father | Who is William Ii, Count Of Flanders's paternal grandmother? | 0.019 | 0.768 |
| HotpotQA | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | Cynthia Ann Stephanie "Cyndi" Lauper | Cyndi Lauper | What artist released the debut album "She's So Unusual"? | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | 0.001 | 0.682 |
| 2WikiMultiHopQA | Who is the paternal grandmother of King Kang Of Zhou? | Yi Jiang | Arsinoe II | King Kang of Zhou >> father | Who is the paternal grandmother of King Kang Of Zhou? | 0.019 | 0.435 |
| 2WikiMultiHopQA | Who is Grimoald The Younger's paternal grandfather? | Ansegisel | Pepin of Herstal | Grimoald the Younger >> father | Who is Grimoald The Younger's paternal grandfather? | 0.003 | -0.175 |
| 2WikiMultiHopQA | Who is the paternal grandmother of Ingegerd Olofsdotter Of Sweden? | Sigrid the Haughty | Estrid of the Obotrites | Ingegerd Olofsdotter of Sweden >> father | Who is the paternal grandmother of Ingegerd Olofsdotter Of Sweden? | 0.010 | -0.274 |
| HotpotQA | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | Apsley George Benet Cherry-Garrard | Apsley Cherry-Garrard | The Worst Journey in the World >> author | Apsley Cherry-Garrard >> country of citizenship | 0.001 | -0.338 |
| MuSiQue | Who was the sibling of Nannina de' Medici? | Giuliano de' Medici | Lorenzo de' Medici | Nannina de' Medici >> sibling | Who was the sibling of Nannina de' Medici? | 0.005 | -0.545 |
| 2WikiMultiHopQA | Who is the father-in-law of Tsandzile Ndwandwe? | Ndvungunye | Sobhuza I | Tsandzile Ndwandwe >> spouse | Ngwane V >> father | 0.021 | -0.582 |
| MuSiQue | Who was Suzanne Farrington's mother's spouse? | Laurence Olivier | Herbert Leigh Holman | Suzanne Farrington >> mother | Vivien Leigh >> spouse | 0.008 | -1.737 |
| HotpotQA | Which of Kylian Hazard's older brothers plays for the German Bundesliga club Borussia Mönchengladbach and the Belgium national team. | Thorgan Ganael Francis Hazard | Thorgan Hazard | Kylian Hazard >> sibling | Which of Kylian Hazard's older brothers plays for the German Bundesliga club Borussia Mönchengladbach and the Belgium national team. | 0.003 | -1.903 |
| 2WikiMultiHopQA | Where was the father of Augustus William, Duke Of Brunswick-Bevern born? | Osterholz | Braunschweig | Augustus William, Duke of Brunswick-Bevern >> father | Ferdinand Albert I, Duke of Brunswick-Lüneburg >> place of birth | 0.007 | -2.119 |

## Top Logprob Role-Risk Wrong Examples

| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | logprob_role_risk |
|---|---|---|---|---|---|---:|---:|
| HotpotQA | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | Apsley George Benet Cherry-Garrard | Apsley Cherry-Garrard | The Worst Journey in the World >> author | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | 0.001 | 0.022 |
| HotpotQA | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | Cynthia Ann Stephanie "Cyndi" Lauper | Cyndi Lauper | What artist released the debut album "She's So Unusual"? | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | 0.001 | 0.006 |
| MuSiQue | What actor had the role of the narrator of How I Met Your Mother? | Josh Radnor | Bob Saget | who narrates how i met your mother | What actor had the role of the narrator of How I Met Your Mother? | 0.005 | -0.030 |
| MuSiQue | Who did the spouse of Hagar marry after the death of Sarah? | Keturah | Abraham | Hagar >> spouse | Who did the spouse of Hagar marry after the death of Sarah? | 0.008 | -0.118 |
| MuSiQue | What is the anime that has Seta Sōjirō an instance of? | OVA | Rurouni Kenshin | Seta Sōjirō >> present in work | What is the anime that has Seta Sōjirō an instance of? | 0.007 | -0.142 |
| MuSiQue | Who was the sibling of Nannina de' Medici? | Giuliano de' Medici | Lorenzo de' Medici | Nannina de' Medici >> sibling | Who was the sibling of Nannina de' Medici? | 0.005 | -0.163 |
| HotpotQA | What american actress/singer born in 1956 recorded the song Something there for beauty and the beast? | Donna Paige Helmintoller | Paige O'Hara | Paige O'Hara and Robby Benson >> place of birth | What american actress/singer born in 1956 recorded the song Something there for beauty and the beast? | 0.003 | -0.195 |
| HotpotQA | Which American music star appeared as a guest in a Will & Grace episode called "Back Up, Dancer"? | Janet Damita Jo Jackson | Janet Jackson | Which American music star appeared as a guest in an episode called "Back Up, Dancer"? | Janet Jackson >> present in work | 0.002 | -0.210 |
| 2WikiMultiHopQA | Who is Grimoald The Younger's paternal grandfather? | Ansegisel | Pepin of Herstal | Grimoald the Younger >> father | Pepin of Herstal >> father | 0.003 | -0.316 |
| HotpotQA | Which of Kylian Hazard's older brothers plays for the German Bundesliga club Borussia Mönchengladbach and the Belgium national team. | Thorgan Ganael Francis Hazard | Thorgan Hazard | Kylian Hazard >> sibling | Which of Kylian Hazard's older brothers plays for the German Bundesliga club Borussia Mönchengladbach and the Belgium national team. | 0.003 | -0.424 |
| HotpotQA | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | reigning monarch of the United Kingdom | Buckingham Palace | Where is Waterloo Vase used as a garden ornament? | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | 0.001 | -0.866 |
| 2WikiMultiHopQA | Who is the father-in-law of Tsandzile Ndwandwe? | Ndvungunye | Sobhuza I | Tsandzile Ndwandwe >> spouse | Ngwane V >> father | 0.021 | -0.978 |
| 2WikiMultiHopQA | Who is William Ii, Count Of Flanders's paternal grandmother? | Mathilde of Bourbon | Margaret II of Flanders | William II, Count of Flanders >> father | Who is William Ii, Count Of Flanders's paternal grandmother? | 0.019 | -1.340 |
| 2WikiMultiHopQA | Where was the father of Augustus William, Duke Of Brunswick-Bevern born? | Osterholz | Braunschweig | Augustus William, Duke of Brunswick-Bevern >> father | Where was the father of Augustus William, Duke Of Brunswick-Bevern born? | 0.007 | -2.170 |
| 2WikiMultiHopQA | Who is the paternal grandmother of Ingegerd Olofsdotter Of Sweden? | Sigrid the Haughty | Estrid of the Obotrites | Ingegerd Olofsdotter of Sweden >> father | Olof Skötkonung >> mother | 0.010 | -2.207 |
| 2WikiMultiHopQA | Who is Blanche Of Portugal (1259–1321)'s paternal grandmother? | Urraca of Castile | Beatrice of Castile | Blanche of Portugal >> father | Who is Blanche Of Portugal (1259–1321)'s paternal grandmother? | 0.004 | -2.400 |

## Caveats

- This is still a small diagnostic run, not the final metric.
- The model-scored role signal uses no gold answer, but it depends on the
  quality of the decomposition trace.
- Pure self-certainty measures distribution sharpness, not necessarily answer
  compatibility. The log-probability role score is included as a QA-specific
  companion signal.
- Some comparison or yes/no questions may still need an explicit final-answer
  synthesis score, because the final answer is not always a sub-question answer.
