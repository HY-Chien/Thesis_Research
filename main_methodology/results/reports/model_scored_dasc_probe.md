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

Selection: up to 40 low-entropy direct cases per dataset, balanced
between official direct-wrong and direct-correct cases when possible.
Context-aware scoring: True.

## Summary

| Dataset | n | wrong | AUROC entropy | AUROC F1-role | AUROC SC-role | AUROC logprob-role | AUROC F1-DASC | AUROC SC-DASC | AUROC logprob-DASC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 40 | 20 | 0.410 | 0.912 | 0.830 | 0.825 | 0.695 | 0.608 | 0.568 |
| HotpotQA | 40 | 20 | 0.802 | 0.476 | 0.550 | 0.398 | 0.565 | 0.580 | 0.590 |
| MuSiQue | 40 | 20 | 0.375 | 0.800 | 0.838 | 0.645 | 0.593 | 0.620 | 0.438 |
| ALL | 120 | 60 | 0.500 | 0.730 | 0.742 | 0.643 | 0.616 | 0.629 | 0.537 |

## Top Self-Certainty Role-Risk Wrong Examples

| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | sc_role_risk |
|---|---|---|---|---|---|---:|---:|
| 2WikiMultiHopQA | Which country the director of film Ata Pata Laapata is from? | Indian | India | Ata Pata Laapata >> director | Rajpal Yadav >> country of citizenship | 0.010 | 2.924 |
| 2WikiMultiHopQA | Who is the paternal grandfather of Sultan Al-Dawla? | 'Adud al-Dawla | Baha' al-Dawla | Sultan al-Dawla >> father | Who is the paternal grandfather of Sultan Al-Dawla? | 0.039 | 2.808 |
| 2WikiMultiHopQA | Where was the husband of Jeanne Raunay born? | Évreux | André Beaunier | Jeanne Raunay >> spouse | Where was the husband of Jeanne Raunay born? | 0.060 | 2.520 |
| MuSiQue | Into what river does the river that Thuli River turns into flow? | Limpopo River | Shashe River | What river does Thuli River turn into? | Into what river does the river that Thuli River turns into flow? | 0.011 | 2.407 |
| HotpotQA | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | reigning monarch of the United Kingdom | Buckingham Palace | Where is Waterloo Vase used as a garden ornament? | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | 0.001 | 2.326 |
| MuSiQue | Of which river is the river Kichi Sipi Bridge crosses a tributary? | Hudson's Bay | Nelson River | Kichi Sipi Bridge >> crosses | Of which river is the river Kichi Sipi Bridge crosses a tributary? | 0.021 | 2.125 |
| 2WikiMultiHopQA | Who is the father of the director of film Wonders Of The Sea 3D? | Jacques Cousteau | Colin Low | Wonders of the Sea 3D >> director | Brian DeLeeuw >> father | 0.037 | 2.109 |
| MuSiQue | Who is part of the band which performed Cherchez le Garçon? | Daniel Darc | Taxi Girl | Cherchez le Garçon >> performer | Who is part of the band which performed Cherchez le Garçon? | 0.014 | 1.188 |
| 2WikiMultiHopQA | Who is William Ii, Count Of Flanders's paternal grandmother? | Mathilde of Bourbon | Margaret II of Flanders | William II, Count of Flanders >> father | William I of Flanders >> mother | 0.019 | 1.029 |
| HotpotQA | Which North Central town in Florida is home to the punk rock band Against Me!? | Gainesville, Florida | Gainesville | Against Me! >> location of formation | Which North Central town in Florida is home to the punk rock band Against Me!? | 0.007 | 0.969 |
| HotpotQA | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | Apsley George Benet Cherry-Garrard | Apsley Cherry-Garrard | The Worst Journey in the World >> author | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | 0.001 | 0.698 |
| 2WikiMultiHopQA | Who is Grimoald The Younger's paternal grandfather? | Ansegisel | Pepin of Herstal | Grimoald the Younger >> father | Who is Grimoald The Younger's paternal grandfather? | 0.003 | 0.675 |
| 2WikiMultiHopQA | Who is the paternal grandfather of Eochaid Ailtlethan? | Connla Cáem | Ailill Caisfiaclach | Eochaid Ailtlethan >> father | Who is the paternal grandfather of Eochaid Ailtlethan? | 0.030 | 0.641 |
| MuSiQue | Who is the grandfather of James Mayer de Rothschild? | Amschel Moses Rothschild | Mayer Amschel Rothschild | James Mayer de Rothschild >> father | Who is the grandfather of James Mayer de Rothschild? | 0.022 | 0.607 |
| MuSiQue | What is the anime that has Seta Sōjirō an instance of? | OVA | Rurouni Kenshin | Seta Sōjirō >> present in work | What is the anime that has Seta Sōjirō an instance of? | 0.007 | 0.340 |
| 2WikiMultiHopQA | Who is Maria Luisa Of Savoy's father-in-law? | Louis, Grand Dauphin | Philip V of Spain | Maria Luisa of Savoy >> spouse | Who is Maria Luisa Of Savoy's father-in-law? | 0.043 | 0.222 |

## Top Logprob Role-Risk Wrong Examples

| Dataset | Question | Gold | Direct pred | Best intermediate question | Best final question | entropy | logprob_role_risk |
|---|---|---|---|---|---|---:|---:|
| MuSiQue | Who is part of the band which performed Cherchez le Garçon? | Daniel Darc | Taxi Girl | Cherchez le Garçon >> performer | Who is part of the band which performed Cherchez le Garçon? | 0.014 | 0.758 |
| 2WikiMultiHopQA | Who is the paternal grandfather of Sultan Al-Dawla? | 'Adud al-Dawla | Baha' al-Dawla | Sultan al-Dawla >> father | Who is the paternal grandfather of Sultan Al-Dawla? | 0.039 | 0.046 |
| 2WikiMultiHopQA | Where was the husband of Jeanne Raunay born? | Évreux | André Beaunier | Jeanne Raunay >> spouse | Where was the husband of Jeanne Raunay born? | 0.060 | 0.010 |
| MuSiQue | What is the anime that has Seta Sōjirō an instance of? | OVA | Rurouni Kenshin | Seta Sōjirō >> present in work | What is the anime that has Seta Sōjirō an instance of? | 0.007 | 0.002 |
| MuSiQue | Who was the sibling of Nannina de' Medici? | Giuliano de' Medici | Lorenzo de' Medici | Nannina de' Medici >> sibling | Who was the sibling of Nannina de' Medici? | 0.005 | 0.001 |
| MuSiQue | What was the original name of the bank that bought FleetBoston Financial? | Bank of Italy | Bank of America | What bank bought FleetBoston Financial? | What was the original name of the bank that bought FleetBoston Financial? | 0.023 | 0.000 |
| 2WikiMultiHopQA | Who is Tekle Haymanot Ii's paternal grandfather? | Iyasu I | Yohannes II | Tekle Haymanot II >> father | Who is Tekle Haymanot Ii's paternal grandfather? | 0.043 | 0.000 |
| MuSiQue | Who is the grandfather of James Mayer de Rothschild? | Amschel Moses Rothschild | Mayer Amschel Rothschild | James Mayer de Rothschild >> father | Who is the grandfather of James Mayer de Rothschild? | 0.022 | 0.000 |
| HotpotQA | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | Apsley George Benet Cherry-Garrard | Apsley Cherry-Garrard | The Worst Journey in the World >> author | The Worst Journey in the World is a memoir written by which English explorer of Antarctica? | 0.001 | 0.000 |
| 2WikiMultiHopQA | Who is the paternal grandfather of Eochaid Ailtlethan? | Connla Cáem | Ailill Caisfiaclach | Eochaid Ailtlethan >> father | Who is the paternal grandfather of Eochaid Ailtlethan? | 0.030 | -0.000 |
| MuSiQue | Into what river does the river that Thuli River turns into flow? | Limpopo River | Shashe River | What river does Thuli River turn into? | Into what river does the river that Thuli River turns into flow? | 0.011 | -0.000 |
| HotpotQA | Which North Central town in Florida is home to the punk rock band Against Me!? | Gainesville, Florida | Gainesville | Against Me! >> location of formation | Which North Central town in Florida is home to the punk rock band Against Me!? | 0.007 | -0.000 |
| 2WikiMultiHopQA | Who is Grimoald The Younger's paternal grandfather? | Ansegisel | Pepin of Herstal | Grimoald the Younger >> father | Who is Grimoald The Younger's paternal grandfather? | 0.003 | -0.000 |
| HotpotQA | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | Cynthia Ann Stephanie "Cyndi" Lauper | Cyndi Lauper | What artist released the debut album "She's So Unusual"? | The True Colors World Tour was by what artist whose debut album was "She's So Unusual"? | 0.001 | -0.000 |
| HotpotQA | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | reigning monarch of the United Kingdom | Buckingham Palace | Where is Waterloo Vase used as a garden ornament? | The Waterloo Vase has been used as a garden ornament at whose residence and administrative headquarters? | 0.001 | -0.000 |
| MuSiQue | Of which river is the river Kichi Sipi Bridge crosses a tributary? | Hudson's Bay | Nelson River | Kichi Sipi Bridge >> crosses | Of which river is the river Kichi Sipi Bridge crosses a tributary? | 0.021 | -0.000 |

## Caveats

- This is still a small diagnostic run, not the final metric.
- The model-scored role signal uses no gold answer, but it depends on the
  quality of the decomposition trace.
- Pure self-certainty measures distribution sharpness, not necessarily answer
  compatibility. The log-probability role score is included as a QA-specific
  companion signal.
- Some comparison or yes/no questions may still need an explicit final-answer
  synthesis score, because the final answer is not always a sub-question answer.
