# Gate Margin Pilot 100

Date: 2026-06-30

Scope: first 100 examples per dataset, open-domain `wiki` corpus, mode C, path-aware title default, same LoRA adapter as the canonical SEEK gate runs. This pilot compares the existing entropy gate against two margin-style replacements at `tau=0.3`.

Signals:

- `entropy@0.3`: existing `max_ent` gate from `results/{ds}_3way_gate.jsonl`.
- `maqa_margin@0.3`: first generated answer token top-1/top-2 softmax probability margin; retrieve when margin <= tau.
- `targ_unc@0.3`: prefix-mean `exp(-raw_logit_gap / beta)` with `prefix_k=8`, `beta=1.0`; retrieve when uncertainty >= tau.

| Dataset | Signal | n | EM | F1(alias) | retr% | retr steps |
|---|---|---:|---:|---:|---:|---:|
| hotpotqa | entropy@0.3 | 100 | 31.0 | 39.8 | 89.7 | 210/234 |
| hotpotqa | maqa_margin@0.3 | 100 | 31.0 | 37.9 | 49.6 | 116/234 |
| hotpotqa | targ_unc@0.3 | 100 | 27.0 | 33.3 | 28.2 | 66/234 |
| 2wiki | entropy@0.3 | 100 | 50.0 | 55.3 | 93.9 | 231/246 |
| 2wiki | maqa_margin@0.3 | 100 | 42.0 | 46.1 | 62.6 | 154/246 |
| 2wiki | targ_unc@0.3 | 100 | 43.0 | 47.2 | 42.3 | 104/246 |
| musique | entropy@0.3 | 100 | 20.0 | 32.1 | 94.5 | 190/201 |
| musique | maqa_margin@0.3 | 100 | 10.0 | 22.7 | 53.2 | 107/201 |
| musique | targ_unc@0.3 | 100 | 8.0 | 21.8 | 31.3 | 63/201 |
| popqa | entropy@0.3 | 100 | 40.0 | 45.0 | 100.0 | 100/100 |
| popqa | maqa_margin@0.3 | 100 | 44.0 | 47.7 | 76.0 | 76/100 |
| popqa | targ_unc@0.3 | 100 | 42.0 | 45.7 | 60.0 | 60/100 |
| ALL | entropy@0.3 | 100x4 | 35.2 | 43.0 | 94.5 | - |
| ALL | maqa_margin@0.3 | 100x4 | 31.8 | 38.6 | 60.4 | - |
| ALL | targ_unc@0.3 | 100x4 | 30.0 | 37.0 | 40.5 | - |

## Takeaway

At the same nominal threshold (`tau=0.3`), neither margin signal improves the overall gate result. Both signals retrieve much less often than entropy, which helps PopQA slightly but hurts the multi-hop datasets, especially MuSiQue. The result suggests these metrics are not drop-in replacements for `max_ent` at the existing threshold.

## Threshold Calibration Note

Using the pilot signal distributions, matching the entropy gate retrieval rate (~94.5%) would require approximately:

- `maqa_margin`: `tau ~= 0.972` for retrieve-if-margin-<=tau.
- `targ_unc`: `tau ~= 0.008` for retrieve-if-uncertainty->=tau.

Those thresholds are close to always-retrieve behavior. A meaningful follow-up would be a cost-accuracy sweep, not replacing entropy with `tau=0.3` directly.

## Files

Pilot traces:

- `results/gate_pilot100_{dataset}_maqa-margin_tau0p3.jsonl`
- `results/gate_pilot100_{dataset}_targ-unc_tau0p3.jsonl`

Logs:

- `scripts/gate_pilot100_{dataset}_maqa-margin_tau0p3.log`
- `scripts/gate_pilot100_{dataset}_targ-unc_tau0p3.log`
