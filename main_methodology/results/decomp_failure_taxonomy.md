# Decomposition Failure Taxonomy: HotpotQA / 2Wiki / MuSiQue

Date: 2026-07-03

Offline replay from `*_1000_verify_v2_pathaware.jsonl` + m3 coverage detail files. HotpotQA is included as a consistency check against `results/hotpotqa_failure_analysis.md`; 2Wiki and MuSiQue are the new datasets. Categories are defined in `scripts/analyze_decomp_failure_taxonomy.py`.

## TL;DR

1. **The dominant decomposition-path failure everywhere is retrieval, not the decomposer**: `retrieval_gap_both` (neither the original question nor the sub-question union reaches full gold coverage @10) accounts for 47% (HotpotQA) / 64% (2Wiki) / 68% (MuSiQue) of decomposition-path errors.

2. **Constraint loss (sub-questions dropping disambiguating context) is a HotpotQA-specific failure**: 10% on HotpotQA vs 1% / 2% on 2Wiki / MuSiQue. The MuSiQue-style KG-triple decomposition hurts precisely where the original question already carries strong lexical constraints — which is why the original-question evidence union helps HotpotQA most (+5.5 EM).

3. **Synthesis failure is a stable 15-18% slice on all three datasets**: some step answer already matches gold but the synthesized final answer is wrong. This is the mass addressed by the synthesis guards, and it caps what decomposition alone can deliver even with perfect retrieval.

4. **The verifier is decompose-biased by construction**: it catches 91-95% of decomp-only wins on every dataset but only 24-51% of direct-only wins. That is the right trade on 2Wiki/MuSiQue (decOnly >> dirOnly) and the wrong one on HotpotQA (dirOnly 151 > decOnly 98).


---

## hotpotqa (n=1000)

### Direct/decompose 2x2 and router capture

| group | n | share |
|---|---:|---:|
| bothC | 152 | 15.2% |
| dirOnly | 151 | 15.1% |
| decOnly | 98 | 9.8% |
| bothW | 599 | 59.9% |

Router capture: direct-only 77/151 (51%), decomp-only 89/98 (91%).

Error sources among final wrong answers:

| source | n | share of final wrong |
|---|---:|---:|
| both paths wrong | 599 | 87.8% |
| router missed direct-only | 74 | 10.9% |
| router missed decomp-only | 9 | 1.3% |

### Retrieval coverage by group (full-coverage@10)

A = original question, title-off (direct path); B-union = sub-questions, title-on (decomp path).

| group | joined n | A full@10 | B-union full@10 |
|---|---:|---:|---:|
| bothC | 152 | 49.3% | 65.8% |
| dirOnly | 151 | 45.7% | 41.1% |
| decOnly | 98 | 28.6% | 62.2% |
| bothW | 599 | 20.4% | 36.1% |

### Failure taxonomy of decomposition-path errors (decomp_em = 0)

| primary category | dirOnly | bothW | total |
|---|---:|---:|---:|
| synthesis_failure | 25 (17%) | 87 (15%) | 112 (15%) |
| constraint_loss | 31 (21%) | 42 (7%) | 73 (10%) |
| answer_despite_cov | 46 (30%) | 164 (27%) | 210 (28%) |
| retrieval_gap_both | 49 (32%) | 306 (51%) | 355 (47%) |
| no_coverage_join | 0 (0%) | 0 (0%) | 0 (0%) |

Orthogonal flags (can overlap with any primary category):

| flag | dirOnly | bothW |
|---|---:|---:|
| intermediate_echo | 39 (26%) | 132 (22%) |
| subq1_restates_q | 21 (14%) | 110 (18%) |

### Per-type 2x2

| type | n | bothC | dirOnly | decOnly | bothW |
|---|---:|---:|---:|---:|---:|
| bridge | 807 | 94 | 125 | 58 | 530 |
| comparison | 193 | 58 | 26 | 40 | 69 |

### Representative examples

**synthesis_failure**

- [dirOnly] Q: What screenwriter with credits for "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
  - gold: `David Weissman` | direct: `David Weissman` | decomp: `David Diamond`
  - chain: Evolution >> screenwriter => amateur screenwriter | amateur screenwriter co-wrote a film starring Nicolas Cage and Téa Leoni => David Diamond
- [dirOnly] Q: Which of Tara Strong major voice role in animated series is an American animated television series based on the DC Comics fictional superhero team, the "Teen Titans"?
  - gold: `Teen Titans Go!` | direct: `Teen Titans Go!` | decomp: `Teen Titans`
  - chain: What are some of Tara Strong's major voice roles in animated series? => Rugrats, The Powerpuff Girls, The Fairly OddParents, Drawn Together, Ben 10, Chowder, Teen... | Which of Rugrats, The Powerpuff Girls, The Fairly OddParents, Drawn Together, Ben 10, Chow... => Teen Titans

**constraint_loss**

- [dirOnly] Q: Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?
  - gold: `Eenasul Fateh` | direct: `Eenasul Fateh` | decomp: `Ali`
  - chain: Aladin >> real name => Ali | Ali >> employer => does not carry the inherent right to violate labor laws.
- [dirOnly] Q: What WB supernatrual drama series was Jawbreaker star Rose Mcgowan best known for being in?
  - gold: `Charmed` | direct: `Charmed` | decomp: `E! Network`
  - chain: What series is Jawbreaker a part of? => E! Network | E! Network >> cast member => None

**answer_despite_cov**

- [dirOnly] Q: Who is older, Annie Morton or Terry Richardson?
  - gold: `Terry Richardson` | direct: `Terry Richardson` | decomp: `Annie Morton`
  - chain: Annie Morton >> date of birth => October 8, 1970 | Terry Richardson >> date of birth => August 14, 1965
- [dirOnly] Q: What was the father of Kasper Schmeichel voted to be by the IFFHS in 1992?
  - gold: `World's Best Goalkeeper` | direct: `World's Best Goalkeeper` | decomp: `Peter Schmeichel`
  - chain: Kasper Schmeichel >> father => Peter Schmeichel | Peter Schmeichel >> IFFHS World's Best Goalkeeper => 1992, 1993

**retrieval_gap_both**

- [dirOnly] Q: What science fantasy young adult series, told in first person, has a set of companion books narrating the stories of enslaved worlds and alien species?
  - gold: `Animorphs` | direct: `Animorphs` | decomp: `[4]`
  - chain: What science fantasy young adult series is told in first person? => [4] | [4] has a set of companion books that narrate the stories of enslaved worlds and alien spe... => Master Rules and Immortal Rules
- [dirOnly] Q: 2014 S/S is the debut album of a South Korean boy group that was formed by who?
  - gold: `YG Entertainment` | direct: `YG Entertainment` | decomp: `WINNER`
  - chain: 2014 S/S >> performer => WINNER | WINNER >> record label => Winner Records


---

## 2wiki (n=1000)

### Direct/decompose 2x2 and router capture

| group | n | share |
|---|---:|---:|
| bothC | 125 | 12.5% |
| dirOnly | 55 | 5.5% |
| decOnly | 347 | 34.7% |
| bothW | 473 | 47.3% |

Router capture: direct-only 13/55 (24%), decomp-only 331/347 (95%).

Error sources among final wrong answers:

| source | n | share of final wrong |
|---|---:|---:|
| both paths wrong | 473 | 89.1% |
| router missed direct-only | 42 | 7.9% |
| router missed decomp-only | 16 | 3.0% |

### Retrieval coverage by group (full-coverage@10)

A = original question, title-off (direct path); B-union = sub-questions, title-on (decomp path).

| group | joined n | A full@10 | B-union full@10 |
|---|---:|---:|---:|
| bothC | 125 | 26.4% | 46.4% |
| dirOnly | 55 | 14.5% | 20.0% |
| decOnly | 347 | 14.4% | 46.4% |
| bothW | 473 | 9.5% | 28.1% |

### Failure taxonomy of decomposition-path errors (decomp_em = 0)

| primary category | dirOnly | bothW | total |
|---|---:|---:|---:|
| synthesis_failure | 11 (20%) | 80 (17%) | 91 (17%) |
| constraint_loss | 0 (0%) | 4 (1%) | 4 (1%) |
| answer_despite_cov | 4 (7%) | 90 (19%) | 94 (18%) |
| retrieval_gap_both | 40 (73%) | 299 (63%) | 339 (64%) |
| no_coverage_join | 0 (0%) | 0 (0%) | 0 (0%) |

Orthogonal flags (can overlap with any primary category):

| flag | dirOnly | bothW |
|---|---:|---:|
| intermediate_echo | 1 (2%) | 21 (4%) |
| subq1_restates_q | 10 (18%) | 140 (30%) |

### Per-type 2x2

| type | n | bothC | dirOnly | decOnly | bothW |
|---|---:|---:|---:|---:|---:|
| compositional | 428 | 30 | 20 | 89 | 289 |
| bridge_comparison | 228 | 11 | 9 | 121 | 87 |
| comparison | 216 | 77 | 19 | 83 | 37 |
| inference | 128 | 7 | 7 | 54 | 60 |

### Representative examples

**synthesis_failure**

- [dirOnly] Q: Where was the wife of Douglas Leiterman born?
  - gold: `Winnipeg, Manitoba` | direct: `Winnipeg, Manitoba` | decomp: `Winnipeg`
  - chain: Douglas Leiterman >> spouse => Beryl Fox | Beryl Fox >> place of birth => Winnipeg
- [dirOnly] Q: Who is the paternal grandfather of Kujō Yoritsugu?
  - gold: `Kujō Michiie` | direct: `Kujō Michiie` | decomp: `Fujiwara no Tadamichi`
  - chain: Kujō Yoritsugu >> father => Kujō Michiie | Kujō Michiie >> father => Fujiwara no Tadamichi

**constraint_loss**

- [bothW] Q: What is the place of birth of the performer of song Damage (Pharoahe Monch Song)?
  - gold: `Queens` | direct: `Brooklyn` | decomp: `Brisbane`
  - chain: Damage >> performer => DJ Damage | DJ Damage >> place of birth => Brisbane
- [bothW] Q: What is the cause of death of composer of song Look At Me (Xxxtentacion Song)?
  - gold: `shot` | direct: `murder` | decomp: `plane crash`
  - chain: Look at Me! >> composer => James Horner | James Horner >> cause of death => plane crash

**answer_despite_cov**

- [dirOnly] Q: Which film was released first, Ekam – Son Of Soil or Farber'S Nerve?
  - gold: `Farber'S Nerve` | direct: `Farber's Nerve` | decomp: `Ekam – Son of Soil`
  - chain: Ekam – Son of Soil >> publication date => 23 April 2010 | Farber's Nerve >> publication date => 2006
- [dirOnly] Q: Where was the place of death of James Adam (Architect)'s father?
  - gold: `Edinburgh` | direct: `Edinburgh` | decomp: `Caracas, Venezuela`
  - chain: James Adam >> father => William Newton Adams, Jr. | William Newton Adams, Jr. >> place of death => Caracas, Venezuela

**retrieval_gap_both**

- [dirOnly] Q: Was Şemsettin Baş or Gwenc'Hlan Le Scouëzec born first?
  - gold: `Gwenc'Hlan Le Scouëzec` | direct: `Gwenc'hlan Le Scouëzec` | decomp: `6 BC`
  - chain: Şemsettin Baş >> date of birth => 6 BC | Gwenc'hlan Le Scouëzec >> date of birth => 11 November 1929
- [dirOnly] Q: Where was the performer of song God (John Lennon Song) born?
  - gold: `Liverpool` | direct: `Liverpool` | decomp: `Los Angeles`
  - chain: God >> performer => God Is a Woman | God Is a Woman >> place of birth => Los Angeles


---

## musique (n=1000)

### Direct/decompose 2x2 and router capture

| group | n | share |
|---|---:|---:|
| bothC | 65 | 6.5% |
| dirOnly | 37 | 3.7% |
| decOnly | 182 | 18.2% |
| bothW | 716 | 71.6% |

Router capture: direct-only 10/37 (27%), decomp-only 171/182 (94%).

Error sources among final wrong answers:

| source | n | share of final wrong |
|---|---:|---:|
| both paths wrong | 716 | 95.0% |
| router missed direct-only | 27 | 3.6% |
| router missed decomp-only | 11 | 1.5% |

### Retrieval coverage by group (full-coverage@10)

A = original question, title-off (direct path); B-union = sub-questions, title-on (decomp path).

| group | joined n | A full@10 | B-union full@10 |
|---|---:|---:|---:|
| bothC | 65 | 21.5% | 46.2% |
| dirOnly | 37 | 16.2% | 35.1% |
| decOnly | 182 | 6.6% | 45.1% |
| bothW | 716 | 5.4% | 19.1% |

### Failure taxonomy of decomposition-path errors (decomp_em = 0)

| primary category | dirOnly | bothW | total |
|---|---:|---:|---:|
| synthesis_failure | 8 (22%) | 126 (18%) | 134 (18%) |
| constraint_loss | 2 (5%) | 10 (1%) | 12 (2%) |
| answer_despite_cov | 8 (22%) | 84 (12%) | 92 (12%) |
| retrieval_gap_both | 19 (51%) | 496 (69%) | 515 (68%) |
| no_coverage_join | 0 (0%) | 0 (0%) | 0 (0%) |

Orthogonal flags (can overlap with any primary category):

| flag | dirOnly | bothW |
|---|---:|---:|
| intermediate_echo | 4 (11%) | 62 (9%) |
| subq1_restates_q | 10 (27%) | 237 (33%) |

### Per-type 2x2

| type | n | bothC | dirOnly | decOnly | bothW |
|---|---:|---:|---:|---:|---:|
| 2hop | 1000 | 65 | 37 | 182 | 716 |

### Representative examples

**synthesis_failure**

- [dirOnly] Q: What is the network which National Cycle Route 57 is part of an example of?
  - gold: `national cycling route network` | direct: `National cycling route network` | decomp: `National Cycle Network`
  - chain: National Cycle Route 57 >> part of => National Cycle Network | National Cycle Network >> instance of => national cycling route network
- [dirOnly] Q: Who is the brother of the artist who made Nothin' My Love Can't Fix's album?
  - gold: `Matthew Lawrence` | direct: `Matthew Lawrence` | decomp: `Andy Lawrence`
  - chain: Nothin' My Love Can't Fix >> performer => Joey Lawrence | Joey Lawrence >> sibling => Andy Lawrence

**constraint_loss**

- [dirOnly] Q: Into what harbour does the river Lennox Bridge in Parramatta crosses flow?
  - gold: `Sydney Harbour` | direct: `Port Jackson` | decomp: `Berowra Creek`
  - chain: Parramatta >> located in the administrative territorial entity => New South Wales | New South Wales >> located in or next to body of water => [1] Berowra Waters, New South Wales >> Berowra Creek [2] Tomago, New South Wales >> Hunter...
- [dirOnly] Q: When did the first mosque open in the place where Tobolar Copra plant is located?
  - gold: `September 2012` | direct: `September 2012` | decomp: `1884`
  - chain: Tobolar Copra >> located in the administrative territorial entity => Madang Province | when was the first mosque built in Madang Province => 1884

**answer_despite_cov**

- [dirOnly] Q: The Move Ya Body song's band is named after who?
  - gold: `Natalie Albino` | direct: `Nina Sky` | decomp: `the sky's limit`
  - chain: Move Ya Body >> performer => Nina Sky | Nina Sky >> named after => the sky's the limit
- [dirOnly] Q: The Nicole and Natalie album's band is named after who?
  - gold: `Natalie Albino` | direct: `Nina Sky` | decomp: `the sky's limit`
  - chain: Nicole and Natalie >> performer => Nina Sky | Nina Sky >> named after => the sky's the limit

**retrieval_gap_both**

- [dirOnly] Q: What is another notable work made by the author of Miss Sara Sampson?
  - gold: `Emilia Galotti` | direct: `Emilia Galotti` | decomp: `Laocoön`
  - chain: Miss Sara Sampson >> author => Gotthold Ephraim Lessing | Gotthold Ephraim Lessing >> notable work => Laocoön
- [dirOnly] Q: Who is the spouse of the creator of The Nanny Reunion: A Nosh to Remember?
  - gold: `Peter Marc Jacobson` | direct: `Peter Marc Jacobson` | decomp: `Husband`
  - chain: Who created The Nanny Reunion: A Nosh to Remember? => Lifetime | Lifetime >> spouse => Husband


---

## Cross-dataset summary (decomp-path errors, dirOnly+bothW pooled)

| category | hotpotqa | 2wiki | musique |
|---|---:|---:|---:|
| synthesis_failure | 112 (15%) | 91 (17%) | 134 (18%) |
| constraint_loss | 73 (10%) | 4 (1%) | 12 (2%) |
| answer_despite_cov | 210 (28%) | 94 (18%) | 92 (12%) |
| retrieval_gap_both | 355 (47%) | 339 (64%) | 515 (68%) |
| no_coverage_join | 0 (0%) | 0 (0%) | 0 (0%) |
| flag: intermediate_echo | 171 (23%) | 22 (4%) | 66 (9%) |
| flag: subq1_restates_q | 131 (17%) | 150 (28%) | 247 (33%) |
