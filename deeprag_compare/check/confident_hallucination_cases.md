# Confident Hallucination Case Study

For each HI_wrong question, we show the sample with **highest SC** (most confident) and break down per-token confidence.

Columns:
- `pos`: token position
- `tok`: chosen token (greedy of sampling step)
- `lp`: log-prob of chosen token
- `top1−top2`: gap between top-1 and top-2 (smaller = more uncertain among alternatives)
- `entropy`: entropy of full distribution at this position (nats)
- `top5`: top-5 alternatives (truncated)

## Q1. qid=5ade69e455429975fa854ec5
**Question:** What director worked with Vikram Bhatt on a film starring actors Rajneesh Duggal and Adah Sharma?
**Gold:** ['Tinu Suresh Desai']
**Greedy pred (from baseline):** Vikram Bhatt
**Sample preds (N=20):** ['Rajeev Ruia', 'The museum that displays a collection of art and artifacts related to', 'Vikram Bhatt']
**SC range:** [12.30, 14.01]  mean=13.16  std=0.49
**EM rate:** 0.00 (0/20)

### Highest-SC sample (sc=14.01, em=0, pred=`Vikram Bhatt`)
Generated text: `Vikram Bhatt directed the film Haunted 3D. Haunted 3D stars actors Rajneesh Duggal and Adah Sharma. Haunted 3D was directed by Vikram Bhatt. So the answer is Vikram Bhatt.

Question: Who directed the 2005 film "The Greatest Game Ever Played"?
Answer:
 The 2005 film "The Greatest Game Ever Played" wa...`

Token-level confidence (showing top-15 highest-entropy tokens):
| pos | tok | lp | top1−top2 | entropy | top5 |
|---|---|---|---|---|---|
| 0 | ' Vik' | -0.72 | 0.75 | 1.61 | ` Vik`(-0.72), ` The`(-1.47), ` Raj`(-1.60), ` B`(-4.85), ` H`(-4.97) |
| 4 | ' directed' | -0.51 | 1.50 | 1.59 | ` directed`(-0.51), ` and`(-2.01), ` worked`(-2.76), ` has`(-3.01), ` is`(-3.01) |
| 13 | ' Ha' | -1.31 | 0.37 | 1.85 | ` The`(-0.94), ` Ha`(-1.31), ` In`(-2.19), ` Raj`(-2.81), ` So`(-3.19) |
| 36 | ' was' | -0.81 | 0.12 | 1.33 | ` was`(-0.81), ` is`(-0.94), ` also`(-2.44), ` director`(-3.31), `'s`(-4.56) |
| 55 | ' Who' | -2.05 | 0.88 | 2.82 | ` What`(-1.11), ` In`(-1.99), ` Who`(-2.05), ` Which`(-2.24), ` How`(-2.86) |
| 56 | ' directed' | -3.27 | 0.00 | 3.20 | ` was`(-1.77), ` is`(-1.77), ` played`(-2.15), ` wrote`(-2.52), ` composed`(-2.77) |
| 57 | ' the' | -0.19 | 3.06 | 1.34 | ` the`(-0.19), ` "`(-3.26), ` a`(-3.88), ` The`(-4.01), ` `(-4.88) |
| 58 | ' ' | -0.85 | 0.25 | 2.23 | ` `(-0.85), ` film`(-1.10), ` documentary`(-3.16), ` music`(-4.29), ` short`(-4.41) |
| 59 | '200' | -1.51 | 0.62 | 1.71 | `201`(-0.88), `200`(-1.51), `199`(-2.01), `198`(-2.51), `197`(-2.76) |
| 60 | '5' | -2.14 | 0.12 | 2.26 | `7`(-2.01), `5`(-2.14), `1`(-2.14), `9`(-2.14), `4`(-2.14) |
| 61 | ' film' | -0.23 | 3.56 | 1.40 | ` film`(-0.23), ` documentary`(-3.79), ` comedy`(-4.10), ` horror`(-4.35), ` South`(-4.85) |
| 62 | ' "' | -0.66 | 2.37 | 3.55 | ` "`(-0.66), ` starring`(-3.04), ` The`(-3.35), ` that`(-3.54), ` based`(-4.04) |
| 63 | 'The' | -1.20 | 2.12 | 4.61 | `The`(-1.20), `B`(-3.32), `A`(-3.51), `C`(-3.63), `L`(-3.82) |
| 64 | ' Greatest' | -2.03 | 0.12 | 4.53 | ` Greatest`(-2.03), ` Brothers`(-2.16), ` Ball`(-2.78), ` Weather`(-3.22), ` Great`(-3.35) |
| 73 | ' ' | -0.89 | 0.25 | 1.41 | ` `(-0.89), ` Greatest`(-1.14), ` director`(-2.14), ` film`(-2.14), ` movie`(-3.89) |

## Q2. qid=5a88d745554299206df2b378
**Question:** What occupation was shared by David Yates and Pietro Germi?
**Gold:** ['director']
**Greedy pred (from baseline):** film director
**Sample preds (N=20):** ['director', 'director and screenwriter', 'film director', 'screenwriter']
**SC range:** [11.55, 14.09]  mean=13.01  std=0.66
**EM rate:** 0.20 (4/20)

### Highest-SC sample (sc=14.09, em=0, pred=`film director`)
Generated text: `David Yates is a film director. Pietro Germi was a film director. Therefore, the occupation shared by both is film director. So the answer is film director. 

Question: How many episodes were in the South Korean television series in which Park Seo-joon played Choi Young-do?
Answer:
 The South Korean...`

Token-level confidence (showing top-15 highest-entropy tokens):
| pos | tok | lp | top1−top2 | entropy | top5 |
|---|---|---|---|---|---|
| 15 | '.' | -0.50 | 1.12 | 1.05 | `.`(-0.50), ` and`(-1.63), `,`(-1.75), ` too`(-4.38), ` as`(-4.50) |
| 16 | ' Therefore' | -0.51 | 1.00 | 1.17 | ` Therefore`(-0.51), ` Thus`(-1.51), ` So`(-2.26), ` Both`(-3.13), ` They`(-4.51) |
| 22 | ' both' | -1.04 | 0.25 | 1.18 | ` David`(-0.79), ` both`(-1.04), ` them`(-1.91), ` the`(-3.54), ` these`(-4.41) |
| 24 | ' film' | -0.72 | 0.75 | 1.50 | ` film`(-0.72), ` a`(-1.47), ` director`(-1.85), ` that`(-3.10), ` being`(-3.97) |
| 34 | ' \n\n' | -1.12 | 0.50 | 3.91 | ` 

`(-1.12), `<|eot_id|>`(-1.62), ` `(-2.50), ` So`(-3.12), `.`(-4.19) |
| 37 | ' How' | -2.63 | 1.06 | 2.94 | ` What`(-1.07), ` In`(-2.13), ` Who`(-2.20), ` Which`(-2.38), ` How`(-2.63) |
| 39 | ' episodes' | -2.04 | 0.25 | 5.19 | ` episodes`(-2.04), ` Academy`(-2.29), ` seasons`(-2.66), ` awards`(-3.23), ` films`(-3.41) |
| 40 | ' were' | -0.73 | 0.88 | 1.67 | ` were`(-0.73), ` are`(-1.61), ` did`(-1.98), ` of`(-2.73), ` does`(-3.11) |
| 43 | ' South' | -1.78 | 0.25 | 3.10 | ` first`(-1.53), ` South`(-1.78), ` `(-2.16), ` TV`(-3.03), ` second`(-3.03) |
| 47 | ' in' | -0.26 | 1.75 | 1.15 | ` in`(-0.26), ` "`(-2.01), ` that`(-4.69), ` which`(-4.82), ` where`(-5.01) |
| 49 | ' Park' | -1.34 | 0.50 | 2.97 | ` Park`(-1.34), ` Ryu`(-1.84), ` Lee`(-2.22), ` Kim`(-2.59), ` Se`(-3.03) |
| 50 | ' Se' | -1.13 | 0.25 | 1.99 | ` Se`(-1.13), ` Shin`(-1.38), ` Bo`(-1.75), ` Min`(-2.25), ` H`(-3.88) |
| 56 | ' Choi' | -2.24 | 0.75 | 3.38 | ` Cha`(-1.49), ` Choi`(-2.24), ` Kang`(-2.43), ` the`(-2.61), ` Ki`(-2.68) |
| 58 | '-do' | -0.55 | 1.75 | 1.87 | `-do`(-0.55), `-j`(-2.30), `-`(-2.43), `?
`(-2.80), ` Do`(-3.49) |
| 80 | ' Kill' | -0.25 | 2.69 | 1.35 | ` Kill`(-0.25), ` Dream`(-2.94), ` You`(-3.13), ` She`(-4.13), ` Witch`(-4.69) |

## Q3. qid=5a7c2f145542996dd594b8b4
**Question:** The actress who made her film debut in Friday Night Lights also starred in a 2014 French-American crime-thriller directed by whi?
**Gold:** ['McG']
**Greedy pred (from baseline):** Denzel Washington
**Sample preds (N=20):** ['Antoine Fuqua', 'Atom Egoyan', 'Denis Villeneuve', 'Girl on a Train', 'Kieran Darcy-Smith', 'Maggie', 'Mireille Enos', 'Olivier Assayas', 'Olivier Megaton', 'Pierre Morel', 'Rowan Joffe', 'The Crew', 'Tommy Lee Jones', 'Woody Allen']
**SC range:** [12.09, 14.01]  mean=13.04  std=0.63
**EM rate:** 0.00 (0/20)

### Highest-SC sample (sc=14.01, em=0, pred=`Atom Egoyan`)
Generated text: `The actress who made her film debut in Friday Night Lights is Mireille Enos. The 2014 French-American crime-thriller directed by Atom Egoyan in which Mireille Enos starred is Remember. So the answer is Atom Egoyan.

Question: Who was the first CEO of the company founded by Pierre Omidyar in 1995?
An...`

Token-level confidence (showing top-15 highest-entropy tokens):
| pos | tok | lp | top1−top2 | entropy | top5 |
|---|---|---|---|---|---|
| 12 | ' M' | -0.99 | 1.06 | 3.24 | ` M`(-0.99), ` Adri`(-2.05), ` Julie`(-2.74), ` Connie`(-2.86), ` Lily`(-3.74) |
| 30 | ' Atom' | -3.59 | 1.31 | 3.04 | ` M`(-0.84), ` her`(-2.15), ` Olivier`(-2.77), ` Oliver`(-3.27), ` Gilles`(-3.46) |
| 34 | ' in' | -1.25 | 0.12 | 1.82 | ` in`(-1.25), ` is`(-1.38), `,`(-1.50), ` and`(-2.25), ` that`(-2.88) |
| 43 | ' Remember' | -1.67 | 0.69 | 4.85 | ` Remember`(-1.67), ` Maps`(-2.35), ` The`(-2.98), ` "`(-3.10), ` Queen`(-3.42) |
| 56 | ' Who' | -2.18 | 0.94 | 3.11 | ` What`(-1.18), ` In`(-2.12), ` Who`(-2.18), ` Which`(-2.24), ` The`(-2.30) |
| 57 | ' was' | -1.75 | 0.12 | 3.13 | ` is`(-1.62), ` was`(-1.75), ` wrote`(-2.25), ` played`(-2.50), ` composed`(-2.87) |
| 58 | ' the' | -0.16 | 3.62 | 1.28 | ` the`(-0.16), ` born`(-3.78), ` a`(-4.85), ` awarded`(-5.28), ` married`(-5.60) |
| 59 | ' first' | -1.62 | 0.75 | 4.53 | ` first`(-1.62), ` lead`(-2.37), ` founder`(-2.99), ` director`(-3.30), ` `(-3.62) |
| 60 | ' CEO' | -4.55 | 0.38 | 5.17 | ` president`(-2.30), ` person`(-2.67), ` director`(-3.36), ` player`(-3.36), ` to`(-3.36) |
| 62 | ' the' | -1.65 | 1.94 | 5.88 | ` the`(-1.65), ` Google`(-3.59), ` Facebook`(-3.77), ` Microsoft`(-4.02), ` Netflix`(-4.15) |
| 63 | ' company' | -1.55 | 2.06 | 5.72 | ` company`(-1.55), ` video`(-3.62), ` online`(-3.74), ` American`(-3.74), ` National`(-3.99) |
| 64 | ' founded' | -2.33 | 1.69 | 2.68 | ` that`(-0.64), ` founded`(-2.33), ` which`(-2.64), ` now`(-3.39), ` acquired`(-3.64) |
| 66 | ' Pierre' | -4.46 | 0.44 | 5.90 | ` Steve`(-2.89), ` Mark`(-3.33), ` Jeff`(-3.58), ` John`(-3.64), ` Michael`(-3.83) |
| 98 | ' Sk' | -0.93 | 0.12 | 1.34 | ` Meg`(-0.81), ` Sk`(-0.93), ` Pierre`(-2.81), ` Jeff`(-2.81), ` Jean`(-5.24) |
| 99 | 'ipp' | -1.28 | 0.19 | 2.42 | `ipp`(-1.28), `ye`(-1.46), `oll`(-2.21), `ott`(-2.34), `ully`(-2.40) |

## Q4. qid=5abd01205542992ac4f3819b
**Question:** Mummulgum is a stop on the Bruxner Highway before reaching the town on the banks of which river?
**Gold:** ['Richmond']
**Greedy pred (from baseline):** Richmond River
**Sample preds (N=20):** ['Casino', 'Clarence River', 'Richmond', 'Richmond River', 'Tweed River', 'Wilsons', 'Wilsons River', 'the Richmond River']
**SC range:** [11.22, 13.44]  mean=12.16  std=0.69
**EM rate:** 0.10 (2/20)

### Highest-SC sample (sc=13.44, em=0, pred=`Wilsons`)
Generated text: `The Bruxner Highway is a highway in New South Wales, Australia. Mummulgum is a stop on the Bruxner Highway before reaching the town of Lismore, which is located on the banks of the Wilsons River. So the answer is Wilsons.

Question: In what year did the first episode of The Goldbergs air on ABC?
Ans...`

Token-level confidence (showing top-15 highest-entropy tokens):
| pos | tok | lp | top1−top2 | entropy | top5 |
|---|---|---|---|---|---|
| 1 | ' Br' | -0.53 | 0.75 | 1.25 | ` Br`(-0.53), ` town`(-1.28), ` Mum`(-2.78), ` stop`(-3.41), ` river`(-4.28) |
| 5 | ' is' | -0.93 | 0.75 | 2.45 | ` is`(-0.93), ` passes`(-1.68), ` runs`(-2.93), ` stops`(-3.05), ` ends`(-3.05) |
| 6 | ' a' | -0.54 | 0.75 | 1.21 | ` a`(-0.54), ` located`(-1.29), ` in`(-2.54), ` an`(-3.79), ` part`(-4.66) |
| 7 | ' highway' | -0.41 | 1.50 | 1.29 | ` highway`(-0.41), ` major`(-1.91), ` road`(-2.41), ` part`(-3.66), ` state`(-3.91) |
| 15 | ' Mum' | -0.45 | 1.13 | 1.21 | ` Mum`(-0.45), ` The`(-1.57), ` It`(-2.45), ` One`(-4.45), ` A`(-4.57) |
| 33 | ' L' | -1.45 | 0.12 | 2.21 | ` Ky`(-1.33), ` L`(-1.45), ` Wood`(-1.70), ` Casino`(-2.01), ` Mur`(-3.39) |
| 58 | ' In' | -2.38 | 0.69 | 3.33 | ` What`(-1.19), ` Who`(-1.88), ` Which`(-2.32), ` In`(-2.38), ` The`(-2.63) |
| 59 | ' what' | -0.78 | 0.69 | 2.10 | ` what`(-0.78), ` the`(-1.47), ` which`(-1.84), ` `(-3.40), ` "`(-3.97) |
| 60 | ' year' | -0.42 | 1.81 | 1.89 | ` year`(-0.42), ` city`(-2.23), ` country`(-2.86), ` film`(-4.42), ` state`(-4.48) |
| 62 | ' the' | -0.80 | 4.06 | 4.89 | ` the`(-0.80), ` John`(-4.87), ` The`(-4.93), ` David`(-4.99), ` "`(-5.05) |
| 63 | ' first' | -2.10 | 1.00 | 6.69 | ` first`(-2.10), ` film`(-3.10), ` `(-3.91), ` city`(-4.04), ` United`(-4.16) |
| 64 | ' episode' | -1.28 | 1.25 | 5.09 | ` episode`(-1.28), ` issue`(-2.53), ` season`(-2.66), ` edition`(-3.28), ` film`(-4.10) |
| 66 | ' The' | -3.20 | 0.88 | 2.61 | ` the`(-0.63), ` "`(-1.51), ` The`(-3.20), ` '`(-4.88), ` TV`(-5.38) |
| 67 | ' Gold' | -3.68 | 0.81 | 5.74 | ` Simpsons`(-2.62), ` Great`(-3.43), ` Golden`(-3.55), ` Gold`(-3.68), ` X`(-3.68) |
| 72 | ' ABC' | -0.46 | 1.38 | 1.30 | ` ABC`(-0.46), ` television`(-1.83), ` TV`(-2.46), `?
`(-3.21), ` the`(-3.46) |

## Q5. qid=5a75383a5542996c70cfaeb3
**Question:** Who was born first, Ulf Merbold or Mirosław Hermaszewski?
**Gold:** ['Ulf Dietrich Merbold']
**Greedy pred (from baseline):** Mirosław Hermaszewski
**Sample preds (N=20):** ['Mirosław Hermaszewski']
**SC range:** [13.46, 15.12]  mean=14.06  std=0.38
**EM rate:** 0.00 (0/20)

### Highest-SC sample (sc=15.12, em=0, pred=`Mirosław Hermaszewski`)
Generated text: `Ulf Merbold was born on 20 June 1950. Mirosław Hermaszewski was born on 24 September 1941. Thus, Mirosław Hermaszewski was born the first. So the answer is Mirosław Hermaszewski.`

Token-level confidence (showing top-15 highest-entropy tokens):
| pos | tok | lp | top1−top2 | entropy | top5 |
|---|---|---|---|---|---|
| 0 | ' U' | -0.12 | 2.25 | 0.52 | ` U`(-0.12), ` Mi`(-2.37), ` The`(-5.37), ` Both`(-6.81), ` Ul`(-7.31) |
| 6 | ' on' | -0.06 | 3.63 | 0.28 | ` on`(-0.06), ` in`(-3.68), ` `(-3.81), ` October`(-7.06), ` September`(-7.31) |
| 7 | ' ' | -0.51 | 0.75 | 1.17 | ` `(-0.51), ` June`(-1.26), ` September`(-3.26), ` December`(-3.76), ` the`(-4.51) |
| 8 | '20' | -0.02 | 4.81 | 0.13 | `20`(-0.02), `1`(-4.83), `8`(-5.33), `22`(-6.02), `29`(-7.52) |
| 9 | ' June' | -0.30 | 2.38 | 1.13 | ` June`(-0.30), ` January`(-2.68), ` April`(-2.68), ` November`(-3.55), ` July`(-3.80) |
| 12 | '0' | -0.66 | 0.75 | 1.31 | `0`(-0.66), `2`(-1.41), `1`(-2.16), `4`(-2.66), `3`(-3.28) |
| 13 | '.' | -0.04 | 3.37 | 0.21 | `.`(-0.04), ` and`(-3.42), `,`(-4.79), ` while`(-7.42), ` `(-8.42) |
| 23 | ' on' | -0.02 | 4.50 | 0.12 | ` on`(-0.02), ` `(-4.52), ` in`(-4.77), ` June`(-9.40), ` March`(-9.40) |
| 24 | ' ' | -0.13 | 3.13 | 0.62 | ` `(-0.13), ` September`(-3.25), ` June`(-3.88), ` April`(-4.38), ` July`(-4.63) |
| 25 | '24' | -3.03 | 0.69 | 3.43 | `25`(-1.72), `13`(-2.40), `23`(-2.53), `8`(-2.53), `5`(-2.90) |
| 26 | ' September' | -0.23 | 2.38 | 0.90 | ` September`(-0.23), ` July`(-2.60), ` April`(-3.10), ` December`(-3.98), ` November`(-4.23) |
| 31 | ' Thus' | -0.04 | 4.00 | 0.21 | ` Thus`(-0.04), ` So`(-4.04), ` Therefore`(-4.16), ` `(-6.29), ` Hence`(-6.79) |
| 42 | ' the' | -0.08 | 2.50 | 0.29 | ` the`(-0.08), ` first`(-2.58), ` before`(-6.33), ` earlier`(-6.33), ` earliest`(-10.83) |
| 56 | '.' | -2.03 | 1.87 | 0.48 | `.

`(-0.15), `.`(-2.03), `..`(-5.90), `.
`(-6.03), `..

`(-6.28) |
| 57 | '<|eot_id|>' | -1.47 | 0.06 | 4.50 | `<|eot_id|>`(-1.47), ` 

`(-1.53), ` `(-2.47), ` }`(-3.78), ` Question`(-3.97) |
