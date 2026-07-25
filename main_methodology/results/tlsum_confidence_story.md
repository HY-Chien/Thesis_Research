# tl_sum as the confidence signal: mechanism + residual failures

Direct path battery, n=1000x4, alias-aware EM. uncertainty = tl_sum unless noted.

## Q1a. Entropy chain-rule decomposition — the discriminative part is the escape term

H_t = H_B(m_t) + m_t*H_top5 + (1-m_t)*H_tail. AUROC(wrong), macro:

| component | meaning | macro AUROC |
|---|---|---:|
| max_ent | full entropy (all three terms mixed) | 0.7263 |
| max_hbin | escape in/out top-5 only | 0.7420 |
| max_htop5 | ambiguity WITHIN the top-5 candidates | 0.7115 |
| max_htail | composition of the tail | 0.5015 |
| min_top5m | escape mass, weakest token | 0.7419 |
| tl_sum | escape surprisal, summed over the answer | 0.7493 |

## Q1b. Where the sum beats the weakest link — multi-token leakage accumulation

AUROC by answer length (pooled 4 ds):

| len (tokens) | n | tl_sum | min_top5m | max_ent |
|---|---:|---:|---:|---:|
| 1 | 419 | 0.747 | 0.747 | 0.702 |
| 2-3 | 1698 | 0.794 | 0.790 | 0.774 |
| 4-6 | 1236 | 0.790 | 0.786 | 0.756 |
| 7+ | 647 | 0.805 | 0.788 | 0.787 |

share of tl_sum carried by the single worst token (answers >=2 tokens): wrong 0.88 vs correct 0.92 — errors leak across several tokens, which max-aggregation cannot see.

Wrong answers CAUGHT by tl_sum but missed by max_ent (per-ds percentile ranks):

- [popqa] q='In what city was Leobardo Soto born?' pred='Celaya' gold='Puebla' | tl_pct=0.76 ent_pct=0.44 | TL/token: C:0.02 elay:0.02 a:0.00

Correct answers max_ent would FALSE-ALARM but tl_sum clears (within-top-5 ambiguity, e.g. format/alias variants):

- [popqa] q="What is Duilio Forte's occupation?" pred='architect' | tl_pct=0.45 ent_pct=0.77

## Q1c. Selective answering with tl_sum (pooled + per ds)

keep the most-confident X% (lowest tl_sum), accuracy inside the kept bucket:

| coverage | pooled acc | 2wiki | hotpotqa | musique | popqa |
|---:|---:|---:|---:|---:|---:|
| 25% | 52.8 | 34.0 | 57.6 | 22.4 | 79.2 |
| 50% | 38.9 | 26.8 | 45.4 | 15.2 | 57.0 |
| 75% | 30.2 | 22.1 | 38.0 | 12.5 | 46.3 |
| 100% | 24.1 | 18.0 | 30.3 | 10.2 | 37.8 |

## Q2. What is left: confident-wrong and unconfident-right

Confident-wrong = wrong with tl_sum <= median(correct):

| category | 2wiki | hotpotqa | musique | popqa | total |
|---|---:|---:|---:|---:|---:|
| clean wrong entity (parametric hallucination) | 116 | 55 | 89 | 21 | 281 |
| boundary/alias near-miss | 43 | 52 | 31 | 22 | 148 |
| partial overlap (mixed entity) | 22 | 9 | 9 | 0 | 40 |
| wrong date/number | 1 | 5 | 16 | 0 | 22 |
| **total CW (share of wrong)** | 182 (22%) | 121 (17%) | 145 (16%) | 43 (7%) | 491 |

Confident-wrong examples (lowest tl_sum first):

- [hotpotqa] (boundary/alias near-miss) tl=0.0000 q='What retailer in ABQ Uptown is headquarted in Poole, Dorset, United Ki' pred='Lush' gold='Lush Ltd.' f1=0.67
- [hotpotqa] (boundary/alias near-miss) tl=0.0000 q='Which novelist wrote under the pen name Walter Ericson: Laura Esquivel' pred='Howard Fast' gold='Howard Melvin Fast' f1=0.80
- [hotpotqa] (boundary/alias near-miss) tl=0.0000 q='Which band formed in England, Of Montreal or Vib Gyor?' pred='Vib Gyor' gold='Vib Gyor was a band that was formed in L' f1=0.33
- [2wiki] (clean wrong entity (parametric hallucination)) tl=0.0000 q='Which film was released more recently, Bidasari or Tangled Trails?' pred='Tangled' gold='Bidasari' f1=0.00
- [popqa] (boundary/alias near-miss) tl=0.0000 q='In what city was Diotima of Mantinea born?' pred='Mantinea' gold='Mantineia' f1=0.00
- [2wiki] (boundary/alias near-miss) tl=0.0000 q="Who is Blanche Of Portugal (1259–1321)'s paternal grandmother?" pred='Beatrice of Castile' gold='Urraca of Castile' f1=0.67
- [popqa] (boundary/alias near-miss) tl=0.0000 q='In what city was Margaret, Countess of Pembroke born?' pred='Windsor' gold='Windsor Castle' f1=0.67
- [musique] (clean wrong entity (parametric hallucination)) tl=0.0001 q='What record label is the performer who released All Your Faded Things ' pred='Cold Crush Records' gold='Kill Rock Stars' f1=0.00
- [2wiki] (clean wrong entity (parametric hallucination)) tl=0.0001 q="Who is Grimoald The Younger's paternal grandfather?" pred='Pepin of Herstal' gold='Ansegisel' f1=0.00
- [popqa] (clean wrong entity (parametric hallucination)) tl=0.0001 q='In what city was Louis Lloyd Winter born?' pred='Toronto' gold='Canada' f1=0.00
- [musique] (boundary/alias near-miss) tl=0.0001 q='Into what river does the river that Thuli River turns into flow?' pred='Shashe River' gold='Limpopo River' f1=0.50
- [musique] (clean wrong entity (parametric hallucination)) tl=0.0001 q='Who was the first African American student at the university Robert Kh' pred='Gene Gray' gold='James Howard Meredith' f1=0.00

Unconfident-right = correct with tl_sum >= median(wrong): n=164 (17% of all correct). mean answer length 2.5 tokens vs 3.2 for all correct; >=5-token answers 9% vs 19%.

Unconfident-right examples:

- [musique] tl=1.060 len=2 q="Who is the brother of the artist who made Nothin' My Love Can't Fix's " pred='Matthew Lawrence'
- [popqa] tl=0.608 len=1 q="What is Pathit Pisitkul's occupation?" pred='Actor'
- [hotpotqa] tl=0.604 len=3 q='The city that contains the Yunnan Provincial Museum is also known by w' pred='the Spring City'
- [musique] tl=0.538 len=5 q='What is the position of the person impeached in England for acts commi' pred='Governor-General of India'
- [popqa] tl=0.534 len=3 q='In what city was Ian Rutherford born?' pred='Dunedin'
- [2wiki] tl=0.459 len=2 q="What is the place of birth of the director of film Peter'S Friends?" pred='Belfast'
- [2wiki] tl=0.422 len=1 q='Where was the performer of song Remember (John Lennon Song) born?' pred='Liverpool'
- [hotpotqa] tl=0.391 len=5 q='What is the name of the number-one single featuring the winner of Cele' pred='Every Rose Has Its Thorn'
