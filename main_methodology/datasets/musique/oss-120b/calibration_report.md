# OSS-120B Decomposition — Calibration Report

Source: `datasets/musique/oss-120b/selected_2048_with_oss120b_decomposition.jsonl`

## 1. Overall

- Rows: **2048**
- `decomposition_error` non-null: **0** (0.0%)
- Sub-question count matches gold: **1189 / 2048** (58.1%)
- `depends_on` graph invalid (forward / unparseable refs): **0**

### Delta histogram (generated − gold)

| Δ | Count | % |
|---|---:|---:|
| -2 | 10 | 0.5% |
| -1 | 273 | 13.3% |
| +0 | 1189 | 58.1% |
| +1 | 417 | 20.4% |
| +2 | 114 | 5.6% |
| +3 | 27 | 1.3% |
| +4 | 16 | 0.8% |
| +6 | 2 | 0.1% |

## 2. Per-hop breakdown

| Hop class | Expected sub-Qs | Total | Match | Match rate | Δ histogram |
|---|---:|---:|---:|---:|---|
| `2hop` | 2 | 512 | 416 | 81.2% | -1:4, +0:416, +1:80, +2:10, +6:2 |
| `3hop1` | 3 | 503 | 309 | 61.4% | -1:63, +0:309, +1:105, +2:19, +3:3, +4:4 |
| `3hop2` | 3 | 9 | 4 | 44.4% | +0:4, +1:4, +3:1 |
| `4hop1` | 4 | 573 | 281 | 49.0% | -2:5, -1:115, +0:281, +1:112, +2:44, +3:9, +4:7 |
| `4hop2` | 4 | 112 | 65 | 58.0% | -2:1, -1:15, +0:65, +1:24, +2:7 |
| `4hop3` | 4 | 339 | 114 | 33.6% | -2:4, -1:76, +0:114, +1:92, +2:34, +3:14, +4:5 |

## 3. Samples

### 3.1 Count-matched

**[2hop] match** — `2hop__160214_334142`

- Question: Who is the sibling of the performer of the singing voice for the U.S. release?
- Gold decomposition:
  - `Who performed the singing voice for the US release?` → `Donny Osmond`
  - `#1 >> sibling` → `Marie Osmond`
- Generated decomposition:
  - `#1` (deps=[]): Who performed the singing voice for the U.S. release?
  - `#2` (deps=[1]): Who is the sibling of the performer identified in #1?

**[3hop1] match** — `3hop1__20848_838364_49811`

- Question: Who were the first European to reach the original country of the film named the singer joined by Madonna on MTV Unplugged to sing and ended with ": Tongue Tied"?
- Gold decomposition:
  - `Which singer did she join on MTV Unplugged to sing?` → `Miley Cyrus`
  - `#1 : Tongue Tied >> country of origin` → `America`
  - `who were the first european to reach #2` → `Norse sailors (often called Vikings) from Iceland`
- Generated decomposition:
  - `#1` (deps=[]): Which singer performed together with Madonna on MTV Unplugged?
  - `#2` (deps=[1]): In which country was the film titled "<answer to #1>: Tongue Tied" originally produced?
  - `#3` (deps=[2]): Who were the first European explorers to reach the country identified in #2?

**[3hop2] match** — `3hop2__130513_661360_54023`

- Question: What year saw the opening of the first restaurant from the company that owns Hamburger University, in the country where the village of Bradden is located?
- Gold decomposition:
  - `What state is Bradden located?` → `England`
  - `Hamburger University >> owned by` → `McDonald's`
  - `when did the first #2 open in #1` → `1974`
- Generated decomposition:
  - `#1` (deps=[]): In which country is the village of Bradden located?
  - `#2` (deps=[]): Which company owns Hamburger University?
  - `#3` (deps=[1, 2]): In which year did the first restaurant of the company identified in #2 open in the country identified in #1?

**[4hop1] match** — `4hop1__16577_52584_71250_135051`

- Question: Who is the mother of the man who led the empire, which reduced to a colony the land where Marwan died, to its greatest extent?
- Gold decomposition:
  - `Where did Marwan die?` → `Egypt`
  - `in the 1st century bc #1 became a colony of` → `the Roman Empire`
  - `when did #2 reach its greatest extent` → `under Trajan`
  - `Who is #3 's mother?` → `Marcia`
- Generated decomposition:
  - `#1` (deps=[]): In which land did Marwan die?
  - `#2` (deps=[1]): Which empire, at its greatest extent, reduced the land where Marwan died to a colony?
  - `#3` (deps=[2]): Who was the man who led that empire at its greatest extent?
  - `#4` (deps=[3]): Who is the mother of that man?

**[4hop2] match** — `4hop2__53235_79417_57802_311656`

- Question: What is the place of death of the leader of the largest country in Europe by square miles after the collapse of America's main rival during the Cold War?
- Gold decomposition:
  - `what is the largest country in europe by square miles` → `Russia *`
  - `which nation was america's primary enemy during the cold war` → `the Soviet Union`
  - `who became leader of #1 after the collapse of #2` → `Boris Yeltsin`
  - `#3 >> place of death` → `Moscow`
- Generated decomposition:
  - `#1` (deps=[]): Who was America's main rival during the Cold War?
  - `#2` (deps=[]): Which country is the largest in Europe by square miles?
  - `#3` (deps=[1, 2]): Who was the leader of the country identified in #2 after the collapse of the rival identified in #1?
  - `#4` (deps=[3]): What is the place of death of the leader identified in #3?

**[4hop3] match** — `4hop3__3152_3356_32417_24628`

- Question: In what year did voters vote for a candidate that dominated South Carolina's state legislature in the state an American Idol contestant who sang Fuel is from?
- Gold decomposition:
  - `Who performed a Fuel song on this season?` → `Chris Daughtry`
  - `What state is American Idol contestant #1 from?` → `North Carolina`
  - `Which party dominated South Carolina's state legislature?` → `Democrat`
  - `What year did #2 voters once again vote for a #3 ?` → `2008`
- Generated decomposition:
  - `#1` (deps=[]): Which American Idol contestant performed the song "Fuel"?
  - `#2` (deps=[1]): What state is the American Idol contestant who performed "Fuel" originally from?
  - `#3` (deps=[]): Which political candidate is described as having dominated South Carolina's state legislature?
  - `#4` (deps=[2, 3]): In what year did voters in the state identified in #2 elect the candidate identified in #3?

### 3.2 Generated < Gold (OSS-120B under-decomposed)

**[2hop] under** — `2hop__51110_52808`

- Question: What area on a global map did the cavalier king charles spaniel originate?
- Gold decomposition:
  - `where did the cavalier king charles spaniel come from` → `the United Kingdom`
  - `where is #1 located on the world map` → `off the north - western coast of the European mainland`
- Generated decomposition:
  - `#1` (deps=[]): In which country or region did the cavalier king charles spaniel originate?

**[3hop1] under** — `3hop1__725408_101762_492752`

- Question: Where are the headquarters of the 48th Highlanders of the country where the Redpath Sugar founder lived?
- Gold decomposition:
  - `Redpath Sugar >> founded by` → `John Redpath`
  - `In which country did #1 live?` → `Canada`
  - `48th Highlanders of #2 >> headquarters location` → `Moss Park Armoury`
- Generated decomposition:
  - `#1` (deps=[]): In which country did the founder of Redpath Sugar live?
  - `#2` (deps=[1]): Where are the headquarters of the 48th Highlanders located in the country identified in #1?

**[4hop1] under** — `4hop1__322597_452185_53858_69706`

- Question: Korolyov was named after a city resident from a country that came what position in the World Cup?
- Gold decomposition:
  - `Korolyov >> named after` → `Sergei Korolev`
  - `#1 >> residence` → `Moscow`
  - `#2 is the capital city of which country` → `Russia`
  - `what place did #3 come in the world cup` → `8th`
- Generated decomposition:
  - `#1` (deps=[]): Who is the city resident after whom Korolyov was named?
  - `#2` (deps=[1]): From which country is the city resident after whom Korolyov was named?
  - `#3` (deps=[2]): What position did that country achieve in the World Cup?

**[4hop2] under** — `4hop2__136460_61674_1952_78767`

- Question: What country sent the most legal immigrants to the city where Gotham is filmed from the place where Descente aux enfers is located?
- Gold decomposition:
  - `Which place is Descente aux enfers in?` → `Caribbean`
  - `where is the tv show gotham filmed at` → `in New York City`
  - `What nation provided the most legal immigrants to #2 in the #1 ?` → `the Dominican Republic`
  - `what is the size of #3 in square miles` → `18,705`
- Generated decomposition:
  - `#1` (deps=[]): What city is the location where Gotham is filmed?
  - `#2` (deps=[]): In which city is Descente aux enfers located?
  - `#3` (deps=[1, 2]): Which country sent the most legal immigrants to the city identified in #1 from the place identified in #2?

**[4hop3] under** — `4hop3__31642_754527_855607_20507`

- Question: When was the country where Capellania is located colonized by the country where a terrorist bombing Gaddafi's Libya was supposedly involved in occurred?
- Gold decomposition:
  - `What terrorist bombing was Gaddafi's Libya supposedly involved in?` → `Lockerbie`
  - `All Saints Church, #1 >> located in the administrative territorial entity` → `Scotland`
  - `Capellanía >> country` → `Panama`
  - `When was #3 colonized by #2 ?` → `1698`
- Generated decomposition:
  - `#1` (deps=[]): In which country is Capellania located?
  - `#2` (deps=[]): In which country did the terrorist bombing that Gaddafi's Libya was supposedly involved in occur?
  - `#3` (deps=[1, 2]): When was the country identified in #1 colonized by the country identified in #2?

### 3.3 Generated > Gold (OSS-120B over-decomposed)

**[2hop] over** — `2hop__983_4661`

- Question: What does the acronym for the location of markets in which 6th gen iPods had the highest volume limited stand for?
- Gold decomposition:
  - `In what markets did 6th gen iPods limit the highest volume of the player?` → `EU`
  - `What does #1 stand for?` → `European Union`
- Generated decomposition:
  - `#1` (deps=[]): In which location of markets did 6th generation iPods have the highest limited volume?
  - `#2` (deps=[1]): What is the acronym used for that location of markets?
  - `#3` (deps=[2]): What does that acronym stand for?

**[3hop1] over** — `3hop1__157791_1887_76121`

- Question: How many roller coasters are at the six flags located in the state that, along with the state that the writer died, contains Ellis Island?
- Gold decomposition:
  - `In what state did the writer die?` → `New York`
  - `Ellis Island is considered in #1 state and which other?` → `New Jersey`
  - `how many roller coasters at six flags #2` → `14`
- Generated decomposition:
  - `#1` (deps=[]): Who is the writer referred to in the original question?
  - `#2` (deps=[1]): In which state did that writer die?
  - `#3` (deps=[]): Which states contain Ellis Island?
  - `#4` (deps=[2, 3]): Which state, other than the writer's death state, is among the states that contain Ellis Island?
  - `#5` (deps=[4]): What is the name of the Six Flags amusement park located in that state?
  - `#6` (deps=[5]): How many roller coasters does that Six Flags amusement park have?

**[3hop2] over** — `3hop2__763533_863695_74533`

- Question: Who was president of the nation where Eckelson Township is located when the city containing Aioi Bridge was bombed?
- Gold decomposition:
  - `Eckelson Township >> country` → `US`
  - `Aioi Bridge >> located in the administrative territorial entity` → `Hiroshima`
  - `who was the #1 president when #2 was bombed` → `President Truman`
- Generated decomposition:
  - `#1` (deps=[]): In which nation is Eckelson Township located?
  - `#2` (deps=[]): Which city contains Aioi Bridge?
  - `#3` (deps=[2]): When was the city containing Aioi Bridge bombed?
  - `#4` (deps=[1, 3]): Who was the president of the nation where Eckelson Township is located on the date when the city containing Aioi Bridge was bombed?

**[4hop1] over** — `4hop1__698_669_158722_74346`

- Question: What's the marginal tax rate where a stringed argent is what a lion is to the country the person who paid for Chopin's funeral invited him to?
- Gold decomposition:
  - `Who paid for Chopin's funeral?` → `Jane Stirling`
  - `Where did #1 invite Chopin?` → `Scotland`
  - `A lion is to #2 as a stringed agent is to?` → `Ireland`
  - `what is the marginal rate of tax in #3` → `52%`
- Generated decomposition:
  - `#1` (deps=[]): Who paid for Frédéric Chopin's funeral?
  - `#2` (deps=[1]): Which country did the person who paid for Chopin's funeral invite him to?
  - `#3` (deps=[2]): What national animal or emblem is associated with the country identified in #2?
  - `#4` (deps=[]): Which country has a silver (argent) stringed instrument, such as a harp, as a national symbol?
  - `#5` (deps=[4]): What is the top marginal personal income tax rate of the country identified in #4?

**[4hop2] over** — `4hop2__53235_90366_57802_311656`

- Question: Where is the place of death of the man who became leader of the largest country in Europe in square miles after the collapse of the nation Germany agreed to sign a non-aggression pact with in 1939?
- Gold decomposition:
  - `what is the largest country in europe by square miles` → `Russia *`
  - `in 1939 germany agreed to sign a non-aggression pact with which country` → `the Soviet Union`
  - `who became leader of #1 after the collapse of #2` → `Boris Yeltsin`
  - `#3 >> place of death` → `Moscow`
- Generated decomposition:
  - `#1` (deps=[]): Which nation did Germany agree to sign a non-aggression pact with in 1939?
  - `#2` (deps=[1]): In what year did the nation identified in #1 collapse or dissolve?
  - `#3` (deps=[1, 2]): Which country was the largest in Europe by square miles after the collapse of the nation identified in #1?
  - `#4` (deps=[1, 3]): Who became the leader of the country identified in #3 after its formation following the collapse of the nation identified in #1?
  - `#5` (deps=[4]): What was the place of death of the person identified in #4?

**[4hop3] over** — `4hop3__38130_8966_51964_32334`

- Question: Are these other languages learned in the country of Eisenhower's VP as popular as the language of the people who own the oldest navy in the world?
- Gold decomposition:
  - `Who served as Eisenhower's vice president?` → `Nixon`
  - `#1 was a president of what country?` → `U.S.`
  - `who has the oldest navy in the world` → `The Spanish`
  - `Are these other languages learned in the #2 as popular as #3 ?` → `totals remain relatively small in relation to the total U.S population.`
- Generated decomposition:
  - `#1` (deps=[]): What is the country of Eisenhower's VP?
  - `#2` (deps=[1]): What are these other languages learned in the country of Eisenhower's VP?
  - `#3` (deps=[2]): What proportion of the population of the country of Eisenhower's VP has learned any of the languages identified in #2?
  - `#4` (deps=[]): Which country has the oldest navy in the world?
  - `#5` (deps=[4]): What language do the people who own the oldest navy in the world speak?
  - `#6` (deps=[5]): What proportion of the people who own the oldest navy in the world speak the language identified in #5 as a first language?

## 4. Interpretation notes

- **Hop number ≠ sub-Q count required.** A 4-hop MuSiQue question may decompose into 4 atomic sub-Qs, but OSS-120B sometimes splits a single hop into multiple steps (over) or fuses two hops (under). Count agreement alone is necessary but not sufficient — the thesis Stage 3.1 step (2) self-verify filter (oracle paragraph + check final answer) is what actually gates SFT inclusion.
- **Hop classes with low match rate** (typically 3-hop / 4-hop) are the ones to watch when deciding whether to retrain the decomposer or to widen self-verify tolerance.
- `depends_on` integrity should be 100% if the JSON parsing was sound; any non-zero count here flags rows that would need manual repair before SFT.