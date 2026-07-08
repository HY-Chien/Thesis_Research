# 範例① — Confident Hallucination(低熵卻錯)

> 來源:`results/2wiki_1000_alwaysskip.jsonl`(強制單步直答,記錄 `skip_probe_ent` = 原問題直答的 max token entropy)。
> 重點:這些題模型**幾乎零猶豫(熵 ≈ 0)卻答錯**——低熵不代表「簡單」,而是「自信地錯」。

## 案例 A — 祖父是誰(熵 0.002)

```
Q   : Who is the paternal grandfather of Alexandru Coconul?
GOLD: Mihnea Turcitul
PRED: Radu Mihnea          ← 答錯(答成另一位同族王公)
skip_probe_ent = 0.002     ← 模型對這個錯答幾乎 100% 確定
EM  = 0
```

## 案例 B — 祖母是誰(熵 0.006)

```
Q   : Who is Blanche Of Portugal (1259–1321)'s paternal grandmother?
GOLD: Urraca of Castile
PRED: Beatrice of Castile   ← 答錯(同樣 "of Castile",貌似合理的幻覺)
skip_probe_ent = 0.006
EM  = 0
```

## 為什麼重要(對應教授方向 1)

- 教授原話:**「高 SC / 高信心不保證答對,要把 confident hallucination 抓出來。」**
- 這正是 entropy gate 的失效模式:在真多跳的**低熵尾端**,門檻會誤判「有信心 → 可以 skip 拆解 / 不檢索」,結果是自信地給出錯誤實體。
- 接 [[research_b1b2b3_framing]]:這類「自信錯誤」可再細分 B1/B2/B3。
