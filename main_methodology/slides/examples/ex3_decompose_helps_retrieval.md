# 範例③ — 拆解救回檢索:bridge entity 被揭露後才撈得到 gold

> 來源:`datasets/musique/raw/dev_first1000.jsonl`(段落池 + `is_supporting` gold)
> × `results/musique_1000_C_maxent0.3_3ds_terse.jsonl`(拆解 trace)。BM25 top-k=5。

## 案例 — 配偶的小孩(2-hop bridge)

```
Q   : Who is the child of Caroline LeRoy's spouse?
GOLD passages: idx {0, 10}        (0 = "Fletcher Webster",10 = Caroline LeRoy 配偶資訊)

[A 臂:用原問題檢索]  top-5 = {4, 5, 10, 15, 16}   → 漏掉 gold idx 0
   原因:原問題完全沒出現 "Daniel Webster",BM25 無從撈到「他兒子」那篇。

[B 臂:拆解後檢索]
  #1: Caroline LeRoy >> spouse            → 解出 bridge entity = Daniel Webster
  #2: Daniel Webster >> child             ← 這個 sub-q 才撈到 gold idx 0 "Fletcher Webster"

final answer = "Fletcher Webster"   ✓ EM = 1
```

## 機制(這就是「拆解查得更準」的核心直覺)

- 多跳題的**第二跳 gold 段落**,其關鍵實體(Daniel Webster)**不在原問題字面上** → 單一 query 的 BM25 結構性撈不到。
- **唯有先拆解、解出 bridge entity**,第二跳 sub-q 才帶著正確實體去檢索,命中 gold。
- 量化(M3,distractor pool,k=5,full-coverage = 整題所有 gold 都撈到):

  | Dataset | 原問題 A | 拆解 B(同預算) | Δ |
  |---|---|---|---|
  | MuSiQue | 27.0% | **64.5%** | **+37.5** |
  | 2Wiki | 51.6% | **89.6%** | +38.0 |
  | HotpotQA | 64.8% | **76.2%** | +11.4 |

## 為什麼重要(對應教授方向 2 + 206 gap)

- 直接回答教授質疑的「為什麼不 always-retrieve 就好」:**拆解讓 sub-query 更精準,撈到原問題撈不到的證據**。
- ⚠️ Caveat:此為 **distractor pool**;在 honest 21M 開放域上,KG-triple 式 sub-q(`X >> child`)是偏弱的 BM25 query,這正是 honest pool 多跳落後 DeepRAG 的待解槓桿(query 改寫)。
