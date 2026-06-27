# 範例④ — B3 角色混淆(最隱形的 confident hallucination)

> 來源:`deeprag_compare/check/confident_hallucination_cases.md` Q1(qid `5ade69e455...`)。
> B3 = 模型把問題裡的**約束實體(constraint)**當成**答案實體(answer)**回出去。

## 案例

```
Q   : What director worked with Vikram Bhatt on a film starring
      actors Rajneesh Duggal and Adah Sharma?
GOLD: Tinu Suresh Desai
PRED: Vikram Bhatt          ← 把問題裡的「約束條件」當答案回出去

SC(self-consistency)= 13.16(很高);20 次取樣 EM = 0/20
生成內容:"Vikram Bhatt directed the film Haunted 3D ... So the answer is Vikram Bhatt."
          (流暢、有推理外觀、每個 token 都自信)
```

## 為什麼它最難抓

- **SC 高、token-level 全自信** → 任何「信心/熵」訊號都把它看成「有把握」。
- 它不是知識錯(B1),也不是亂填(B2),而是**把題目解析錯**(誰是答案、誰是條件)。
- **唯一抓得到的**:`role-overlap × schema` 偵測器 —— 用 spaCy 解析每個實體的句法角色
  (CONSTRAINT / SUBJECT / DESCRIPTOR),若 pred 命中的是 **CONSTRAINT 實體** → 標記 B3。
  - v2 偵測器在 20Q pilot:sensitivity / specificity 皆 100%;修掉 v1 對 `5abdf12255` 的誤報。

## 為什麼重要(對應教授方向 1)

- 教授要的正是「**單一 SC 分數以外、能分辨真信心 vs 自信幻覺的指標**」。
- B3 證明:有一整類錯誤,**原則上**無法用信心/熵訊號偵測,需要 **QA-native 的句法-角色指標**。
- 連結 [[research_skip_decompose_signal]]:這也是 SEEK 的 entropy gate 在低熵尾端必然失效的根因之一。
