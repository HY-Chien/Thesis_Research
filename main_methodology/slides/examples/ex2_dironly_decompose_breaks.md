# 範例② — dirOnly:直答對,一拆就錯

> 來源:join `2wiki_1000_alwaysskip.jsonl`(直答 EM=1)× `2wiki_1000_C_maxent0.3_3ds_terse.jsonl`(拆解 EM=0)。
> 2Wiki 原本 **83 題** 落在這格(直答對、拆解反而錯)。這是「拆解不是免費」的證據。
>
> **2026-06-11 後記:本例這一型已被修掉。** 合成 value→entity guard
> (`seek/pipeline.py::_rescue_value_answer`)把合成步回的「值」(1932)確定性映射回
> 持有該值的 step 實體(The Mask Of Fu Manchu)。2Wiki dirOnly 83 → **39**
> (修掉的 44 題全是此型);剩下的 39 題才是真正的拆解 trade-off。

## 案例 — 比較兩部片誰先上映

```
Q   : Which film came out first, Blind Shaft or The Mask Of Fu Manchu?
GOLD: The Mask Of Fu Manchu

[直答 / 強制單步]
PRED = "The Mask of Fu Manchu"     ✓ 正確(模型本來就知道)

[拆解 pipeline]
decomposition:
  #1: Blind Shaft >> publication date
  #2: The Mask of Fu Manchu >> publication date
final answer = "1932"               ✗ 錯誤
```

## 壞因(失敗模式)

- 拆解 + 各 sub-answer 其實都對(撈到兩個年份),但**合成步(synthesize)壞掉**:
  比較題要回「對應的**片名**」,模型卻回了被比較的**值(年份 1932)**。
- 典型 `bridge_comparison` 失敗:多此一舉地把一個模型「直接就會」的題拆開,反而在合成階段把答案類型搞錯。

## 為什麼重要(對應教授方向 2)

- 證明「**該不該拆解**」是真實 trade-off,不是越拆越好 → 替動態決策(router)提供動機。
- 也說明 M2 router 在某些 dataset(如 HotpotQA)該偏向「不拆」是有道理的。
