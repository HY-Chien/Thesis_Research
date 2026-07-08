# 範例②(完整版)— dirOnly 的五大失敗類別:直答對,一拆就錯

> 來源:逐資料集 join `{ds}_1000_alwaysskip.jsonl`(直答 EM=1)× `{ds}_1000_C_maxent0.3_3ds_terse.jsonl`(拆解 EM=0),同一顆 12K adapter。
> 全量 384 筆(2Wiki 83 + HotpotQA 197 + MuSiQue 65 + PopQA 39)。
> 類別數字為啟發式規則全量分類(方向可信,邊界案例可能歸錯);每個附例皆逐筆驗證過。
> 單例版見 [`ex2_dironly_decompose_breaks.md`](ex2_dironly_decompose_breaks.md)。

## dirOnly 題數(直答對 → 拆解錯)

| | 2Wiki | HotpotQA | MuSiQue | PopQA |
|---|---|---|---|---|
| dirOnly | 83 | **197** | 65 | 39 |

> ⚠️ 投影片舊表欄位錯位(把 197 標成 PopQA、65 標成 HotpotQA、39 標成 MuSiQue)。
> 正解:197=HotpotQA、65=MuSiQue、39=PopQA。來源 `results/decompose_vs_direct_2x2.md`。

---

## 類別 1 ⭐ 比較題壞掉(2Wiki 主力,~58/83)

拆解把「哪個先 / 哪個多 / 誰較年輕」拆成各自查屬性,但**合成步沒把比較做完**。兩種壞法:

**(1a) 回了被比較的「值」而非實體**(= ex2 的型)
```
Q: Which film came out first, Blind Shaft or The Mask of Fu Manchu?
   gold=The Mask of Fu Manchu  直答✓  拆解='1932' ✗
   steps: [Blind Shaft >> date -> 2003] [Mask of Fu Manchu >> date -> 1932]
   → 算對了 1932 最早,卻把「1932」當答案吐出來,沒換回片名

Q: Which song was released more recently, Go Ahead And Break My Heart or Knee Deep In My Heart?
   gold=Go Ahead And Break My Heart  直答✓  拆解='2016' ✗
   steps: [Go Ahead... -> May 9, 2016] [Knee Deep... -> 29 November 2013]
```

**(1b) 比錯邊(挑錯實體)**
```
Q: who is younger Keith Bostic or Jerry Glanville?
   gold=Keith Bostic  直答✓  拆解='Jerry Glanville' ✗
   steps: [Glanville -> 1941] [Bostic -> 1961]  → 1961 較晚=較年輕,卻選了 1941 那個

Q: Which band, Letters to Cleo or Screaming Trees, had more members?
   gold=Letters to Cleo  直答✓  拆解='Screaming Trees' ✗
```

---

## 類別 2 ⭐ 答案粒度/語意標籤不符(PopQA 幾乎全部 + 2Wiki 日期)

拆解從檢索文字抓回一個**「對但對不準」**的字串,EM 掛掉:

```
PopQA(全是單跳 occupation 題,39 筆幾乎都屬此類):
  occupation? gold='journalist'  拆解='broadcast journalist' ✗  (太細)
  occupation? gold='composer'    拆解='British classical composer' ✗  (加修飾詞)
  occupation? gold='guitarist'   拆解='musician'            ✗  (太粗/換詞)
  occupation? gold='chef'        拆解='cricketer'           ✗  (檢索抓到別人)

2Wiki 日期題(合成步把日期截成年份):
  ...born?  gold='30 June 1963'      拆解='1963' ✗
  ...death? gold='25 July 1182'      拆解='1173' ✗

HotpotQA 地點(只回上層、漏掉細節):
  gold='Canary Islands, Spain'   拆解='Spain'         ✗
  gold='Indianapolis Motor Speedway'  拆解='Indianapolis' ✗
```

---

## 類別 3 Y/N 問句極性翻錯(HotpotQA ~6)

證據都撈到了,但**最後的「是/否」判斷做反**:

```
Q: Are both Dafeng District and Dazhou located in the same province?
   gold=no  直答=No✓  拆解=Yes ✗
   steps: [Dafeng -> Yancheng] [Dazhou -> Sichuan]  → 明明不同省,卻合成成 Yes

Q: Are both Cypress and Ajuga genera?
   gold=no  直答=No✓  拆解=Yes ✗
   steps: [Cypress >> genus -> Cupressus] [Ajuga >> genus -> Ajuga]
```

---

## 類別 4 ⭐ 過度拆解 → 子步答錯往下傳遞(HotpotQA/MuSiQue 主力)

本來直答就對的題,硬拆之後**某一子步抓錯實體/解錯橋接**,錯誤一路傳到最後:

```
MuSiQue:
  Q: Where was Mary's betrothed born?  gold=Nazareth  直答✓  拆解='Brooklyn' ✗
     steps: [Mary >> betrothed -> 28]  ← 橋接實體解成亂碼 "28",後面全錯
  Q: Who is the father of the Labo M performer?  gold=Louis Chedid  拆解='Andrée Chedid' ✗
     steps: [Labo M -> -M-] [-M- >> father -> Andrée Chedid]  ← 那是「母親」不是父親

HotpotQA:
  Q: What screenwriter ... "Evolution" co-wrote a film starring Nicolas Cage and Téa Leoni?
     gold=David Weissman  直答✓  拆解='David S. Goyer' ✗
     steps: [Evolution >> screenwriter -> David S. Goyer]  ← 抓錯了那部叫 Evolution 的片
  Q: Brown State Fishing Lake is in a country with a population of how many inhabitants?
     gold='9,984'  直答✓  拆解='331,449,281' ✗  ← 把「美國總人口」當成答案(問的是該湖所在地人口)
```

---

## 類別 5 合成輸出格式壞掉 / 沒收斂(HotpotQA 少數)

拆解的中間產物**直接外洩當成答案**,或卡在模板:

```
Q: D1NZ is a series based on what oversteering technique?  gold=Drifting
   拆解='D1NZ >> based on' ✗   ← 連模板都沒填就吐出來

Q: In what city did the "Prince of tenors" star in a film based on a Puccini opera?  gold=Rome
   拆解='Corelli >> present in work -> Tosca..., La fanciulla..., Gianni Schicchi' ✗
   ← 把 sub-answer 清單整包吐出,沒做最後合成
```

---

## 每個資料集被哪一類主導

| 資料集 | dirOnly | 主導類別 |
|---|---|---|
| **2Wiki** | 83 | 類別1 比較題壞掉(~70%)—— 多為「回年份不回片名」 |
| **HotpotQA** | 197 | 類別4 過度拆解傳遞錯誤(~60%)+ 類別1 比較題 |
| **MuSiQue** | 65 | 類別4 橋接實體解錯傳遞(~66%) |
| **PopQA** | 39 | 類別2 標籤粒度/語意不符(幾乎 100%,全是單跳 occupation) |

## 結論(對應方向 2:拆解不是免費的)

這五類對應兩種「不該拆」:

1. **單跳題根本不用拆(類別 2 / PopQA)** —— 一拆反而引入檢索雜訊或換到對不準的標籤。
   這類最該被 router 用「低熵=有信心就直答」攔下。
2. **拆解的執行本身有風險(類別 1/3/4/5)** —— 合成沒收尾、比錯邊、橋接解錯、格式外洩。
   越是模型本來就答得出的簡單題,硬拆的期望損失越大。

→ **「該不該拆」不只是難度問題,更是風險問題**:router 要在「拆解的潛在收益」與「這五類執行失敗的風險」之間取捨。
