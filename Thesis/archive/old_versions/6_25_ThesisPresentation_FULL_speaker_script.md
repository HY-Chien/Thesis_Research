# 6/25 Thesis Presentation - Full Speaker Script
Merged in final PDF order: Introduction / Motivation -> Methodology -> Experiments -> Analysis -> Conclusions -> Appendix notes.

---

# Part 1 - Introduction / Motivation

# 6/25 Thesis Presentation Speaker Script - Intro and Conclusion

## Slide 11. Motivation

**Script:**

這張是整個研究的 motivation。

在 RAG 裡，multi-step retrieval 通常會把一個複雜問題拆成幾個比較小的 sub-questions。這樣做的好處是，每一步的 query 會更清楚，retriever 比較容易找到相關 evidence。

但是 decomposition 不是永遠可靠。當問題被拆短之後，原問題裡一些有用的線索可能會被拿掉。例如原本問題裡同時提到的兩個對象、比較條件，或幫助定位答案的文字，在 sub-question 裡可能不見。這時候 retrieval 反而可能找錯資料，後面的回答也會一路被影響。

所以這篇研究把 adaptive RAG 重新看成一個 decomposition-routing problem。系統不只要決定什麼時候 retrieve，也要決定什麼時候該相信 decomposition，什麼時候應該保留原始問題的路徑。

我們的目標就是建立一個輕量的 sub-question-level adaptive RAG framework。它可以控制每個 sub-question 要不要 retrieval，同時保留 original-question path，避免每一題都被強制拆解。

---

## Slide 12. SEEK

**Script:**

基於前面的 motivation，我們提出 SEEK，也就是 Self-certainty Evaluated Entropy-gated Knowledge-retrieval。

SEEK 是一個 lightweight adaptive retrieval framework。它的核心不是單純做 question decomposition，而是讓 decomposition 變成一條可以被控制、也可以被比較的路徑。

整個方法有四個重點。

第一是 dual-path answering。SEEK 同時保留 direct answer 和 decomposed answer。Direct path 用原始問題檢索並回答，decomposition path 則把問題拆成 sub-questions 後逐步回答。

第二是 entropy-gated sub-question retrieval。對每個 sub-question，模型會先嘗試回答一次；如果答案的 max entropy 高，代表模型不確定，才觸發 retrieval。

第三是 original-question context preservation。因為 sub-question 可能丟掉原問題的線索，所以我們會把 original-question evidence 加回 decomposed retrieval 的 context。

第四是 training-free verification。最後 SEEK 不再訓練一個新的 router，而是用 inference 過程中已經產生的 confidence 和 consistency signals，選出比較可靠的 final answer。

所以 SEEK 的定位是：保留 decomposition 對 multi-hop retrieval 的好處，但不要讓系統被 always-decompose 綁住。

---
---

# Part 1.5 - Related Work Summary

## Slide 24. Summary of Related Works

**Script:**

這張表刻意只留下三個最容易理解的維度：decision unit、retrieval signal，以及 decision transparency。

Adaptive-RAG 的 decision unit 是整個 question。它用 classifier 判斷一題要不要 retrieval、要 single-step 還是 multi-step。這個方法很直覺，但對 multi-hop reasoning 來說太粗，因為它不能在中間步驟調整 retrieval decision。

FLARE 則是 token 或 sentence level。它用 token probability 來判斷什麼時候 retrieve，比 question-level 更細，但對 multi-hop QA 來說又太細、也比較 noisy。它偵測的是局部 token uncertainty，不一定對應到一個完整 sub-question 是否需要 evidence。

DeepRAG 的 granularity 最接近我們想要的方向，因為它也在 sub-question level 做 retrieval decision。不過它的 signal 是 learned policy，所以 decision 藏在 policy 裡，比較難直接說明某一步為什麼 retrieve。

SEEK 同樣選擇 sub-question 作為 decision unit，但使用的是可以直接觀察的 tail-leakage signal，最後的 verifier 也是 explicit rules。也就是說，我們不只知道系統選了哪一條路，也可以指出是哪些 confidence 與 consistency 條件觸發這個選擇。

所以這張表的 takeaway 很簡單：SEEK 是這裡唯一同時具有 sub-question-level decision，以及 explicit、interpretable confidence signal 的方法。這就是下一段 Methodology 要介紹的切入點。

---

# Part 2 - Methodology

# 6/25 Thesis Presentation Speaker Script - Methodology

## Methodology Narrative

adaptive retrieval 是必要的，但 retrieval decision 不能只在整個 question level 做，也不適合只靠 token-level trigger。對 multi-hop QA 來說，比較自然的粒度是 sub-question level。

SEEK 的方法就是把這個問題重新整理成 decomposition routing problem。
也就是說，我們不把 question decomposition 當成固定流程，而是問：什麼時候應該相信拆解後的路徑？
什麼時候應該保留原始問題的完整脈絡？

因此 SEEK 有三個核心設計：

1. 用 SFT 訓練一個輕量 decomposer，產生可執行的 sub-question plan。
2. 用 max entropy 判斷每個 sub-question 是否需要 retrieval。
3. 同時保留 original-question path 和 decomposition path，最後用 training-free verifier 選擇較可靠的答案。

保留 decomposition 對 multi-hop retrieval 的好處，同時避免 always-decompose 帶來的 context loss、retrieval drift 和 error propagation。
---

## Slide 25. Methodology
**Script:**

接下來進入 Methodology。

前面 related work 的結論是，multi-step retrieval 確實可以幫助複雜問題，但現有方法還有兩個限制。
第一，如果只在 question level 做 routing，decision 太粗，沒有辦法控制每一個 reasoning step。
第二，像 DeepRAG 這類在 sub-question level 學 policy 的方法，效果雖然好，但要額外訓練，整個流程比較重。

所以 SEEK 想回答的是：能不能保留 sub-question-level decision 這個正確粒度，但用更輕量、可解釋的方式來做？

我們的做法是把 question decomposition 視為 routing problem：系統不只要會拆問題，
也要知道哪些 step 需要 retrieval，以及最後要相信原始問題路徑還是拆解路徑。
---

## Slide 26. SEEK Framework Overview
**Script:**

這張是 SEEK 的整體架構，簡單講就是一個 decomposer 加上兩條路徑。

decomposer 把 original question 拆成 structured sub-questions，這也是我們唯一需要訓練的部分。

接下來系統會跑兩條路：direct path 保留原始問題去檢索、生成一個答案，好處是保留原問題裡的線索；
decomposition path 則逐步回答 sub-question，並用 max entropy 決定每一步要不要 retrieval。

最後 Verifier 根據 confidence 和 consistency 從兩條路裡選出 final answer。

核心精神就是：讓 decomposition 變成可控制的路徑，而不是每一題都強制拆解。

---
## Slide 27. Method - Data Construction
**Script:**

這張投影片說明 decomposer 的訓練資料怎麼建。

我們希望把一個複雜問題，拆成幾個比較簡單的子問題，讓系統可以一步一步去回答。
每個子問題只負責問一件小事；當後面的子問題需要用到前面得到的答案時，我們就用 `#N` 來標記、把它們串接起來。

以這個例子來看，原始問題是：

```text
When was the institute that owned The Collegian founded?
擁有《The Collegian》的那所機構是什麼時候創立的？
```

原始 decomposition records 裡有兩個 step。第一步先找出 `The Collegian` 是由誰 owned，答案是 `Houston Baptist University`。
第二步再問這個 institute 是什麼時候 founded，答案是 `1960`。

資料轉換時，我們把原本不連續的 step id 重新編成連續的 `#1`, `#2`，並讓 placeholder 自動對齊前面 step 的答案。

最後 SFT target 會變成：

```text
#1: The Collegian >> owned by
#2: When was #1 founded?
```
這樣做可以讓模型學到穩定可執行的 decomposition format。

---
## Slide 28. Method - Training Stage
**Script:**

訓練階段很單純：我們只訓練 decomposer。

它的輸入就是一個原始問題，輸出就是剛剛看到的那種子問題清單。我們用大約 12K 筆的拆解資料來訓練，裡面有需要多步推理的題目，也有比較簡單、一步就能回答的題目。我們希望模型學會兩件事：一個是怎麼把複雜問題拆成可以執行的子問題，另一個是怎麼用 `#N` 把前後相依的步驟串起來。

這裡要強調的是，decomposer 並不負責回答問題，它只負責產生一份系統可以照著跑的計畫。真正去回答、決定要不要檢索、以及最後選哪個答案，都是在推論階段才發生。

換句話說，SEEK 不需要再額外訓練一個檢索策略，也不像 DeepRAG 那樣要用 binary tree search 去產生檢索決策的標註。我們的 entropy gate 和 verifier 都是推論時才用、完全不需要訓練的元件。

---

## Slide 29. Method - Inference Stage

**Script:**

Inference 階段是 SEEK 最重要的設計。

每一題進來後，系統會產生兩個 candidate answers。

第一個是 direct path。這條路保留原始問題，不做 decomposition，直接用 original question 做 retrieval，然後根據 retrieved evidence 產生答案。

第二個是 decomposition path。這條路會先把問題拆成 sub-questions，再一步一步回答。
每一個 sub-question 都會先讓模型嘗試回答一次，並計算 answer的 max entropy。
如果這一步不確定，才針對這個 sub-question 做 retrieval；如果這一步有把握，就不額外檢索。

最後，verifier 會從兩個 candidate answers 中選出 final answer。


所以這張投影片可以把 SEEK 理解成三個元件：
Direct Path 保留原問題脈絡，
Decomposition Path 提供 step-by-step retrieval control，
Verifier 則負責最後的 routing decision。

---

## Slide 30. Method - Entropy-Gated Retrieval
**Script:**

這張投影片說明為什麼我們用 entropy 來做 retrieval gate。

如果只看 generated token 的 probability，我們只能知道模型最後選出來的那個 token 有多高機率。
但是這個數字沒有告訴我們整個 next-token distribution 長什麼樣子。

Entropy 則不同。Entropy 衡量的是整個 distribution 的不確定性：
```text
H = - sum_i p_i log p_i
```

也就是說，它會同時考慮所有候選 token 的 probability。
如果模型把機率分散在很多可能答案上，entropy 就會高，表示模型其實不確定。如果模型雖然有其他候選，但機率主要集中在少數選項上，entropy 就會比較低。

在 SEEK 裡，我們不是只看一個 token，而是對 answer裡每個 generated token 都計算 entropy，然後取最大值，也就是 max entropy。

(這個設計的直覺是：答案通常是一個 entity、年份或短語，只要其中一個關鍵 token 很不確定，這個答案就可能不可靠。)

因此 max entropy 可以作為一個便宜的 confidence signal：entropy 高，代表這一步可能需要 retrieval；
entropy 低，代表模型對這個 sub-answer 比較有把握。

---

## Slide 31. Method - Inference Stage: Direct Path
**Script:**

先看 direct path。

Direct path 的重點是 preserve the original question without decomposition。
也就是說，系統不把問題拆成 sub-questions，也不做 query rewrite，而是直接使用 original question 去 retrieve document。

接著，模型會根據這些 retrieved document 生成一個 direct candidate answer，並記錄這個答案的 max entropy，讓後面的 verifier 可以知道 direct path 的 confidence。

這條路很重要，因為 decomposition 並不是永遠有幫助。
有些問題如果硬拆成短短的 sub-question，反而會丟掉原問題裡的線索，結果就檢索到錯的資料。

所以 direct path 在 SEEK 裡不是 baseline 而已，它是一條保留原始問題脈絡的 candidate path。
後面的 verifier 可以在 decomposition 不穩定時回到這條路。

---

## Slide 32. Method - Inference Stage: Decomposition Path
**Script:**

第二條是 decomposition path。

這條路會先把原始問題拆成 sub-questions，然後照順序逐步處理。
這裡的流程不是先回答完所有 sub-questions 之後，再決定要不要全部檢索；而是每一個 sub-question 都單獨做一次 confidence check。

對每個 sub-question，系統會先不檢索，讓模型產生一個 暫時的 answer，並計算這個 answer 的 max entropy。
如果 max entropy 高於 threshold，代表模型對這一步不確定，系統才針對這個 sub-question 做 retrieval，然後用 retrieved context 重新回答。
如果 max entropy 低，系統就保留原本的答案，不額外檢索。

每一步只在需要時檢索，最後再把所有 finalized sub-answers 合成一個 decomposed candidate answer。

---

## Slide 33. Method - Decompose Example
**Script:**

這張用前面的例子，把 decomposition 在 inference 裡怎麼運作講清楚。

原始問題是：擁有《The Collegian》的那所機構是什麼時候創立的？

系統會先拆成兩步。第一步先問 The Collegian 是由誰 owned，得到 Houston Baptist University。第二步再把第一步的答案接進來，問 Houston Baptist University 是什麼時候 founded，得到 1960。

這裡的重點是，sub-question 不是各自獨立的句子，而是一個可以一步一步執行的 plan。前一步的答案會被帶到後一步，所以如果前一步不確定，就需要先透過 retrieval 把它回答穩，再繼續往下走。

---

## Slide 34. Method - Inference Stage: Rule-Based Verifier
**Script:**

最後是 rule-based verifier。

Verifier 的輸入是兩個 candidate answers：一個來自 original-question direct path，另一個來自 decomposition path。
它的任務是根據 inference 過程中的 confidence，選出比較可靠的答案。

在目前的設定裡，decomposition path 是預設保留的路徑，因為對 multi-hop 問題來說，拆成 sub-questions 通常有助於 retrieval 和 reasoning。

但是如果 direct path 的 original-question answer 很有把握，而 decomposition path 在 retrieval 之後仍然不穩定，系統就會切回 direct path。

投影片上的條件就是這個 decision rule：

```text
Direct answer max entropy < 0.3
and
Decomposed sub-question post-retrieval max entropy > 0.8
```

第一個條件表示 original-question answer 很 confident。第二個條件表示 decomposition path 即使做了 retrieval，某些 sub-question 仍然很不確定。

因此大部分情況仍然相信 decomposition，但當 decomposition 顯示不穩定，而原始問題路徑又很有信心時，就選 direct answer。

這讓 SEEK 可以同時利用 decomposition 的好處，也保留避免 over-decomposition 的保護機制。
---

# Part 3 - Experiments

# 6/25 Thesis Presentation Speaker Script - Experiments

## Experiment Narrative

這一段的主線是：先給 final performance，再一步一步回答「這個結果從哪裡來」。不要讓實驗看起來像一串獨立表格，而是要讓每一頁都接到下一頁。

故事線可以這樣講：

1. SEEK 在四個 dataset 平均上超過 DeepRAG，但 HotpotQA 仍然低於 DeepRAG。
2. 為什麼平均能贏？先從 retrieval 層看，decomposition 確實提高 answer-containing recall。
3. 但是只拆解不夠，拆解後仍然需要 retrieval。
4. Entropy gate 的主要價值是 efficiency：接近 always-RAG 的 accuracy，但少做一部分 retrieval。
5. Path-aware title retrieval 解決不同 query 型態的 retrieval 差異：長 original question 和短 sub-question 不應該用同一個 title policy。
6. Original-question evidence union 補回 decomposition 可能丟掉的 global context，是 final performance 的重要提升來源。

---

## Slide 35. Experiment
**Script:**

接下來進入 Experiment。

前面 Methodology 說明 SEEK 怎麼做 decomposition、entropy-gated retrieval 和 final verification。實驗段要回答的是：整體表現如何、每個 component 帶來什麼效果，以及這個方法還有哪些限制。

---

## Slide 36. Experiment Setup
**Script:**

這張是實驗設定。

Backbone model 使用 `Llama-3-8B-Instruct`。SEEK 另外訓練一個 LoRA SFT decomposer，訓練資料是 12K decomposition examples，來自 MuSiQue、2Wiki 和 PopQA，每個 dataset 各 4K。

LoRA 設定是 rank 16、alpha 32、dropout 0.05，訓練 3 epochs。這裡的重點是，訓練只發生在 decomposer；
retrieval gate 和 verifier 都不是另外訓練出來的 classifier。

Evaluation datasets 包含四個：HotpotQA、2WikiMultihopQA、MuSiQue 和 PopQA。
前三個主要測 multi-hop reasoning，PopQA 則偏 single-hop factual QA，可以測試系統在簡單題上會不會過度拆解。

Metrics 使用 Exact Match，也就是答案字串完全符合；以及 token-level F1，也就是預測答案和 gold answer 在 token overlap 上的 F1。

Compared methods 包含 CoT、FLARE 和 DeepRAG。其中 DeepRAG 是最重要的 comparison，因為它也在 sub-question level 做 adaptive retrieval decision；差別是 DeepRAG 用 learned policy，而 SEEK 用 entropy signal 和 rule-based verifier。

---

## Slide 37. Main Result
**Script:**

這張是主要結果。

在四個 dataset 的平均上，SEEK 達到 `40.52 EM` 和 `47.61 F1`。相對 DeepRAG 的 `38.45 EM` 和 `45.75 F1`，平均提升是 `+2.07 EM` 和 `+1.86 F1`。

分 dataset 來看，SEEK 在 2Wiki、MuSiQue 和 PopQA 都超過 DeepRAG。2Wiki 從 DeepRAG 的 `48.10 EM` 提升到 `50.00 EM`；MuSiQue 從 `22.50` 提升到 `24.70`；PopQA 的提升最大，從 `42.50` 到 `51.90 EM`。

但 HotpotQA 是唯一低於 DeepRAG 的 dataset。SEEK 是 `35.50 EM`，DeepRAG 是 `40.70 EM`。這也很重要，因為它告訴我們：decomposition 不是在所有資料集上都穩定有利。
---

## Slide 38. Retrieval Recall

**Script:**
這頁的 goal 是評估 decomposition 是否真的改善 retrieval quality。我們想知道 decomposition 本身有沒有讓 evidence 更容易被找到。

這裡使用的是 answer-containing hit@5。意思是：top-5 retrieved passages 裡面，只要有一篇包含 gold answer ，就算 hit。

我們的結果是，decomposed-query retrieval 在大部分 dataset 上明顯高於 original-query retrieval。2Wiki 從 `31.73` 提升到 `62.38`，MuSiQue 從 `16.90` 提升到 `37.40`，HotpotQA 從 `46.55` 提升到 `56.54`。PopQA 幾乎持平，因為它本來就是 single-hop。

所以這頁的結論是：decomposition 的可以讓 retrieval 更容易找到 answer-containing evidence，特別是在 multi-hop questions 上。

**Transition:**
但是 retrieval hit 變好，不代表最後一定答對。下一步要問的是：拆解後到底需不需要 retrieval？

---

## Slide 39. Retrieval Is Necessary After Decomposition

**Script:**

這頁要回答的是：decomposition 之後，retrieval 到底是不是必要？

我們比較三個設定。

第一個是 `without-RAG`，也就是只做 decomposition，但每個 sub-question 都不檢索，只靠模型本身的 parametric memory 回答。

第二個是 `gate`，也就是 SEEK 的 entropy-gated setting。每個 sub-question 先回答一次，如果 max entropy 高，才觸發 retrieval。

第三個是 `always-RAG`，也就是每個 sub-question 都 retrieval。

這個 ablation 可以拆成兩個問題。`without-RAG` 對 `gate` 可以看 retrieval after decomposition 是否必要；
`gate` 對 `always-RAG` 則可以看 entropy gate 是否能在接近 always-RAG accuracy 的情況下降低 retrieval cost。

用這個設計，後面兩頁會分別看不同 threshold 下的 gate 效果。

---
## Slide 40. Gating Improves Efficiency, tau = 0.3
**Script:**

在這個 threshold 下，gate 和 always-RAG 的 EM 幾乎一樣。
HotpotQA 是 `25.0` vs. `25.1`，2Wiki 是 `47.2` vs. `47.1`，MuSiQue 是 `24.7` vs. `25.0`，PopQA 是 `51.5` vs. `51.4`。差距都非常小。

但 gate 並不是每一步都 retrieval。它的 retrieval rate 是 HotpotQA `92.7%`、2Wiki `95.2%`、MuSiQue `90.8%`、PopQA `99.3%`。

所以這裡要誠實定位 entropy gate：它不是主要 accuracy gain 的來源。主要 accuracy gain 來自 decomposition 加上 retrieval；entropy gate 的價值是效率，也就是用接近 always-RAG 的 accuracy，少做一部分 retrieval。

另外，如果對照 without-RAG，差距會很大：HotpotQA 從 `15.7` 到 `25.0`，2Wiki 從 `29.4` 到 `47.2`，MuSiQue 從 `9.6` 到 `24.7`，PopQA 從 `24.2` 到 `51.5`。這證明 retrieval after decomposition 是必要的。

**Transition:**

那 threshold 應該選多少？下一頁看完整 final configuration 的 threshold sensitivity。

---

## Slide 41. Threshold Sensitivity
**Script:**

這張表比較 final SEEK configuration 在不同 threshold 下的表現。

`tau = 0.3` 是目前的選擇，因為它給最高的 average result：`40.52 EM` 和 `47.61 F1`，平均 retrieval rate 是 `94.16%`。

當 threshold 變高時，retrieval rate 會下降，但 accuracy 也會小幅下降。比如 `tau = 0.8` 時，平均 retrieval rate 降到 `85.97%`，但是 average EM/F1 也降到 `39.98 / 47.04`。`tau = 1.0` 時 retrieval rate 進一步降到 `82.35%`，但 average EM/F1 變成 `39.50 / 46.63`。


分 dataset 看，HotpotQA 和 PopQA 對 threshold 比較不敏感；2Wiki 和 MuSiQue 掉得比較明顯，表示這些 multi-hop datasets 對 retrieval 更依賴。

**Transition:**

到目前為止，我們已經看到 decomposition 和 retrieval gate 的作用。接下來要看 retrieval implementation 本身：
原問題和 sub-question 的 query 型態不同，所以 retrieval setting 也不應該完全一樣。

---

## Slide 42. Path-Aware Title Retrieval

**This slide's goal:** 說明為什麼 direct path 和 decomposition path 要使用不同的 title retrieval policy。

**Script:**

這張是 path-aware title retrieval。

例如 `Houston Baptist University founded` 這種短 sub-question，核心就是一個 entity 加上一個 relation。
這時候讓 title 參與打分，可以更快找到正確的 Wikipedia article。

但 original question 通常比較長，裡面可能同時有多個 entity 和條件。
如果 title 也一起參與打分，retriever 有時候會被其中某個標題關鍵字吸走，而沒有真正抓到整個問題要問的關係。

所以我們採用 path-aware strategy：direct path 用原始問題檢索，title OFF；
decomposition path 用短的 sub-question 檢索，title ON。
重點不是 title matching 一定好或一定不好，而是不同 query type 要用不同 retrieval setting。

結果可以看到，path-aware 在四個 dataset 上都是最穩的設定。
PopQA 從 title-off 的 `51.2 / 53.5` 到 `51.8 / 54.2`；MuSiQue 從 `22.5 / 30.5` 到 `24.6 / 33.5`；2Wiki 從 `45.2 / 51.3` 到 `46.9 / 53.5`。HotpotQA 則避免了 global title-on 的下降，維持在 `31.8 / 41.2`。

這個實驗的意義是：adaptive RAG 不只要決定要不要 retrieve，也要讓 retrieval setting 符合 query type。

**Transition:**
但 decomposition 還有另一個問題：短 sub-question 可能丟掉原始問題的 global context。下一頁就是處理這件事。

---

## Slide 43. Original-Question Evidence Union Ablation

**Script:**

這張是 original-question evidence union ablation，也是 final system 裡很重要的改良。

Decomposition 的好處是把問題拆短，讓每一步比較容易檢索。但問題是，拆得太短時，也可能把原問題裡的線索拿掉。

所以我們把 original-question retrieval evidence 加回每個 sub-question 的 context。簡單來說就是：

```text
sub-question evidence = retrieve(sub-question) union retrieve(original question)
```

把 original question 當成另一個 retrieval query，讓每個 decomposed step 保留 global context。

結果平均提升 `+1.47 EM` 和 `+1.47 F1`。HotpotQA 從 `31.80 / 41.25` 提升到 `34.30 / 44.26`，2Wiki 從 `46.90 / 53.51` 提升到 `49.90 / 56.42`。PopQA 小幅提升，MuSiQue 的 EM 小幅提升但 F1 稍微下降。

這個結果支持一個重要結論：decomposition 不應該等於丟掉原問題。
最好的做法是同時保留兩種 evidence：sub-question 的精準 retrieval，以及 original question 的 global context。
---

# Part 4 - Analysis

# 6/25 Thesis Presentation Speaker Script - Analysis

## Slide 44. Analysis
**Script:**

接下來進入 Analysis。

前面的實驗說明了每個 component 的效果；這一段要回答更高層的問題：direct path 和 decomposition path 到底是不是互補？如果可以完美選路徑，還有多少提升空間？以及 entropy 這個 confidence signal 有什麼限制？

---


## Slide 45. Correct Answer Distribution Across Two Paths

**Script:**

這張表把每一題依照 direct path 和 decomposition path 的 correctness 分成四類。

`Both Correct` 表示兩條路都答對；`Direct only` 表示只有 original-question direct path 答對；`Decompose only` 表示只有 decomposition path 答對；`Both Wrong` 表示兩條路都答錯。

如果 decomposition 永遠比較好，我們應該看到 decompose-only 很大，而且 direct-only 幾乎沒有。但實際結果不是這麼單純。

在 2Wiki、MuSiQue 和 PopQA，decomposition path 明顯解到更多 direct path 解不到的題。以 2Wiki 為例，decompose-only 有 `361` 題，direct-only 只有 `39` 題。MuSiQue 也是類似，decompose-only 是 `184`，direct-only 是 `31`。這說明在需要 intermediate entity 的 multi-hop 問題上，decomposition 的確很有價值。

但是 HotpotQA 很不一樣。HotpotQA 的 direct-only 是 `111`，decompose-only 是 `113`，幾乎打平。這表示 HotpotQA 裡 direct path 和 decomposition path 各自救了一批題，沒有哪一條路完全支配另一條。

這也解釋了為什麼 HotpotQA 是比較困難的 dataset。有些 HotpotQA 題目在原始問題裡就有重要線索，例如題目同時提到哪兩個對象、要比較什麼。如果拆成太短的 sub-questions，這些線索可能會不見，系統就比較容易找錯資料，或在最後合成答案時選錯。

另外，both-wrong 這欄也很重要。HotpotQA 有 `584` 題兩條路都錯，MuSiQue 有 `714` 題兩條路都錯。這代表有一大部分錯誤不是光靠 router 可以解決，還需要更好的 retrieval、reranking、decomposition quality 或 final synthesis。

所以這頁的 takeaway 是：decomposition 很有用，但不是 always better。Direct path 和 decomposition path 是互補的，因此 final system 需要 routing，而不是固定永遠使用其中一條路。

**Transition:**

既然兩條路各自會解不同題，下一步自然要問：如果我們能完美選對路徑，最多還能提升多少？

---

## Slide 46. Direct-vs-Decompose Oracle Gap

**This slide's goal:** 量化 routing headroom，支撐「decomposition routing problem」這個 novelty。

**Script:**

這張是 direct-vs-decompose oracle gap。

先定義欄位。`Direct EM` 表示永遠選 direct answer。`Decomp EM` 表示永遠選 decomposed answer。`Current Router EM` 是目前 SEEK 的 verifier/router 選出來的結果。`Oracle EM` 則是事後用 gold answer 選 direct 或 decompose，代表不可部署的 upper bound。

所以 oracle 不是我們的方法，也不是 inference 時可以使用的策略。它的目的只是分析：如果 routing 完美，dual-path design 還有多少潛力。

先看 HotpotQA。Direct EM 是 `30.30`，Decomp EM 是 `30.50`，兩條固定路徑都差不多。但 current router 可以到 `35.50`，表示目前 verifier 已經救回一部分單一路徑的失敗。

不過 HotpotQA 的 Oracle EM 可以到 `41.60`，比 current router 高 `+6.10 EM`。這個數字很重要，因為它說明 HotpotQA 的剩餘差距不只是 decomposition 完全無效，也不只是 direct path 完全不行，而是「什麼時候該相信哪條路」還有很大空間。

2Wiki、MuSiQue 和 PopQA 也都有大約 `+3.8` 到 `+4.1 EM` 的 oracle gap。這表示就算在 decomposition dominant 的 datasets，上面仍然有一些題是 direct path 能救回的。

這頁最重要的 takeaway 是：SEEK 目前的 rule-based verifier 並沒有完全解決 routing problem。相反地，oracle gap 量化了這個問題還有多少 headroom。

這也就是 thesis novelty 的位置。Question decomposition 本身不是新的；但我們把 decomposition 放進 direct-vs-decompose routing framework，並且量化了這個 routing decision 的剩餘空間。

**Transition:**

那我們用來做 confidence 判斷的 entropy，到底能不能解決這個 routing 問題？下一頁看 entropy 的能力和限制。

---

## Slide 47. Entropy as a Confidence Signal

**This slide's goal:** 給 entropy 正確定位：有用，但不能被過度宣稱成完美 router。

**Script:**

這張是 entropy as a confidence signal。

SEEK 使用 entropy 的直覺是：如果模型在產生答案時不確定，next-token distribution 會比較分散，entropy 就會比較高。因此 high entropy 通常代表模型比較沒有把握，也比較可能需要 retrieval。

整體趨勢是成立的。把四個 dataset 合在一起看，higher entropy 通常 associated with higher error rate。這支持我們在 Methodology 裡用 max entropy 做 sub-question-level retrieval gate。

但是第二點也同樣重要：low entropy does not always mean correctness。

也就是說，模型有時候會很有信心地產生錯答案。這種 case 我們稱為 confident-wrong 或 confident-but-wrong。從 entropy 的角度來看，模型分布很集中，所以 entropy 很低；但從 correctness 來看，它其實答錯了。

因此 entropy 適合做 step-level retrieval gating，因為它可以偵測很多「模型不確定、可能需要外部知識」的 sub-questions。但它不足以單獨解決 whole-question direct-vs-decompose routing，因為有些錯誤不是 uncertainty，而是 confident wrong knowledge。

所以這頁要誠實收束：entropy 是 useful signal，不是 perfect signal。這也是為什麼 SEEK 最後保留 dual-path design 和 verifier，而不是只靠一個 entropy threshold 決定所有事情。

如果要用數字補充，可以口頭說：在 pooled analysis 裡，entropy 和 error rate 呈正相關，Spearman 大約是 `+0.326`；但低 entropy bins 仍然有錯誤，因此它不能被當成 correctness guarantee。

---

## Analysis Closing Transition

**Script:**

總結 Analysis，這裡有三個結論。

第一，direct path 和 decomposition path 是互補的。Decomposition 對很多 multi-hop questions 有幫助，但不是每一題都比 direct path 好。

第二，oracle gap 顯示 direct-vs-decompose routing 還有明確 headroom。這支持我們把 decomposition 視為 routing problem，而不是固定 always-decompose strategy。

第三，entropy 是有效的不確定性訊號，可以支撐 sub-question-level retrieval gate；但低 entropy 不保證正確，所以它不能單獨解決 whole-question routing。

接下來 Conclusion 會把這些結果收束成 SEEK 的主要貢獻：一個 lightweight adaptive retrieval framework，結合 question decomposition、entropy-gated retrieval 和 training-free verifier，在保持可解釋性的同時提升整體 QA performance。

---

## Optional Q&A Notes

### Q1. 這是不是代表 decomposition 沒有用？

不是。Retrieval recall 和 decompose-only counts 都顯示 decomposition 很有用。更精準的說法是：decomposition 對很多 multi-hop questions 有用，但不是每一題都最好，因此需要 routing。

### Q2. Oracle gap 是不是你的方法？

不是。Oracle 使用 gold answer 事後選 direct 或 decompose，所以不能部署。它只是用來量化如果 routing 做得更好，最多還有多少 headroom。

### Q3. Entropy 既然會失敗，為什麼還用？

因為 entropy 在 step-level retrieval gating 上仍然有效，而且成本低、可解釋。失敗的是把 entropy 當成唯一的 global router。SEEK 的設計是把 entropy 當作 confidence signal，而不是 correctness guarantee。

### Q4. HotpotQA 為什麼仍然輸 DeepRAG？

HotpotQA 的 direct-only 和 decompose-only 幾乎打平，代表它不是單純適合 always-decompose 的 dataset。它同時有 over-decomposition、context loss、retrieval drift 和 both-wrong bottleneck。Oracle 到 `41.60 EM` 說明 routing 還有空間，但 current router 只到 `35.50 EM`，所以這是主要 limitation。
---

# Part 5 - Conclusions

## Slide 48. Conclusions

**Script:**

接下來進入 conclusion。

前面的方法、實驗和分析可以收束成一個主軸：SEEK 把 adaptive RAG 裡的 retrieval decision 和 decomposition decision 拆開處理。

我們不是只問模型要不要 retrieve，而是進一步問：這一題應不應該拆解？拆解後的每一步要不要 retrieve？最後應該相信 original-question path，還是 decomposition path？

---

## Slide 49. Conclusion and Limitations

**Script:**

總結來說，SEEK 提供了一個 lightweight adaptive retrieval framework for question answering。

它結合了 question decomposition、entropy-gated sub-question retrieval，以及 training-free verifier。透過 decomposition，SEEK 可以把複雜問題轉成比較容易檢索的 sub-questions；透過 entropy gate，系統可以在 sub-question level 控制 retrieval；透過 dual-path 和 verifier，系統可以在 decomposition 不穩定時保留 original-question path。

實驗結果顯示，SEEK 在四個 dataset 的平均上超過 DeepRAG，特別是在 2Wiki、MuSiQue 和 PopQA 上有提升。這表示我們不一定需要一個很重的 learned retrieval policy，也可以做出有效的 sub-question-level adaptive retrieval。

不過，這個方法也有幾個限制。

第一，decomposition 不是永遠有幫助。它可以讓 retrieval 更精準，但也可能拿掉原問題裡重要的線索，造成後續錯誤。

第二，entropy 可以偵測很多不確定的情況，但它無法處理 confident-wrong errors。也就是說，模型有時候很有信心地答錯，這種情況 entropy 不一定抓得到。

第三，有些錯誤不是 routing 可以解決的。如果 direct path 和 decomposition path 都錯，系統就需要更好的 retrieval、reranking，或 final synthesis。

所以最後的 take-away 是：SEEK 證明 adaptive RAG 不只應該問 when to retrieve，也應該問 when to trust decomposition。這也是後續工作可以繼續改善的方向。

---

## Slide 50. Thanks

**Script:**

以上是我的 thesis presentation，謝謝各位老師。

接下來我很樂意回答問題。
---

# Appendix - DeepRAG Detailed Notes

# DeepRAG Detailed Speaker Script

Source paper: Guan et al., **DeepRAG: Thinking to Retrieve Step by Step for Large Language Models**, arXiv:2502.01142v2, 2025.  
Paper URL: https://arxiv.org/pdf/2502.01142

## Why This Script Exists

你的 thesis 主要 comparison method 是 DeepRAG，所以 Related Work 和 Appendix 裡要能講清楚它到底做了什麼。

DeepRAG 的核心不是單純「也有 decomposition」，而是：

```text
It trains the model to generate subqueries and decide, for each subquery,
whether to answer from parametric knowledge or retrieve external documents.
```

它的強點是 decision granularity 很對：**sub-question level**。  
它的成本是 training pipeline 比較重：**binary tree search data synthesis + imitation learning + preference calibration**。

這剛好是 SEEK 的切入點：SEEK 保留 sub-question-level retrieval decision，但用 entropy gate 和 rule-based verifier 來做 lightweight alternative。

---

## One-Slide Summary

**Slide title: DeepRAG: Learned Sub-Question-Level Retrieval Policy**

**Slide text:**

```text
DeepRAG models retrieval-augmented reasoning as an MDP.

At each step, the model decides:
1. Continue or terminate?
2. Retrieve external knowledge or use parametric knowledge?

Training pipeline:
• Binary Tree Search: find correct paths with minimal retrieval
• Imitation Learning: learn subquery generation and intermediate answers
• Chain of Calibration: preference-train atomic retrieve/parametric decisions

Inference:
The trained model generates subqueries and emits retrieval actions step by step.
```

**Speaker script:**

DeepRAG 的核心想法是把 retrieval-augmented reasoning 建模成一個 MDP。每一步模型都要做兩個決策：第一，要不要繼續產生下一個 sub-query，還是已經可以回答 final answer；第二，這個 sub-query 要不要 retrieve，還是直接用 parametric knowledge 回答。

它的 training pipeline 有三個部分。先用 Binary Tree Search 離線搜尋「能答對且 retrieval 次數最少」的 reasoning path；再用這些 path 做 imitation learning，讓模型學 subquery generation、atomic decision 和 intermediate answer；最後用 Chain of Calibration，也就是一種 preference training，去校準每一步 retrieve 或 parametric 的選擇。

Inference 的時候不再跑 binary tree search。訓練好的模型會自己一步一步產生 Follow up sub-query，並決定要不要輸出 search action。

---

## 1. DeepRAG's Motivation

**Slide title: Why DeepRAG Was Proposed**

**Slide text:**

```text
Two problems in iterative RAG:

• Poor task decomposition
  LLMs may generate vague or non-atomic subqueries.

• Redundant retrieval
  Not every subquery needs external knowledge.

DeepRAG goal:
Learn when to retrieve and when to rely on parametric knowledge at each step.
```

**Speaker script:**

DeepRAG 這篇論文的 motivation 有兩個。

第一，multi-hop QA 需要 decomposition，但 LLM 自己產生的 subquery 常常不夠 atomic，也不一定能形成一條穩定的 reasoning chain。

第二，iterative RAG 如果每一步都 retrieve，會造成 redundant retrieval。不是所有 subquery 都需要外部知識。有些 step 只是把前面答案做整理或計算，有些常識模型可能本來就知道。這時候 retrieval 不只增加成本，也可能帶入 noise。

所以 DeepRAG 的目標是讓模型在每個 subquery level 做 atomic decision：這一步到底要 retrieve，還是用 parametric knowledge 直接回答。

**Bridge to SEEK:**

這點跟 SEEK 的 motivation 很接近：decision granularity 應該在 sub-question level。差別是 DeepRAG 用訓練學這個 policy，SEEK 用 entropy signal 做 lightweight gate。

---

## 2. MDP Formulation

**Slide title: DeepRAG as an MDP**

**Slide text:**

```text
State:
Original question + previous subqueries + intermediate answers

Action:
σ ∈ {continue, terminate}
δ ∈ {retrieve, parametric}

Transition:
Generate next subquery, answer it with or without retrieved documents,
then update the reasoning state.

Reward:
Correct final answer with minimal retrieval cost.
```

**Speaker script:**

DeepRAG 把整個 reasoning process 寫成 MDP。

State 是目前的 partial solution，也就是 original question，加上目前已經產生過的 subqueries 和 intermediate answers。如果某一步有 retrieval，state 裡也會包含 retrieved documents 和基於 documents 產生的 answer。

Action 有兩個部分。第一個是 termination decision，記作 sigma，也就是要 continue 還是 terminate。第二個是 atomic decision，記作 delta，也就是這個 subquery 要 retrieve 還是 parametric。
