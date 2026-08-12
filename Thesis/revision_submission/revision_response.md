# 口試意見回覆與修改頁碼

> 本文件僅列出實際反映在論文中的修改。頁碼以修改後論文內文頁面下方的頁碼為準，不是 PDF 閱讀器顯示的實體頁序。

## Comment（To-do List）

1. Related Work 目前較偏重於介紹既有方法，尚未明確指出其 retrieval signals 的限制，因此難以充分建立 Tail Leakage 的研究動機。建議補充既有方法的不足，並說明本研究為何需要不同的信心指標。

2. 建議補充未使用 Tail-Leakage Surprisal 作為 confidence signal 時的系統表現，例如改用 entropy、log-probability 或其他信心指標進行比較。

3. SEEK 相較於 Always Retrieve 減少約 6–7% 的 retrieval，但目前尚不清楚這些被省略檢索的題目原先是答對還是答錯。建議進一步分析這些題目的作答情形。

4. 建議補充幾個重點：(1) Tail Leakage 的研究動機與直觀範例；(2) Shannon Entropy 與 Rényi Entropy 在 LLM 機率分布上的概念差異。

## Response

1. 已在 Related Work 補充既有方法的限制：只對整題判斷是否檢索的方法，較難處理多步推理中每個步驟的不同需求；依字詞信心判斷的方法，可能把用字變化誤認為知識不足；需要額外訓練的方法，則較難直接解釋每次檢索的原因。這些討論進一步說明 SEEK 為何在每個子問題分別判斷是否檢索，並採用 Tail Leakage 作為信心依據。（修改位置：Chapter 2「Adaptive RAG」，p. 7；「Reasoning and Knowledge Boundaries in RAG」，p. 10。）

2. 已補充其他信心指標的比較，包括 entropy、答案機率、重複生成結果的一致性，以及模型對自己答案的判斷。結果顯示，Tail Leakage 辨識錯誤答案的平均 AUROC 為 0.749，高於 entropy 的 0.726；使用 Tail Leakage 時，整體 EM／F1 也由 40.52／47.61 提升至 41.88／48.90。（修改位置：Chapter 5「Tail Leakage as a Confidence Signal」，pp. 37–38；Table 5.10 為錯誤辨識能力比較，Table 5.11 為最終答案表現比較。）

3. 已進一步分析 SEEK 省略的檢索是否真的不影響答案。6.16% 指的是全部 7,761 個推理步驟中，省略了 478 次檢索，而不是省略 6.16% 的題目；這些省略分布在 418 題中。逐題比較後，402 題（96.17%）沒有因省略檢索而降低正確性，僅 16 題（3.83%）失去原本正確的答案。（修改位置：Chapter 5「Ablation Studies」，pp. 31–32；Table 5.4 呈現安全省略與錯誤省略的比例。）

4. 已在 Chapter 3 補充 Tail Leakage 的設計動機與直觀範例。論文先說明：模型有把握時，機率通常集中在少數可能答案；模型缺乏知識時，機率會散到許多不相關字詞。接著比較 Shannon Entropy 與 Rényi Entropy：Shannon Entropy 會同時受到高機率候選與低機率尾端影響，Rényi Entropy 則可以調整對兩者的重視程度。實驗結果顯示，越重視低機率尾端，越能辨識錯誤答案，因此 SEEK 最後保留候選答案之外的機率外溢資訊，形成 Tail Leakage。論文也加入「兩個分布具有相同 Shannon Entropy、但尾端風險不同」的例子，說明 Tail Leakage 能區分傳統 entropy 無法區分的情況。（修改位置：Chapter 3「Token-Level Confidence Probe」，pp. 18–22；Figures 3.4–3.7 分別呈現 Rényi 權重比較、相同 entropy 的範例、Tail Leakage 分數變化與錯誤率分布。）

## 附件

- **main.pdf**：論文乾淨版。
- **main_diff.pdf**：論文差異標注版；新增內容以藍色標示，刪除內容以刪除線標示。
