# 7/7 Progress Report 詳細講稿

對應檔案: `7_7_ProgressReport .pptx (1).pdf`  
建議用途: meeting 逐頁口頭報告。這版刻意寫得比較完整,實際講的時候可以依時間刪短。

## 整體主軸

今天這份 progress report 主要回答三件事。

第一,SEEK 的整體方法現在已經整理成 dual-path adaptive RAG:先產生 direct answer 和 decomposed answer 兩個候選,再用 confidence、decomposition stability、以及兩路答案是否一致來做最後選擇。

第二,entropy 確實能反映模型不確定性,但它不是正確性的保證。它適合當 retrieval gate 或風險排序訊號,不適合單獨當 whole-question correctness verifier。

第三,最近補上的幾個實驗更清楚地支持 decomposition 的價值:decomposition 能提升 retrieval recall,但主要瓶頸不是 decomposition text 本身,而是中間答案錯誤造成後續 query drift。因此 SEEK 的後續改進方向會集中在 bridge entity、retrieval robustness、以及 verifier。

---

## Slide 1: Progress Report

各位老師好,今天報告的是 7 月 7 日的進度。這次我主要把前一次老師提到的幾個問題補齊,包含 entropy 跟 accuracy 的關係、decomposition 對 retrieval 的實際幫助、gold decomposition 的 oracle 分析、以及 confidence-only baseline 能不能取代 decomposition。

這份報告的重點不是再介紹一個全新的方法,而是把 SEEK 的設計動機補得更完整:為什麼需要 decomposition、為什麼不能只靠 confidence、以及目前系統的瓶頸到底在哪裡。

---

## Slide 2: Method Overview

這一頁先回顧整體方法。SEEK 的核心設定是 dual-path:同一個問題會同時保留兩種回答方式。

第一條是 direct path,也就是直接針對原始問題回答。這條路徑的好處是問題裡的完整條件都還在,不會因為拆解而丟失比較條件或 disambiguation clues。缺點是對多跳問題來說,單一 query 常常 cover 不到完整 evidence chain。

第二條是 decomposition path,也就是把原問題拆成幾個 sub-question,每一步依序回答,需要時再檢索。這條路徑的好處是每個 sub-question 通常更短、更像 entity-centric query,所以 retrieval 比較容易打到正確文章。缺點是如果第一步 bridge entity 答錯,第二步 query 就會被帶歪,後面會產生 retrieval drift。

所以 SEEK 並不是假設 decomposition 永遠比較好,而是把問題改成:什麼時候該相信 direct path,什麼時候該相信 decomposition path。

---

## Slide 3: Method Overview: Signals and Routing

這一頁把 routing rule 講得更具體。

系統會先產生兩個 candidate answers:一個 direct answer,一個 decomposed answer。接著看三個訊號。

第一個是 direct-answer confidence。這裡主要用 answer token 的 entropy,值越低代表模型在生成 direct answer 時越有信心。

第二個是 decomposition stability。意思是 decomposition path 裡每一步的回答是否穩定,如果某一步很不確定,代表這條 evidence chain 可能有風險。

第三個是 direct answer 跟 decomposed answer 是否一致。如果兩條路徑給出同一個答案,通常風險比較低;如果兩者衝突,就需要 verifier 判斷哪條路徑比較可信。

目前的 rule 是比較保守的:如果 direct answer 很有信心,而 decomposition path 又不穩,就選 direct;其他情況多半偏向 decomposition。這個設計背後的想法是,多跳 QA 裡 decomposition 通常能帶來更好的 evidence path,但我們仍然保留 direct path 來處理 decomposition 會破壞問題條件的情況。

補充說法:這裡的 verifier 是 rule-based,不是額外訓練一個 policy router。這也是 SEEK 和 DeepRAG 的一個重要差異:SEEK 的 routing、gating、verification 都是 training-free 的規則,主要訓練成本集中在 LoRA decomposer。

---

## Slide 4: Recap: Correlation Between Entropy and Accuracy

接下來先回顧 entropy 和 accuracy 的關係。這裡我想回答老師之前問的問題:模型答對時的低 entropy,到底是真的有信心,還是只是剛好猜對?

目前的結論是:entropy 在整體上是有用的。也就是說,如果把很多題放在一起看,低 entropy 的題目確實比較容易答對,高 entropy 的題目比較容易答錯。這表示它不是完全沒有校準能力。

但是,entropy 量到的是答案分布的分散程度,不是推理路徑是否正確。換句話說,模型可以很穩定地收斂到一個錯答案,這時 entropy 會很低,但答案仍然是錯的。這就是後面會看到的 confident-but-wrong 問題。

所以這一段我會先立再破:先證明 entropy 有訊號,再說明它的限制在哪裡。

---

## Slide 5: Dataset-level AUROC Comparison

這張表用 AUROC 比較 entropy 對錯誤的辨別力。AUROC 可以理解成:隨機抽一個答錯題和一個答對題,entropy 能不能把答錯題排得更不確定。0.5 是隨機猜,1.0 是完美。

左欄是 query-level,也就是原問題 direct answer 的 entropy。可以看到四個資料集大概都在 0.716 到 0.813 之間,表示 direct answer 的 entropy 對是否答錯有穩定訊號。HotpotQA 是 0.813,2Wiki 是 0.781,MuSiQue 是 0.716,PopQA 是 0.748。

右欄是 step-level,也就是 decomposition 之後 sub-answer 的 entropy。這裡比較有趣:decomposition 之後訊號不是一致變好或變差,而是 dataset-dependent。MuSiQue step-level 反而提高到 0.809,PopQA 維持在 0.749,HotpotQA 下降到 0.696,但 2Wiki 掉到 0.540,幾乎接近隨機。

這表示 2Wiki 的 sub-question 常出現 confident-but-wrong:模型對某個中間答案很有信心,但那個中間答案其實錯了。所以對 2Wiki 來說,entropy 很難抓到這種錯誤。

這頁的重點是:entropy 不是壞訊號,但它的角色必須界定清楚。它可以做風險排序或 retrieval gate,但不能直接保證每一題正確。

---

## Slide 6: AUROC Example

這頁用一個小例子解釋 AUROC。假設有四題,A 和 C 答錯,B 和 D 答對。每一題都有一個 max entropy 分數。AUROC 做的是 pairwise comparison:把每個錯題拿去跟每個對題比,看錯題的 entropy 是否比較高。

例如 A 的 entropy 是 3.2,比 B 的 2.5 高,也比 D 的 0.4 高,所以這兩組都排對。C 的 entropy 是 1.8,比 D 高,所以 C-D 這組排對;但 C 比 B 低,所以 C-B 這組排錯。四組裡排對三組,AUROC 就是 3/4,也就是 0.75。

我放這頁是因為後面會一直用 AUROC。它不是在說某個 threshold 下正確率是多少,而是在說這個分數對「錯題通常比較不確定」這件事有多強的排序能力。

因此,當我們說 2Wiki step-level AUROC 只有 0.540,意思不是某個 threshold 選錯,而是整體排序就已經接近隨機。這就解釋了為什麼單純調 threshold 無法解決 2Wiki 的 confident-but-wrong。

---

## Slide 7: Results on Four Datasets

這頁是目前 final candidate 的主結果。設定是 Llama-3-8B-Instruct,SEEK 使用 path-aware retrieval、original-question evidence union,以及 type-aware verifier,gate threshold τ = 0.3。

平均來看,SEEK 達到 40.52 EM、47.61 F1,DeepRAG 是 38.45 EM、45.75 F1,所以平均提升是 +2.07 EM、+1.86 F1。

分資料集看,PopQA 的提升最大:SEEK 是 51.90 EM、54.71 F1,DeepRAG 是 42.50 EM、47.80 F1。2Wiki 也有提升,SEEK 是 50.00 EM、56.58 F1,DeepRAG 是 48.10 EM、53.25 F1。MuSiQue 也略高,SEEK 是 24.70 EM、33.06 F1,DeepRAG 是 22.50 EM、30.40 F1。

唯一落後的是 HotpotQA。SEEK 是 35.50 EM、46.11 F1,DeepRAG 是 40.70 EM、51.54 F1。這個 gap 後面也會呼應:HotpotQA 的原問題本身常常已經包含兩篇 supporting passages 的表面詞,所以 direct retrieval 比較強;decomposition 在 HotpotQA 上反而比較容易丟失原題的限制條件。

所以這頁的主張是:整體上 SEEK 是 competitive 的,平均超過 DeepRAG;但我們也不掩蓋 HotpotQA 的限制,而是把它當成分析 decomposition 什麼時候失效的重要案例。

---

## Slide 8: Retrieval Recall

接下來幾頁要回答的是:decomposition 為什麼有用?它是不是只是多做了一些步驟,還是真的讓 retrieval 變好?

這裡的 metric 是 answer-containing hit@5,也就是 top-5 retrieved passages 裡有沒有包含 gold answer 或 alias。這不是最後 QA 的 EM,而是 retrieval proxy,用來看檢索階段有沒有把答案相關 evidence 撈進來。

這個指標的好處是可以跨資料集比較,包含 PopQA 這種沒有完整 multi-hop supporting passage annotation 的資料集。它問的是一個比較基本的問題:retriever 至少有沒有把含答案的段落放進 top-5。

補充數字可以這樣講:四資料集上,decomposed query 的 answer-containing hit@5 都高於 original query。2Wiki 從 72.63 到 93.89,提升 +21.26;HotpotQA 從 81.08 到 87.89,提升 +6.80;MuSiQue 從 41.40 到 73.70,提升 +32.30;PopQA 從 57.30 到 68.20,提升 +10.90。這說明 decomposition 確實改善了 retrieval,尤其在 MuSiQue 和 2Wiki 這種 intermediate entity 很重要的資料集上最明顯。

---

## Slide 9: Oracle Decomposition Ceiling

這頁進一步問:如果 decomposition 是有用的,那 SEEK 的 decomposition 距離 oracle 還有多遠?

這個實驗用 MuSiQue,因為 MuSiQue 有官方 gold decomposition。我們比較四種 retrieval query。

第一個是 original query,也就是不拆解直接查原問題。answer-containing hit@5 是 19.0%。這是 baseline。

第二個是 SEEK decomposition,也就是用我們的 decomposer 產生 sub-questions,並且用模型自己預測的中間答案代入下一步。這個 hit@5 提升到 38.8%。所以 decomposition 本身把 retrieval recall 大幅拉高,大約是原問題的兩倍。

第三個是 gold decomposition 加上 gold bridge,也就是 oracle decomposition 和 oracle intermediate answer。這個可以到 55.8%。所以 SEEK 目前距離 oracle 還有大約 +17 個百分點的 headroom。

第四個是 gold raw,也就是保留 `#1` 這種 placeholder,不把第一步答案代入第二步。這時 hit@5 只有 19.8%,幾乎和 original query 的 19.0 一樣。

這個對照很重要,因為它說明 decomposition 的價值不是「把問題文字切開」而已,而是「拆解加逐步作答」形成一條 evidence chain。對 2-hop 問題來說,第二步 query 必須知道第一步 bridge entity 是誰,否則 gold decomposition text 本身也撈不到答案。

---

## Slide 10: Where Is the Gap?

上一頁看到 SEEK decomposition 是 38.8,gold decomposition 是 55.8,中間差了大約 17 個百分點。這頁要定位差距到底在哪裡。

我做了兩個 isolation experiments,把「decomposition text」和「intermediate answer」分開。

第一個 arm 是原本的 SEEK:decomposition text 是 SEEK 產生的,intermediate answer 也是模型預測的,hit@5 是 38.8。

第二個 arm 是 SEEK with answer:decomposition text 仍然用 SEEK 產生的,但把中間答案換成 gold bridge。這時 hit@5 直接上升到 54.1,幾乎追上 gold decomposition 的 55.8。

這代表 SEEK 的 decomposition text 不是主要瓶頸。只要 bridge entity 正確,SEEK 產生的 sub-question 幾乎可以達到 oracle decomposition 的 retrieval 效果。

下面這個表把題目依照 SEEK step-1 bridge 是否答對切開。step-1 答對的 424 題裡,SEEK hit@5 是 56.4,gold 是 58.5,兩者非常接近。step-1 答錯的 576 題裡,SEEK hit@5 掉到約 26%,但 gold 在同一批題上仍然有 53.8。

這表示不是那些題本身比較難檢索,因為 gold bridge 仍然可以撈到答案。真正的問題是第一步中間答案錯了,第二步 query 就跟著錯,形成 retrieval drift。

所以這頁的結論是:目前 retrieval ceiling 的 gap 主要來自 bridge entity error,不是 decomposer 文字品質。這也給出很明確的 future work:要改善 intermediate-answer verification、bridge grounding,或是在第二步 query 前做更強的 evidence-aware correction。

---

## Slide 11: Decomposer vs Base Model

這頁回答另一個問題:既然 decomposition text 不是瓶頸,那為什麼還需要 train decomposer?直接用 base model 拆不行嗎?

這裡用同一個 Llama-3-8B-Instruct base model、同一個 prompt,比較有沒有 LoRA decomposer training。

結果很明顯。SEEK decomposer 在 MuSiQue 1000 題裡,正確拆成 2 步的是 963 題;未微調 base model 只有 149 題。base model 很常過度拆解,把本來 2-hop 的問題拆成 3 到 11 步。

格式解析也差很多。SEEK 幾乎沒有 parsing failure,base model 有 153 題 parsing failure,也就是 15.3%。在實際 pipeline 裡,這些題的 decomposition path 會直接失效。

query style 也不同。SEEK decomposer 產生的是短的 KG-triple style query,比較 BM25-friendly;base model 會產生比較長的自然語言 query,有時還會把自己猜的 entity 放進 query,這會增加 drift 風險。

有趣的是,如果給 oracle bridge,base model 的 hit@5 是 52.9,SEEK 是 54.1,差距不大。但這裡有一個 caveat:base model 平均拆 2.98 步,SEEK 約 2.03 步,所以 base 其實用了大約 1.5 倍的 retrieval budget。

所以 decomposer training 買到的不是神奇的 retrieval ability,而是結構可靠性:步數正確、格式穩定、query 短、成本可控,也降低多一步就多一次 bridge error 的風險。

---

## Slide 12: Retrieval Is Necessary After Decomposition

前面證明 decomposition 能改善 retrieval。這一頁則回答另一個問題:拆解之後,還需不需要真的去檢索?能不能只靠 parametric knowledge 回答每一步?

這裡比較三種設定。without-RAG 是 decompose only,完全不檢索。gate 是 entropy 高的 step 才檢索。always-RAG 是每一步都檢索。

結果是,拆解後 retrieval 仍然是必要的。以 EM 來看,HotpotQA without-RAG 是 15.7,gate 變成 25.0,提升 +9.3。2Wiki 從 29.4 到 47.2,提升 +17.8。MuSiQue 從 9.6 到 24.7,提升 +15.1。PopQA 從 24.2 到 51.5,提升 +27.3。

也就是說,decomposition 讓問題變成比較好的 sub-query,但它不等於已經知道答案。真正把 factual knowledge 補進來的還是 retrieval。

這頁我想強調兩層意思。第一,只拆不查不夠,所以 SEEK 不是 pure parametric decomposition。第二,gate 的角色不是讓 accuracy 超過 always-RAG,而是在接近 always-RAG accuracy 的情況下降低一點 retrieval cost。

---

## Slide 13: Gating Improves Efficiency

這頁把 gate 和 always-RAG 比較。τ = 0.3 是目前 canonical setting。

可以看到 gate 的 EM 和 always-RAG 幾乎一樣。PopQA gate 是 51.5,always 是 51.4,差 +0.1。HotpotQA gate 是 25.0,always 是 25.1,差 -0.1。2Wiki gate 是 47.2,always 是 47.1,差 +0.1。MuSiQue gate 是 24.7,always 是 25.0,差 -0.3。

也就是說,gate 跟 always 的差距都在 0.3 EM 以內,但 gate 不是每一步都檢索。retrieval rate 分別是 PopQA 99.3%、HotpotQA 92.7%、2Wiki 95.2%、MuSiQue 90.8%。

所以這頁的解讀要很精準:gate 不是 accuracy improvement 的主張,因為 always-RAG 幾乎同樣準,有時還略高。gate 的主張是 efficiency:在不太損 accuracy 的情況下,少做一部分檢索。

如果被問「為什麼檢索率還是 90% 以上」,可以回答:在 honest wiki-21M open-domain setting 下,多數 sub-question 都需要 evidence,所以 gate 自然接近 always-open。這不是壞事,因為它表示系統沒有為了省成本而犧牲準確率。它比較像一個 conservative efficiency knob。

---

## Slide 14: Threshold Sensitivity

這頁看 τ 改變時的 cost-accuracy trade-off。τ 越高,代表要更不確定才檢索,所以 retrieval rate 會下降,但 accuracy 也可能下降。

τ = 0.3 是目前 final setting,平均 40.52 EM、47.61 F1,平均 retrieval rate 是 94.16%。τ = 0.5 時,平均 retrieval rate 降到 91.32%,EM 幾乎不變,是 40.45。這表示 τ = 0.5 幾乎是免費省一點成本,但省得不多。

τ = 0.8 時,retrieval rate 降到 85.97%,平均 EM 變成 39.98,大概掉 0.54 EM。τ = 1.0 時,retrieval rate 降到 82.35%,平均 EM 變成 39.50,大概掉 1.0 EM。

分資料集看,HotpotQA 和 PopQA 對 threshold 比較不敏感。2Wiki 和 MuSiQue 比較依賴 retrieval,所以 τ 拉高之後掉得比較明顯。2Wiki 從 50.00 掉到 48.10,MuSiQue 從 24.70 掉到 22.60。

因此 canonical 選 τ = 0.3 的理由是 accuracy-first:它是平均 EM/F1 最好的點。若未來要做 cost-saving variant,τ = 0.8 是一個合理中間點,可以省約 8 個百分點 retrieval rate,但會掉約 0.5 EM。論文主表仍然維持 τ = 0.3,比較穩健。

---

## Slide 15: Can Direct Gate Replace Decomposition?

這頁回答一個很重要的 baseline:如果不做 decomposition,只用 confidence 決定要不要 RAG,能不能達到類似效果?

baseline 是 direct-gate:先讓模型 no-RAG 直接回答原問題,量 max entropy。如果 entropy 低,就採用 parametric answer;如果 entropy 高,才檢索原問題後重答。這和 SEEK 的 gate 用同一個訊號,差別是它完全不拆解。

結果顯示,在三個資料集上 direct-gate 明顯追不上 decompose-gate。2Wiki direct-gate 最好的 τ 只有 18.6 EM,SEEK decompose-gate 是 47.2,差 +28.6。MuSiQue direct-gate 是 10.8,SEEK 是 24.7,差 +13.9。PopQA direct-gate 是 39.2,SEEK 是 51.5,差 +12.3。

HotpotQA 是例外:direct-gate 是 30.7,decompose-gate 是 25.0。這和前面的分析一致,HotpotQA 的原問題本身常常就是很強的 retrieval query,而 decomposition 可能丟失一些原題限制。不過完整 SEEK 不是只用 decompose-gate,而是 dual-path verifier,所以 final SEEK 在 HotpotQA 是 35.5,高於 direct-gate 30.7。

這頁的重點是:不是 confidence signal 不好,而是不拆解時候選答案的天花板太低。direct-gate 只能在「parametric direct answer」和「single-query retrieval answer」之間選,但多跳問題裡這兩個候選常常都錯。真正把 candidate quality 拉高的是 decomposition,不是 gate 本身。

可以補一句:query-level entropy 的 AUROC 其實不差,大約 0.72 到 0.81,所以問題不在 signal 沒有資訊,而是在沒有 decomposition 時,就算 signal 選對了要檢索,單一原問題檢索也常常 cover 不到第二跳 evidence。

---

## Slide 16: Path-aware Title Retrieval

這頁是 path-aware title retrieval 的 ablation。這裡先定義 title matching:Wikipedia index 裡每個 passage 有 article title 和 passage text。Title-on 表示 BM25 同時比對 title 和 text;title-off 表示只比對 passage text。

觀察是:短 sub-question 通常接近一個 entity name,所以 title matching 很有幫助。例如 sub-question 像「Adriana Trigiani 的工作地點」,query 裡的 entity 幾乎就是正確文章標題,開 title 可以直接打到正確文章。

但長 original question 不一樣。原問題裡有很多 content words、constraints、relation words,這些字每一個都可能剛好撞到不相關文章的 title。開 title 後,不相關但標題剛好有同樣詞的文章會被拉進 top results,反而形成 noise。

所以策略是:direct path 的 original question 關 title;decomposition path 的 sub-question 開 title。這不是依照資料集 cherry-pick,而是依照 query type:長原題關 title,短子題開 title。

結果也支持這個設計。PopQA title-off 是 51.2/53.5,title-on 是 51.5/53.9,path-aware 是 51.8/54.2。HotpotQA title-off 是 31.9/41.2,title-on 會掉到 30.5/39.9,path-aware 回到 31.8/41.2。MuSiQue path-aware 是 24.6/33.5,比另外兩個都好。2Wiki path-aware 是 46.9/53.5,也比 title-off 和 global title-on 更好。

這頁可以用一句話收束:path-aware title retrieval 吃到兩邊好處,讓 entity-like sub-question 受益於 title matching,同時避免長 original question 被 stray title hit 污染。

如果被追問例子,可以講 HotpotQA 的 Big Stone Gap 例子。原題裡有 romantic comedy,開 title 後會撈到名叫 Romantic Comedy 的舞台劇和 1983 年電影,跟題目完全無關,卻擠掉真正需要的 Adriana Trigiani 頁面。這就是長 query 開 title 的風險。

---

## Slide 17: Action Item

這頁是目前 action items 的進度。

E-23 是 survey question decomposition methods,目前還在進行中。這部分主要是補 related work,整理其他方法如何做 decomposition,以及它們是否有 train router、是否依賴 retrieved evidence。

E-24 是 evaluate our method across different models,也還在進行中。這個項目會用來確認 SEEK 的設計是不是只對 Llama-3-8B-Instruct 有效,還是能跨 backbone 成立。

E-25 是 analyze confidence scores in correct and wrong cases,已完成。這部分對應今天前面的 entropy/AUROC、both-correct vs both-wrong,以及 confident-but-wrong 的分析。

E-26 是 analyze decomposition errors and confidence-based retrieval limitations,也已完成。這部分對應 bridge entity error、retrieval drift、以及 entropy 在 2Wiki step-level 接近隨機的限制。

E-27 是 use a confidence-only baseline to show the benefit of decomposition,已完成。這就是今天 direct-gate baseline 那頁,證明不拆解只靠 confidence gate 無法取代 SEEK 的 decompose+gate。

接下來我會把這幾個已完成分析整合回 thesis 的章節,並且把未完成的 cross-model evaluation 和 related work survey 補齊。

---

## Slide 18: Thanks

以上是今天的報告。總結來說,這次補實驗後,SEEK 的定位更清楚了。

第一,decomposition 確實提升 retrieval,特別是在需要 intermediate entity 的資料集上。

第二,目前 decomposition 的主要瓶頸不是 query text 本身,而是 bridge entity 答錯後造成 retrieval drift。

第三,entropy 是有用的訊號,但它有明確限制:它可以做 retrieval gate 和風險排序,但不能保證答案正確,也不能單獨解決 direct-vs-decompose routing。

第四,confidence-only direct gate 追不上 decompose+gate,所以 SEEK 的提升主要來自 decomposition 改善 candidate quality,不是單純 threshold tuning。

接下來我會繼續補 related work、跨模型評估,以及針對 bridge entity 和 verifier 的後續改進。

---

## 備用問答

### Q1. Entropy 既然會 confident-wrong,那為什麼還要用?

可以回答:因為 entropy 在 aggregate level 仍然有效。direct entropy 的 both-correct vs both-wrong pooled AUROC 約 0.78,五分位校準也呈單調趨勢。所以它不是沒用,而是不能被過度解讀。SEEK 把 entropy 放在比較保守的位置:做 per-step retrieval gate 和風險訊號,不是單獨做 whole-question correctness verifier。

### Q2. Gate retrieval rate 都 90% 以上,是不是沒有省多少?

可以回答:是,在 honest open-domain corpus 下,gate 的省檢索幅度是保守的,大約省 0.7% 到 9.2%。所以我不把它包裝成大幅降低 inference cost,而是說它在幾乎不傷 accuracy 的情況下提供 efficiency knob。SEEK 的主要效率主張應該是 training-free routing 和不需要 DeepRAG 那種 policy training,不是每題檢索一定比較少。

### Q3. 為什麼 HotpotQA 仍然輸 DeepRAG?

可以回答:HotpotQA 是 SEEK 最明確的限制案例。它的問題通常由標註者看著兩篇 supporting passages 寫出來,所以原問題本身包含很多 supporting evidence 的表面詞,單 query retrieval 已經很強。decomposition 有時反而會丟失 comparison condition 或 disambiguating clues。因此 HotpotQA 上 direct path 很重要,final verifier 已經把 decompose-only 25.0 拉到 35.5,但仍低於 DeepRAG 40.7。這表示剩下問題在 routing、retrieval drift、以及 both-wrong bottleneck。

### Q4. Evidence union 會不會讓 direct path 和 decomposition path 變成同一條?

可以回答:不會。union 共享的是 retrieval pool,不是 reasoning path。而且共享是單向的:decomposition path 可以借用 original-question retrieved documents,但 direct path 不會看到 decomposition 的 sub-question evidence。decomposition path 仍然會做自己的 sub-question retrieval、step-by-step answering 和 final synthesis。所以兩條 path 的差別仍然存在,只是 decomposition path 補回原問題中可能被拆掉的全域線索。

### Q5. Train decomposer 的價值是什麼?

可以回答:訓練主要買到結構可靠性,不是讓 query text 神奇變好。未微調 base 只有 149/1000 題拆成正確 2 步,而 SEEK decomposer 是 963/1000;base 還有 153/1000 formatting failure。即使給 oracle bridge,base 的 retrieval hit@5 也只是 52.9,而且用了 1.5 倍 retrieval budget。LoRA decomposer 的價值是步數穩、格式穩、query 短,降低 drift 和推論成本。

### Q6. Direct-gate baseline 已經用了 confidence,為什麼還輸?

可以回答:因為 direct-gate 的候選集合太弱。它只能在 no-RAG direct answer 和 original-query RAG answer 中選,但多跳問題常常需要 intermediate entity,單一原問題檢索 cover 不到完整 evidence chain。即使 confidence signal 本身有 AUROC 0.72 到 0.81,選對「要檢索」也不代表 single-query retrieval 能找到第二跳。decomposition 改善的是 candidate quality,這是 gate 取代不了的。

