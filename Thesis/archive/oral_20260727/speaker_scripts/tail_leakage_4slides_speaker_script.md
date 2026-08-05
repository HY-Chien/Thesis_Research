# Tail-Leakage Score：四頁口試講稿

> 適用順序：① entropy 分解與訊號定位、② 為什麼使用 $-\log m_5$、③ 定義答案層級分數、④ 錯誤率驗證。
>
> 若「數學分解」與「三個 AUROC」仍分成兩張投影片，可將第一頁講稿在 AUROC 結果前換頁。

## 第 1 頁：Entropy 分解與訊號定位

前面我們先普查了 42 個常見的 confidence signals，觀察到一個一致的現象：越重視低機率尾部的指標，通常越能辨識錯誤。

這其實符合生成式 QA 的特性。即使答案正確，top candidates 之間仍可能因為實體別名、拼字或 tokenization 而互相競爭，所以頭部的不確定性不一定代表模型真的不知道答案。相較之下，當模型缺乏明確知識時，更多 probability mass 可能會從主要候選集合流到後面的 tail。

【指 PO／po 示意圖】

右邊用一個示意例子說明。模型前兩名可能是 `PO` 和 `po`，只是大小寫不同，語意上仍可能是同一個答案。左邊的 top-5 一共保留 97% probability mass，tail 只有 3%；右邊的前兩名看起來仍然很高，但 top-5 只保留 78%，有 22% 流到第六名以後。因此，我們不只問第一名是不是很高，而是看後面到底累積了多少占比。這組數字是概念示意，真正的結論來自左邊的 AUROC 實驗。

因此，我們進一步把 Shannon entropy 拆開，確認訊號究竟來自哪裡。

【指 top-5／tail 圖】

我們把 vocabulary 分成 top-5 和 tail，並令 $m_5$ 代表 top-5 的總機率。Entropy 可以精確分成三部分：

$$
H(p)
=
\underbrace{-m_5\log m_5-(1-m_5)\log(1-m_5)}
_{\text{top-5/tail mass split}}
+
\underbrace{m_5H(p_{\mathrm{top5}})}
_{\text{top-5 internal shape}}
+
\underbrace{(1-m_5)H(p_{\mathrm{tail}})}
_{\text{tail internal shape}}.
$$

第一部分看 probability mass 留在 top-5 還是流到 tail；第二部分看 top-5 裡面的候選有多分散；第三部分則看漏出去之後，tail mass 如何分布。

【指 AUROC bar】

為了分辨 mass 和 shape，我們把後兩個區域重新正規化，單獨測試它們的 internal shape，不乘 $m_5$ 或 $1-m_5$，避免把 mass 訊號重新混進來。

四個資料集的 macro AUROC 顯示，top-5／tail mass split 最高，為 0.742；top-5 shape 是 0.712；tail shape 只有 0.502，接近 random guessing。

所以真正重要的不是「漏出去之後怎麼分」，而是：

> 有多少 probability mass 留在 top-5，又有多少流到了 tail。

這一步讓我們把關鍵變數定位在 $m_5$。

**轉場：**

> 找到 $m_5$ 之後，下一個問題是：要怎麼把它轉成一個方向清楚的風險分數？

## 第 2 頁：From Top-5 Mass to Leakage Risk

Entropy decomposition 找到了 $m_5$，但沒有直接決定最後應該使用什麼計分函數。

我們希望風險分數具備一個直觀性質：

$$
m_5\downarrow
\quad\Longrightarrow\quad
\text{risk}\uparrow.
$$

也就是 top-5 留住的 mass 越少，tail leakage 越嚴重，分數就應該持續上升。

【指左圖】

但是 entropy 裡的 $-m_5\log m_5$ 是 probability-weighted surprisal。因為前面多乘了一個 $m_5$，它到達最高點後會轉向下降。也就是 leakage 已經非常嚴重時，分數反而可能變小。

這種函數適合描述 entropy contribution，但不適合描述單方向的風險程度。

【指右圖】

因此，我們保留 decomposition 找到的變數 $m_5$，但將風險函數設計為：

$$
-\log m_5.
$$

它在 $m_5=1$ 時等於零，而且：

$$
\frac{d(-\log m_5)}{dm_5}
=
-\frac{1}{m_5}<0.
$$

所以只要 top-5 mass 下降，風險分數就一定上升，不會在嚴重 leakage 時反向下降。

這裡要強調：

> $-\log m_5$ 不是從 entropy 代數化簡出來的。Entropy decomposition 用來找出關鍵變數 $m_5$；$-\log m_5$ 則是針對這個變數設計的單調風險函數。

**轉場：**

> 到這裡，我們已經得到每一個 token 的 leakage risk，接下來要把它變成整段答案的分數。

## 第 3 頁：Defining the Tail-Leakage Score

在答案的第 $t$ 個生成位置，我們重新根據當下的機率排序取 top-5，並計算其總機率 $m_{5,t}$。

單一 token 的 leakage risk 是：

$$
-\log m_{5,t}.
$$

但 QA 答案通常包含多個 token，而且錯誤不一定只出現在某一個特別差的位置，也可能是很多位置都各自漏出一點 probability mass。

因此，我們把所有生成位置的 leakage risk 相加：

$$
\boxed{
\mathrm{TailLeakage}(y)
=
\sum_{t=1}^{T}-\log m_{5,t}
}
$$

【指 token $t_1,t_2,t_3$】

每一個 token 都貢獻自己的 leakage risk；只要某個位置的 top-5 mass 降低，就會增加總分。多個位置的小 leakage，也會在答案層級累積起來。

我們也比較過不同的 aggregation：

- 加總的 macro AUROC 是 **0.749**；
- 只看最差 token 是 **0.742**；
- 取平均則是 **0.713**。

這表示錯誤訊號不只存在於單一最差 token，而是可能分散在整段答案中，因此使用 sum 最能保留累積效應。

另外，這個指標直接從 greedy decoding 已經產生的 logits 計算，不需要重新執行模型，也不需要額外 sampling。

**轉場：**

> 最後，我們要確認這個分數是否真的和答案錯誤有一致的關係。

## 第 4 頁：Higher Tail Leakage Means More Errors

【指 stacked bar chart】

這張圖將四個資料集共 4,000 筆 direct-path answers，按照 Tail-Leakage Score 分成八個等量區間。

綠色代表答對比例，紅色代表答錯比例。

可以看到，隨著 Tail-Leakage Score 上升，錯誤比例也持續增加。最低分區間的錯誤率大約是 33%；到了最高分區間，錯誤率上升到約 97%。

這表示 Tail-Leakage Score 不只是 AUROC 數字比較高，它的方向也非常直觀：

$$
\boxed{
\text{higher tail leakage}
\quad\Longrightarrow\quad
\text{higher error risk}
}
$$

它不是校準後的錯誤機率，而是一個 ranking signal；分數越高，模型越可能答錯。

所以完整的發現過程是：

$$
\text{tail-sensitive signals perform better}
$$

$$
\Downarrow
$$

$$
\text{entropy decomposition identifies }m_5
$$

$$
\Downarrow
$$

$$
-\log m_5\text{ provides monotonic token-level risk}
$$

$$
\Downarrow
$$

$$
\sum_t-\log m_{5,t}
\text{ forms the answer-level Tail-Leakage Score}.
$$

**收尾：**

> 簡單來說，我們不是直接修改原本的 entropy，而是先利用 entropy decomposition 找出訊號所在，再將這個訊號設計成一個單調、可累積，而且不需要額外推論成本的答案層級風險分數。
