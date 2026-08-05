# Same Entropy, Different Risk — 口頭講稿

口委可能會問：原本的 entropy 本來就會看完整 vocabulary，為什麼還需要另外看 tail？

這個問題是對的。我們的論點不是 entropy 忽略 tail，而是 entropy 同時受到 tail 裡有多少總機率，以及這些機率如何分散所影響。

這兩張圖都是按照 token rank 排列的機率分布。橘色是 head，也就是前五名候選；藍色是 rank 6 之後的 tail。兩張圖使用相同的垂直尺度，所以柱高可以直接比較。

先看前兩根橘色柱子：A 和 B 的 `PO` 都是 0.38，`po` 都是 0.26，因此模型最主要的兩個判斷完全相同。兩組 Shannon entropy 四捨五入後也相同，都是 1.83。

差異在藍色部分。A 的 tail 總機率只有 10%，分散在 12 個很小的 tokens 上，每個約 0.83%；B 的 tail 總機率是 20%，集中在 5 個 tokens 上，每個是 4%。所以右圖的藍色柱子明顯較高，而流入 tail 的總機率剛好是左圖的兩倍。

因此，兩組 leading candidates 相同，總 entropy 也相同，但落在 top-5 之外的總機率相差一倍。Entropy 之所以相同，是因為 A 較小但較分散的 tail，和 B 較大但較集中的 tail，在 entropy 裡互相抵銷。

這就是我們要分開 tail amount 和 tail shape 的原因。Entropy 看得到兩者，但只輸出一個總數；tail leakage 則直接回答有多少機率落在主要候選集合之外。

下一頁會把 entropy 精確拆成三塊：head／tail probability split、head internal shape，以及 tail internal shape；接著用 AUROC 驗證哪一塊真正與錯誤相關。

如果口委直接追問，可以用一句話回答：

> Entropy sees the full distribution, but it mixes tail amount with tail shape. Tail leakage isolates how much probability lies outside the top candidate set.
