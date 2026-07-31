# Same Leading Candidates, Different Tail Leakage — 口頭講稿

這一頁把比較條件控制得更簡單。左右兩邊的前兩名完全相同：`PO` 都是 0.38，`po` 都是 0.26，所以 top-2 合計都是 64%。

真正改變的是剩下的 36% probability mass 放在哪裡。

左邊的排名第三到第五分別是 0.15、0.10 和 0.08，合計 33%；只有 3% 跑到第六名以後。因此 top-5 retained mass 是 97%，主要候選集合仍然吸收了幾乎全部機率。

右邊的 `PO` 和 `po` 完全沒有改變，但排名第三到第五只剩 0.05、0.05 和 0.04，合計 14%；原本留在 head 的另外 19 個百分點移到了 tail，使 tail leakage 從 3% 上升到 22%。

如果只看前兩名，兩邊看起來一樣；但從 tail leakage 可以看到，右邊的模型仍然替更多主要候選之外的 token 保留機率，因此整體信心較低。

這是一組數學一致的示意分布，不是實際 dataset trace。它的目的是隔離 mass allocation 的差異。下一張 matched-entropy example 則進一步說明：即使總 entropy 相同，不確定性的來源仍可能不同。

一句話收尾：**leading candidates 一樣，不代表整體 confidence 一樣；差別可能在剩餘機率是否逃出 top-5。**
