# MAX_ENTROPY vs Correctness

驗證「模型的 MAX_ENTROPY 與答對與否」之間的關係。

## 結論
- **MAX_ENTROPY 越低 → 越容易答對;越高 → 越容易答錯。**
- 即 MAX_ENTROPY 與「答錯」呈明顯正相關(Spearman r = **+0.326**, p≈1e-99, pooled n=4000),
  與「答對」呈負相關(point-biserial = −0.306),AUROC = 0.686。
- 四個資料集方向一致(見 `RESULTS.md`)。

## 資料來源
`/home/S113065528/Research/main_methodology/results/*_msample.jsonl`
(self-consistency 取樣結果),使用欄位:
- `mean_max_ent` — 每題的 MAX_ENTROPY 訊號(SC 各樣本 max-token-entropy 的平均)
- `majority` / `gold` — 判定對錯(沿用 self_certainty 的 normalized EM / token-F1≥0.5)

四個資料集:2WikiMultiHopQA、HotpotQA、MuSiQue、PopQA,各 1000 題。

## 執行
```bash
conda activate thesis   # 需要 matplotlib / numpy / scipy
python code/analyze_maxent_correctness.py
```

## 產出圖片(`figures/`)
| 檔案 | 內容 |
|---|---|
| `fig1_main_maxent_vs_accuracy.png` | **主圖**:準確率隨 MAX_ENTROPY 單調下降(equal-frequency bins + 95% Wilson CI) |
| `fig2_maxent_vs_error_positive.png` | 正相關視角:錯誤率隨 MAX_ENTROPY 上升 |
| `fig2b_bar_accuracy_down_error_up.png` | **長條圖對比**:每個 bin 答對率(綠)往下、答錯率(紅)往上,中間交叉 |
| `fig3_distribution_correct_vs_incorrect.png` | 答對/答錯兩群的 MAX_ENTROPY 分布(直方圖 + 小提琴圖) |
| `fig4_per_dataset_grid.png` | 四個資料集各自的準確率 vs MAX_ENTROPY |

## 程式
`code/analyze_maxent_correctness.py`(對錯判定與 AUC 邏輯沿用
`experiment/self_certainty/scripts/` 的 `exp4` / `exp5b`)。
