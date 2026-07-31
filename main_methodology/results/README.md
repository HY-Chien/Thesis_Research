# results/ 結構

> 2026-07-31 整理:原本 500+ 檔案平鋪,現依類型/資料集分層。
> **注意**:`scripts/` 內舊分析腳本寫死的是整理前的平鋪路徑(`results/xxx.jsonl`);
> 重跑歷史腳本時需自行對應到新位置。RESULTS_MASTER.md 與 ADVISOR_FEEDBACK_QA.md 的引用已全部更新。
>
> **橋接技巧**(2026-07-31 驗證可行):跑舊腳本前在 results/ 建 symlink 指向新位置,跑完刪除:
> ```bash
> # 例:replay_verifier_tlsum_fixA.py 需要的 16 個檔
> for ds in hotpotqa 2wiki musique popqa; do
>   for f in "${ds}_1000_verify_v2_pathaware_directctx_typeaware_replay.jsonl" \
>            "tlsum_fixA_${ds}_shard0.jsonl" "tlsum_fixA_${ds}_shard1.jsonl" \
>            "conf_battery_${ds}.jsonl"; do
>     ln -sf "traces/$ds/$f" "results/$f"; done; done
> # 用畢:find results -maxdepth 1 -type l -delete
> ```
> 以此流程已完整重現 RESULTS_MASTER §4.19 final adopted headline(41.62/48.66,逐位一致)。

| 位置 | 內容 |
|------|------|
| `traces/{hotpotqa,2wiki,musique,popqa}/` | 各資料集的實驗 trace(jsonl;依檔名中的資料集關鍵字分類) |
| `traces/other/` | 跨資料集 / smoke test / manifest 類 jsonl |
| `reports/` | 分析報告 md(confidence battery、failure taxonomy、sweep 摘要等)+ sweep 文字摘要 |
| `tables/` | 最終表格(xlsx / tsv / json) |
| `figures/` | 圖檔 |

正式結果的權威彙整在上層 `../RESULTS_MASTER.md`(2026-06-13 起唯一正典,fragment md 已收斂於此)。
