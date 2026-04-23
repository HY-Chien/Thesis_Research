import json
import os
import pandas as pd
from collections import defaultdict

# ================= 設定區 =================
# 您指定的目標檔案路徑
TARGET_FILE = '/home/S113065528/Adaptive-RAG/experiment/entity_attack/result_epoch_30/dict_id_pred_results.json'

# 輸出統計結果的 CSV 檔名 (會存在當前目錄)
OUTPUT_CSV = 'prediction_stats.csv'
# =========================================

def main():
    print(f"正在讀取檔案: {TARGET_FILE}")
    
    if not os.path.exists(TARGET_FILE):
        print("錯誤: 找不到檔案，請確認路徑是否正確。")
        return

    try:
        with open(TARGET_FILE, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"讀取 JSON 失敗: {e}")
        return

    # 初始化統計容器
    # 結構: { 'nq': {'A': 0, 'B': 0, 'C': 0}, ... }
    stats = defaultdict(lambda: {'A': 0, 'B': 0, 'C': 0})
    
    # 開始遍歷統計
    for qid, info in data.items():
        dataset = info.get('dataset_name', 'unknown')
        prediction = info.get('prediction', 'unknown')
        
        # 只統計 A, B, C，如果有其他的標記為 Unknown
        if prediction in ['A', 'B', 'C']:
            stats[dataset][prediction] += 1
        else:
            # 預防萬一有異常值
            if 'Other' not in stats[dataset]:
                stats[dataset]['Other'] = 0
            stats[dataset]['Other'] += 1

    # ================= 輸出顯示 =================
    print("\n" + "="*50)
    print(f"{'Dataset':<15} | {'A (Zero)':<8} | {'B (Single)':<10} | {'C (Multi)':<8} | {'Total':<6}")
    print("-" * 50)

    rows = []
    
    # 排序資料集名稱
    for dataset in sorted(stats.keys()):
        count_a = stats[dataset].get('A', 0)
        count_b = stats[dataset].get('B', 0)
        count_c = stats[dataset].get('C', 0)
        total = count_a + count_b + count_c
        
        # 計算百分比 (可選)
        # pct_b = (count_b / total * 100) if total > 0 else 0
        
        print(f"{dataset:<15} | {count_a:<8} | {count_b:<10} | {count_c:<8} | {total:<6}")
        
        rows.append({
            "Dataset": dataset,
            "A": count_a,
            "B": count_b,
            "C": count_c,
            "Total": total
        })

    print("-" * 50)
    print(f"總共有 {len(data)} 筆資料被處理。")
    print("="*50)

    # 存成 CSV
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n統計結果已儲存至: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()