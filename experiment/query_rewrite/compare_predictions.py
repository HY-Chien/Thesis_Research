import json
import os
from collections import Counter

def main():
    # 1. 設定檔案路徑
    base_dir = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite"
    
    orig_file = os.path.join(base_dir, "experiment/query_rewrite/t5_large_original_nq_result_epoch_30/dict_id_pred_results.json")
    rewritten_file = os.path.join(base_dir, "experiment/query_rewrite/t5_large_rewritten_nq_result_epoch_30/dict_id_pred_results.json")
    output_file = os.path.join(base_dir, "diff_predictions.json")

    # 2. 讀取 JSON 資料
    print("正在讀取檔案...")
    with open(orig_file, 'r', encoding='utf-8') as f1:
        orig_data = json.load(f1)
        
    with open(rewritten_file, 'r', encoding='utf-8') as f2:
        rewritten_data = json.load(f2)

    # 3. 找出 prediction 不同的題目並進行統計
    diff_dict = {}
    transition_counter = Counter() # 統計轉移次數 (例: A->B)
    orig_counter = Counter()       # 統計原始預測標籤 (例: 原本 A 有幾筆)
    rewritten_counter = Counter()  # 統計改寫後預測標籤
    
    valid_total = 0 # 紀錄兩邊都有的題目總數

    for qid, orig_info in orig_data.items():
        if qid in rewritten_data:
            valid_total += 1
            orig_pred = orig_info["prediction"]
            rewritten_pred = rewritten_data[qid]["prediction"]
            
            # --- 新增：統計整體的標籤分佈 ---
            orig_counter[orig_pred] += 1
            rewritten_counter[rewritten_pred] += 1
            
            # 如果 prediction 不一樣
            if orig_pred != rewritten_pred:
                # 紀錄詳細資訊
                diff_dict[qid] = {
                    "original_prediction": orig_pred,
                    "rewritten_prediction": rewritten_pred,
                    "dataset_name": orig_info.get("dataset_name", "nq")
                }
                
                # 紀錄轉移方向 (例如 "A -> B")
                transition_key = f"{orig_pred} -> {rewritten_pred}"
                transition_counter[transition_key] += 1

    # 4. 將詳細差異寫入新的 JSON 檔
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(diff_dict, f_out, indent=4, ensure_ascii=False)

    # 5. 印出統計結果
    print("\n" + "="*50)
    print("📊 原始題目預測標籤分佈 (Original):")
    for label, count in sorted(orig_counter.items()):
        print(f"  標籤 {label} : {count} 筆")

    print("\n📊 改寫後題目預測標籤分佈 (Rewritten):")
    for label, count in sorted(rewritten_counter.items()):
        print(f"  標籤 {label} : {count} 筆")

    print("\n" + "="*50)
    print("🔄 預測標籤轉移統計 (Transitions):")
    for transition, count in transition_counter.most_common():
        print(f"  {transition} : {count} 題")
        
    print("-" * 50)
    print(f"共比對了 {valid_total} 題。")
    print(f"其中 {len(diff_dict)} 題 ({len(diff_dict)/valid_total*100:.1f}%) 的 prediction 發生改變。")
    print(f"詳細差異名單已儲存至: {output_file}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()