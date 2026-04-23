import json
import os

def main():
    # 設定檔案路徑 (已經將 original 和 perturbed 的路徑互換)
    perturbed_file_path = "/home/S113065528/Adaptive-RAG/experiment/entity_attack/t5_large_perturbed_2wikimultihopqa_result_epoch_30/dict_id_pred_results.json"
    original_file_path = "/home/S113065528/Adaptive-RAG/experiment/entity_attack/t5_large_original_2wikimultihopqa_result_epoch_30/dict_id_pred_results.json"
    
    # 設定輸出檔案路徑 (加上 .json 副檔名以便閱讀)
    output_file_path = "/home/S113065528/Adaptive-RAG/experiment/entity_attack/perturbed_prediction/t5_large_perturbed_prediction_2wikimultihopqa_diff.json"

    # 1. 讀取 Original 檔案
    print("正在讀取 Original 檔案...")
    with open(original_file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 2. 讀取 Perturbed 檔案
    print("正在讀取 Perturbed 檔案...")
    with open(perturbed_file_path, 'r', encoding='utf-8') as f:
        perturbed_data = json.load(f)

    diff_results = {}

    # 3. 找出所有的 key (以 original 為主，也可以取聯集)
    all_keys = set(original_data.keys()).union(set(perturbed_data.keys()))

    # 4. 開始比對
    for key in all_keys:
        # 取得 prediction，如果該 key 不存在則回傳 None
        pred_orig = original_data.get(key, {}).get("prediction")
        pred_pert = perturbed_data.get(key, {}).get("prediction")

        # 條件：兩個 prediction 不一樣，且其中至少有一個是 "B"
        if pred_orig != pred_pert and (pred_orig == "B" or pred_pert == "B"):
            diff_results[key] = {
                "original_prediction": pred_orig,
                "perturbed_prediction": pred_pert
            }

    # 5. 確保輸出目錄存在
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 6. 將結果寫入新的 JSON 檔案
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(diff_results, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"比對完成！共找到 {len(diff_results)} 筆 prediction 不一樣且與 B 相關的資料。")
    print(f"結果已儲存至: {output_file_path}")

if __name__ == "__main__":
    main()