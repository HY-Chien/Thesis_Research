import json
import numpy as np

# 檔案路徑
DIFF_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/diff_predictions.json"
PPL_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/rewrite_comparison_ppl.json"

def analyze_ppl_changes():
    # 載入預測轉變資料
    with open(DIFF_FILE, 'r') as f:
        diff_data = json.load(f)
    
    # 分類 ID
    b_to_a_ids = [qid for qid, v in diff_data.items() if v["original_prediction"] == "B" and v["rewritten_prediction"] == "A"]
    b_to_c_ids = [qid for qid, v in diff_data.items() if v["original_prediction"] == "B" and v["rewritten_prediction"] == "C"]

    # 載入 PPL 比較資料
    with open(PPL_FILE, 'r') as f:
        ppl_data = json.load(f)
    
    # 建立 ID 到 ppl_diff 的映射
    ppl_map = {item["id"]: item["ppl_diff"] for item in ppl_data if "id" in item}
    
    # 計算平均值
    def get_avg(id_list):
        diffs = [ppl_map[qid] for qid in id_list if qid in ppl_map]
        return np.mean(diffs) if diffs else 0, len(diffs)

    avg_all = np.mean([item["ppl_diff"] for item in ppl_data])
    avg_ba, count_ba = get_avg(b_to_a_ids)
    avg_bc, count_bc = get_avg(b_to_c_ids)

    print("="*40)
    print(f"📊 PPL 變化統計 (ppl_diff = 原 PPL - 新 PPL)")
    print("-"*40)
    print(f"🔹 整體平均改善 (All): {avg_all:.4f}")
    print(f"🔹 B -> A 平均改善: {avg_ba:.4f} (樣本數: {count_ba})")
    print(f"🔹 B -> C 平均改善: {avg_bc:.4f} (樣本數: {count_bc})")
    print("-"*40)
    print("註：正值代表重寫後 PPL 下降（變流暢），負值代表 PPL 上升。")
    print("="*40)

if __name__ == "__main__":
    analyze_ppl_changes()