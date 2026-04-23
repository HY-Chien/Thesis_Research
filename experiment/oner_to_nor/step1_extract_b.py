import json
import os

# ================= 設定區 =================
# 1. 分類器預測結果的路徑
PREDICT_FILE = "/home/S113065528/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40/predict/dict_id_pred_results.json"

# 2. NQ 原始測試集的路徑 (從這裡拿完整題目與答案)
ORIGINAL_NQ_FILE = "/home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl"

# 3. 輸出資料夾與檔案名稱
OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/oner_to_nor"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nq_all_questions.jsonl") # 檔名順便幫你正名為 all
# ==========================================

def extract_nq_questions():
    # 建立實驗用的輸出資料夾
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 讀取分類器預測結果，找出目標 ID
    target_ids = set()
    try:
        with open(PREDICT_FILE, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
            for qid, info in predictions.items():
                if info.get("dataset_name") == "nq":
                    target_ids.add(qid)
        print(f"🔍 從分類器預測結果中，共找到 {len(target_ids)} 個 NQ 題目的 ID。")
    except FileNotFoundError:
        print(f"❌ 找不到預測檔: {PREDICT_FILE}")
        return

    # 2. 讀取原始 NQ 測試集，對照 ID 抽取完整資料
    extracted_data = []
    found_ids = set() # 新增一個 set 來記錄到底找到了哪些 ID
    
    try:
        with open(ORIGINAL_NQ_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                qid = data.get("question_id")
                if qid in target_ids:
                    extracted_data.append(data)
                    found_ids.add(qid) # 記錄成功找到的 ID
        print(f"✅ 從原始測試集中，成功抽取到 {len(extracted_data)} 筆完整資料。")
    except FileNotFoundError:
        print(f"❌ 找不到 NQ 測試集檔案: {ORIGINAL_NQ_FILE}")
        return

    # 🕵️‍♂️ 新增功能：抓出消失的題目 ID
    missing_ids = target_ids - found_ids
    if missing_ids:
        print("\n" + "="*50)
        print(f"⚠️ 抓到蟲了！有 {len(missing_ids)} 題在預測檔中，但不在原始資料集裡：")
        for mid in missing_ids:
            print(f"   - 遺失的 ID: {mid}")
        print("="*50 + "\n")
    else:
        print("✅ 所有 ID 都完美對應！")

    # 3. 存成新的 jsonl 檔案，保持原本的格式以便後續評分
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for data in extracted_data:
            # 存檔時保留 jsonl 格式 (一行一筆 JSON)
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"📁 抽取完成！實驗檔案已儲存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_nq_questions()