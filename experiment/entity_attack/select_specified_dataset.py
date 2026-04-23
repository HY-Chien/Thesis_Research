import json
import os

# ================= 設定區 =================
# 輸入檔案路徑 (請確認路徑是否正確)
INPUT_FILE = '/home/S113065528/Adaptive-RAG/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json'

# 輸出檔案路徑 (存取所有的 NQ 資料)
OUTPUT_FILE = 'predict_original_2wikimultihopqa.json'

# 目標資料集名稱
TARGET_DATASET = '2wikimultihopqa'
# =========================================

def main():
    print(f"Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤：找不到檔案 {INPUT_FILE}")
        return

    # 讀取原始大檔
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    print(f"Total samples in file: {len(data)}")
    
    # 篩選出所有的 NQ 資料
    nq_data = [entry for entry in data if entry.get('dataset_name') == TARGET_DATASET]
    
    count = len(nq_data)
    print(f"\nExtracted {count} samples from dataset '{TARGET_DATASET}'.")
    
    # 存檔
    if count > 0:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(nq_data, f, indent=4)
        print(f"Saved all {TARGET_DATASET} samples to: {OUTPUT_FILE}")
    else:
        print(f"Warning: No samples found for dataset '{TARGET_DATASET}'.")

if __name__ == "__main__":
    main()