import json
from collections import Counter

def main():
    # 設定檔案路徑
    file_path = "/home/S113065528/Adaptive-RAG/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/flan_t5_xl/binary_silver/train.json"

    print(f"正在讀取檔案: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("找不到檔案，請確認路徑是否正確。")
        return

    # 1. 篩選 dataset_name 為 "nq" 的資料，並取出所有的 answer
    nq_answers = [item.get("answer") for item in data if item.get("dataset_name") == "nq"]

    # 2. 統計各個 answer 的數量
    answer_counts = Counter(nq_answers)

    # 3. 印出結果
    print("-" * 30)
    print("資料集 'nq' 的 answer 統計結果：")
    
    # 特別列出 A 和 B (如果沒有則顯示 0)
    print(f"Answer A: {answer_counts.get('A', 0)} 筆")
    print(f"Answer B: {answer_counts.get('B', 0)} 筆")
    
    # 順便列出 C 或其他可能出現的標籤
    for ans, count in sorted(answer_counts.items()):
        if ans not in ["A", "B"]:
            print(f"Answer {ans}: {count} 筆")

    print("-" * 30)
    print(f"總共找到 {len(nq_answers)} 筆 'nq' 的資料。")

if __name__ == "__main__":
    main()