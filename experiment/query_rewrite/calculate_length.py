import json

# 設定檔案路徑
file_path = '/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/rewrite_comparison_ppl.json'

try:
    # 讀取 JSON 檔案
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立表格標題 (改成 Words)
    print(f"{'ID':<25} | {'Original Words':<15} | {'Rewritten Words':<16}")
    print("-" * 62)

    total_original_words = 0
    total_rewritten_words = 0
    count = len(data)

    # 走訪每一筆資料並計算單字數量
    for item in data:
        item_id = item.get('id', 'Unknown')
        orig_query = item.get('original_query', '')
        rewritten_query = item.get('rewritten_query', '')

        # 使用 split() 以空白切割字串，並計算單字數量
        orig_words = len(orig_query.split())
        rewritten_words = len(rewritten_query.split())

        # 累加總單字數以計算平均
        total_original_words += orig_words
        total_rewritten_words += rewritten_words

        # 印出單筆結果
        print(f"{item_id:<25} | {orig_words:<15} | {rewritten_words:<16}")

    # 印出統計摘要
    print("-" * 62)
    if count > 0:
        print(f"Total entries: {count}")
        print(f"Average Original Words: {total_original_words / count:.2f} words")
        print(f"Average Rewritten Words: {total_rewritten_words / count:.2f} words")

except FileNotFoundError:
    print(f"錯誤：找不到檔案，請確認路徑 {file_path} 是否正確。")
except json.JSONDecodeError:
    print("錯誤：檔案內容不是有效的 JSON 格式。")
except Exception as e:
    print(f"發生未預期的錯誤：{e}")