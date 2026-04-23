import json
import jsonlines
import os

# 設定您的檔案路徑
SOURCE_FILE = '../../processed_data/nq/test_subsampled.jsonl'
OUTPUT_FILE = 'first_item_inspect.json'

def inspect_first_item():
    print(f"Reading from: {os.path.abspath(SOURCE_FILE)}")
    
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: File not found at {SOURCE_FILE}")
        return

    first_item = None
    
    # 讀取第一筆資料
    with jsonlines.open(SOURCE_FILE) as reader:
        try:
            first_item = reader.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

    if first_item:
        # 印出所有欄位名稱 (Keys)
        print("\n=== Field Names Found ===")
        for key in first_item.keys():
            print(f"- {key}")
            
        # 特別檢查我們關心的問題欄位
        q_text = first_item.get('question') or first_item.get('question_text')
        print(f"\n[Check] Question Content: {q_text}")
        
        # 存成漂亮的 JSON 檔
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(first_item, f, indent=4, ensure_ascii=False)
            
        print(f"\nSaved first item to: {OUTPUT_FILE}")
        print("You can open this file to see the full structure.")
    else:
        print("File is empty.")

if __name__ == "__main__":
    inspect_first_item()