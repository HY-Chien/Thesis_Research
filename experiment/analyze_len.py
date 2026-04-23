import json
import jsonlines
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
# 1. 設定路徑與參數
# ==========================================

# ★ 請確認這行路徑是您最新的預測結果
PRED_DIR = '../classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40/predict/'

# 原始資料的根目錄 (假設在上一層的 processed_data)
DATA_DIR = '../processed_data'

# 要分析的資料集列表
DATASETS = ['nq', 'trivia', 'squad', 'musique', 'hotpotqa', '2wikimultihopqa']

# ==========================================
# 2. 輔助函數
# ==========================================

def load_json(path):
    """讀取標準 JSON 檔案"""
    with open(path, 'r') as f:
        return json.load(f)

def load_questions_from_jsonl(dataset_name):
    """
    從原始 jsonl 檔案中讀取問題文本。
    路徑格式假設為: ../processed_data/{dataset_name}/test_subsampled.jsonl
    """
    path = os.path.join(DATA_DIR, dataset_name, 'test_subsampled.jsonl')
    qid_to_text = {}
    
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path}")
        return {}

    with jsonlines.open(path) as reader:
        for obj in reader:
            # 嘗試抓取各種可能的 ID 欄位名稱
            qid = obj.get('question_id') or obj.get('id') or obj.get('_id')
            # 嘗試抓取各種可能的 Question 欄位名稱
            q_text = obj.get('question') or obj.get('question_text')
            
            if qid and q_text:
                qid_to_text[str(qid)] = q_text
    return qid_to_text

# ==========================================
# 3. 主程式邏輯
# ==========================================

def main():
    # --- A. 讀取預測結果 ---
    pred_file = os.path.join(PRED_DIR, 'dict_id_pred_results.json')
    if not os.path.exists(pred_file):
        print(f"[Error] 找不到預測檔案，請檢查路徑：\n{pred_file}")
        return

    print(f"Loading predictions from: {pred_file}")
    preds = load_json(pred_file)
    
    # --- B. 讀取原始問題文本 ---
    print("Loading original questions text from jsonl files...")
    all_questions = {}
    for ds in DATASETS:
        q_map = load_questions_from_jsonl(ds)
        all_questions.update(q_map)
        print(f"  - {ds}: loaded {len(q_map)} questions")

    # --- C. 整合資料 (Merge) ---
    data_list = []
    missing_count = 0
    
    for qid, info in preds.items():
        label = info['prediction']  # A, B, or C
        ds_name = info['dataset_name']
        
        # 透過 ID 找回原始問題文本
        q_text = all_questions.get(qid)
        
        if q_text:
            # 計算長度 (以空白分割計算單字數)
            length = len(q_text.split())
            data_list.append({
                'qid': qid,
                'dataset': ds_name,
                'label': label,
                'length': length,
                'text': q_text
            })
        else:
            missing_count += 1

    print(f"Total analyzed: {len(data_list)} samples (Missing text mapping: {missing_count})")
    
    if len(data_list) == 0:
        print("No data found to analyze. Exiting.")
        return

    # 轉成 DataFrame 方便操作
    df = pd.DataFrame(data_list)
    
    # --- D. 視覺化 (Visualization) ---
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 6))

    # 定義 Label 順序
    order = ['A', 'B', 'C'] 

    # 子圖 1: 整體長度分佈 (Boxplot)
    plt.subplot(1, 2, 1)
    sns.boxplot(x='label', y='length', data=df, order=order, palette="Set2")
    plt.title('Question Length Distribution by Predicted Complexity')
    plt.xlabel('Predicted Label (A:No, B:Single, C:Iterative)')
    plt.ylabel('Question Length (Words)')

    # 子圖 2: 每個資料集的平均長度 (Barplot)
    plt.subplot(1, 2, 2)
    sns.barplot(x='dataset', y='length', hue='label', data=df, hue_order=order, palette="Set2", errorbar=('ci', 95))
    plt.title('Average Length by Dataset & Label')
    plt.xticks(rotation=45)
    plt.ylabel('Avg Question Length')
    plt.legend(title='Complexity')

    plt.tight_layout()
    output_img = 'analysis_length_correlation.png'
    plt.savefig(output_img)
    print(f"\n[Plot] Analysis plot saved to: {os.path.abspath(output_img)}")
    
    # --- E. 統計數據輸出 (Statistics) ---
    
    # 設定 pandas 顯示選項，避免表格被截斷
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print("\n" + "="*60)
    print("=== 1. 詳細統計數據 (Group by Dataset & Label) ===")
    print("顯示每個資料集中，各個 Label 的平均長度(mean)、標準差(std)與樣本數(count)")
    # Group by dataset AND label
    detailed_stats = df.groupby(['dataset', 'label'])['length'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(detailed_stats.round(2))

    print("\n" + "="*60)
    print("=== 2. 平均長度比較表 (Pivot Table: Mean Length) ===")
    print("分析：同一個資料集中，Label C 是否比 Label A/B 長？")
    pivot_mean = df.pivot_table(index='dataset', columns='label', values='length', aggfunc='mean')
    # 重新排序欄位確保是 A, B, C (使用 reindex 避免報錯如果某個 label 不存在)
    pivot_mean = pivot_mean.reindex(columns=['A', 'B', 'C'])
    print(pivot_mean.round(2))

    print("\n" + "="*60)
    print("=== 3. 樣本分布矩陣 (Dataset Distribution) ===")
    print("這是您論文中需要的 Label 分布表")
    pivot_count = df.pivot_table(index='dataset', columns='label', values='length', aggfunc='count', fill_value=0)
    pivot_count = pivot_count.reindex(columns=['A', 'B', 'C'])
    
    # 加上 Total 欄位
    pivot_count['Total'] = pivot_count.sum(axis=1)
    print(pivot_count)

    print("\n" + "="*60)

if __name__ == "__main__":
    main()