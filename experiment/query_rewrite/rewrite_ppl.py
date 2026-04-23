import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= 設定區 =================
# 使用 Flan-T5-XL 作為基準，這是在邏輯推理研究中表現較穩健的模型 [cite: 216, 219]
BASE_MODEL_PATH = "google/flan-t5-xl" 

# 修改路徑至你指定的比較資料夾與檔名
ORIGINAL_DATA_FILE = "/home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl"

OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset"
REWRITTEN_DATA_FILE = os.path.join(OUTPUT_DIR, "predict_nq_rewritten_qwen.json")
COMPARISON_FILE = os.path.join(OUTPUT_DIR, "rewrite_comparison_ppl.json")

DEVICE = "cuda:1" if torch.cuda.is_available() >= 2 else ("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_ppl(text, model, tokenizer):
    """計算句子的 Perplexity (PPL). 低 PPL 代表模型認為該句子較自然、符合語言規律"""
    if not text or text.strip() == "":
        return float('inf')
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        # labels = input_ids 是 Seq2Seq 計算序列似然率的標準做法
        outputs = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = outputs.loss.item()
        return np.exp(loss)

def main():
    # 建立目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 載入原始資料與改寫後的資料
    orig_map = {}
    if not os.path.exists(ORIGINAL_DATA_FILE):
        print(f"❌ 找不到原始檔案: {ORIGINAL_DATA_FILE}")
        return

    with open(ORIGINAL_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            orig_map[item["question_id"]] = item["question_text"]

    if not os.path.exists(REWRITTEN_DATA_FILE):
        print(f"❌ 找不到改寫檔案: {REWRITTEN_DATA_FILE}")
        return

    with open(REWRITTEN_DATA_FILE, 'r', encoding='utf-8') as f:
        rewrite_list = json.load(f)

    # 2. 載入基準模型 (Flan-T5-XL) [cite: 219]
    print(f"🧠 載入模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.float16 # 使用半精度節省內存
    ).to(DEVICE)
    model.eval()

    # 3. 逐題計算比較
    ppl_comparison_results = []
    stats_orig = []
    stats_new = []

    print("📝 正在計算並對比原題與改寫題的 PPL...")
    for item in tqdm(rewrite_list):
        qid = item["id"]
        rewritten_q = item["question"]
        original_q = orig_map.get(qid)

        if original_q:
            p_orig = calculate_ppl(original_q, model, tokenizer)
            p_new = calculate_ppl(rewritten_q, model, tokenizer)

            # 建立對比格式
            comparison_entry = {
                "id": qid,
                "original_query": original_q,
                "original_ppl": round(p_orig, 4),
                "rewritten_query": rewritten_q,
                "rewritten_ppl": round(p_new, 4),
                "ppl_diff": round(p_orig - p_new, 4),
                # 🌟 關鍵修正：加上 bool() 確保可被 JSON 序列化
                "is_more_natural": bool(p_new < p_orig) 
            }
            ppl_comparison_results.append(comparison_entry)
            stats_orig.append(p_orig)
            stats_new.append(p_new)

    # 4. 儲存比較結果
    with open(COMPARISON_FILE, 'w', encoding='utf-8') as f:
        json.dump(ppl_comparison_results, f, ensure_ascii=False, indent=4)

    # 5. 輸出統計摘要
    print("\n" + "="*50)
    print("✅ PPL 比較對比表已生成！")
    print(f"檔案路徑: {COMPARISON_FILE}")
    print("-" * 50)
    if stats_orig and stats_new:
        print(f"原始問題 平均 PPL: {np.mean(stats_orig):.4f}")
        print(f"改寫問題 平均 PPL: {np.mean(stats_new):.4f}")
        print(f"平均改進幅度: {np.mean(stats_orig) - np.mean(stats_new):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()