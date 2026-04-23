import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm  #用來顯示進度條

# ================= 設定區 =================
# 1. 模型路徑
MODEL_PATH = "/home/S113065528/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40"


# 2. Log 檔案路徑 (請確認檔名是否正確，或是上一支程式產生的檔名)
LOG_FILE = "./attack_dataset/nq_subject_attack_log_100.json"

# 3. 圖片輸出目錄
OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/entity_attack/attention_heatmap_flan_t5_xl"

# ==========================================

def load_all_examples():
    """
    讀取 Log 檔中所有的資料
    """
    if not os.path.exists(LOG_FILE):
        print(f"Error: 找不到 Log 檔 {LOG_FILE}，請確認檔案位置。")
        return []

    with open(LOG_FILE, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("Error: Log 檔是空的")
        return []

    print(f"成功讀取 Log 檔，共有 {len(data)} 筆資料。")
    return data

def visualize_cross_attention(model, tokenizer, question):
    inputs = tokenizer(question, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device) # 確保 input 在同一個 device
    
    decoder_start_token = model.config.decoder_start_token_id
    decoder_input_ids = torch.tensor([[decoder_start_token]]).to(model.device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids, 
            decoder_input_ids=decoder_input_ids,
            output_attentions=True
        )

    # 抓取最後一層的 Cross Attention
    last_layer_attn = outputs.cross_attentions[-1]
    
    # [Batch, Heads, Seq_len_out, Seq_len_in]
    # mean(dim=1) -> 平均所有 Heads
    # squeeze(0) -> 移除 Batch 維度
    # mean(dim=0) -> 平均所有 Decoder Steps (看整體輸入的重要性)
    attn_avg = last_layer_attn.mean(dim=1).squeeze(0).mean(dim=0)
    
    # 轉為 numpy
    attn_weights = attn_avg.cpu().numpy().flatten()
    
    # 處理 Tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    # T5 的 SentencePiece 會有   符號，替換掉方便顯示
    clean_tokens = [t.replace(' ', '') for t in tokens]
    
    return clean_tokens, attn_weights

def plot_heatmap(tokens_1, weights_1, tokens_2, weights_2, q1_text, q2_text, save_path):
    """
    繪製並儲存熱力圖
    """
    # 設定後端，避免在伺服器上因為沒有 X11 而報錯
    plt.switch_backend('Agg') 
    
    plt.figure(figsize=(22, 10))
    sns.set(font_scale=1.4)

    # 確保兩張圖共用同一個色階最大值，才能互相比較
    max_val = max(np.max(weights_1), np.max(weights_2))

    # 畫第一張圖 (原始問題)
    plt.subplot(2, 1, 1)
    sns.heatmap([weights_1], xticklabels=tokens_1, yticklabels=['Attn'], 
                cmap="Reds", cbar=True, annot=False, square=True,
                vmin=0, vmax=max_val) 
    plt.title(f"Original: {q1_text}", fontsize=16)
    plt.xticks(rotation=45, ha='right')

    # 畫第二張圖 (修改後問題)
    plt.subplot(2, 1, 2)
    sns.heatmap([weights_2], xticklabels=tokens_2, yticklabels=['Attn'], 
                cmap="Reds", cbar=True, annot=False, square=True,
                vmin=0, vmax=max_val)
    plt.title(f"Perturbed: {q2_text}", fontsize=16)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close() # ★重要：關閉圖表釋放記憶體

def main():
    # 1. 建立輸出資料夾
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"建立輸出目錄: {OUTPUT_DIR}")
    else:
        print(f"輸出目錄已存在: {OUTPUT_DIR}")

    # 2. 讀取所有資料
    data_list = load_all_examples()
    if not data_list:
        return

    # 3. 載入模型 (只載入一次)
    print(f"\nLoading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: 找不到模型路徑，請檢查路徑：\n{MODEL_PATH}")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    
    # 如果有 GPU 就用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()
    
    print("\n開始批次生成 Attention Heatmaps...")
    
    # 4. 迴圈處理每一筆資料
    for entry in tqdm(data_list, desc="Generating Images"):
        entry_id = entry.get('id')
        q1 = entry.get('original_question')
        q2 = entry.get('perturbed_question')
        
        if not q1 or not q2:
            continue
            
        # 計算 Attention
        tokens1, weights1 = visualize_cross_attention(model, tokenizer, q1)
        tokens2, weights2 = visualize_cross_attention(model, tokenizer, q2)
        
        # 移除最後的 </s> token (通常是最後一個，權重通常很高但沒意義)
        if tokens1 and tokens1[-1] == '</s>':
            tokens1 = tokens1[:-1]
            weights1 = weights1[:-1]
        if tokens2 and tokens2[-1] == '</s>':
            tokens2 = tokens2[:-1]
            weights2 = weights2[:-1]

        # 設定儲存路徑
        file_name = f"{entry_id}.png"
        save_path = os.path.join(OUTPUT_DIR, file_name)
        
        # 繪圖並儲存
        plot_heatmap(tokens1, weights1, tokens2, weights2, q1, q2, save_path)

    print(f"\n全部完成！圖片已儲存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()