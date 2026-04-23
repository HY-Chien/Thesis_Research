import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import re
import json
import logging
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

INPUT_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset/predict_nq_original.json"
OUTPUT_DIR = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/dataset"
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "predict_nq_rewritten_qwen_expansion.json")
LOG_FILE = os.path.join(OUTPUT_DIR, "rewrite_qwen_nq_expansion.log")

HF_TOKEN = os.environ.get("HF_TOKEN", "hf_dummy_token_for_local_use")
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logging.info("⏳ 正在下載/載入本地模型，這可能需要一些時間...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    token=HF_TOKEN,
    local_files_only=True  # 🌟 新增這行
)

# 在 Model 加入 local_files_only=True
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda",           
    torch_dtype=torch.bfloat16,  
    token=HF_TOKEN,
    local_files_only=True  # 🌟 新增這行
)

def rewrite_question(original_question):
    messages = [
        {
            "role": "system", 
            "content": (
                "You are an expert linguist. Your task is to rewrite the question to be significantly LONGER and MORE VERBOSE, "
                "but you must STRICTLY rely ONLY on the words provided in the original text.\n\n"
                "CRITICAL RULES:\n"
                "1. NO EXTERNAL KNOWLEDGE: Do not add any background information or categories. (e.g., do not add 'automobile racing event' if it only says 'brickyard 400').\n"
                "2. NO PUNCTUATION AT ALL: The output must NOT contain any punctuation marks (no question marks, no commas, no quotes, no periods).\n"
                "3. STRICTLY LOWERCASE: The entire output must be in lowercase letters.\n"
                "4. Output ONLY the rewritten string and absolutely nothing else.\n\n"
                "Here are some examples of the expected output:\n\n"
                "Example 1:\n"
                "Original Question: who won the first brickyard 400 in 1994\n"
                "Rewritten Question: could you please provide information regarding the individual who emerged victorious in the inaugural iteration of the brickyard 400 which took place in the year 1994\n\n"
                "Example 2:\n"
                "Original Question: who wrote the world is waiting for the sunrise\n"
                "Rewritten Question: could you please provide information regarding the authorship of the work entitled the world is waiting for the sunrise\n\n"
                "Example 3:\n"
                "Original Question: who did the virginia plan give more power to\n"
                "Rewritten Question: could you please provide information regarding to which segment or entity the virginia plan allocated a greater degree of authority or influence according to its original proposals"
            )
        },
        {"role": "user", "content": f"Original Question: {original_question}\nRewritten Question:"}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
        attention_mask = model_inputs.attention_mask
        
        generated_ids = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,  # 保持較大的長度上限
            temperature=0.5,     # 🌟 降低溫度至 0.3，讓模型的輸出更穩定、更貼近範例
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        rewritten = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # 🛡️ 雙重保險 (Post-processing)：強制轉小寫，並用正規表達式移除所有標點符號
        rewritten = rewritten.lower()
        rewritten = re.sub(r'[^\w\s]', '', rewritten)
        
        return rewritten.strip()
        
    except Exception as e:
        logging.error(f"❌ 本地模型生成發生錯誤: {e}")
        return original_question

def main():
    logging.info(f"🚀 開始執行 Query 改寫實驗 (使用本地模型 {MODEL_ID})...")
    
    if not os.path.exists(INPUT_FILE):
        logging.error(f"找不到輸入檔案: {INPUT_FILE}")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    logging.info(f"📁 成功載入 {len(data)} 筆資料。\n" + "="*50)
    
    rewritten_data = []
    
    for i, item in enumerate(tqdm(data, desc="Rewriting Queries")):
        original_q = item.get("question", "")
        if not original_q:
            rewritten_data.append(item)
            continue
            
        rewritten_q = rewrite_question(original_q)
        
        logging.info(f"[{item['id']}]")
        logging.info(f"   原句: {original_q}")
        logging.info(f"   改寫: {rewritten_q}")
        logging.info("-" * 40)
        
        new_item = item.copy()
        new_item["question"] = rewritten_q
        rewritten_data.append(new_item)
        
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(rewritten_data, f, ensure_ascii=False, indent=4)
        
    logging.info("="*50)
    logging.info(f"✅ 改寫完成！")
    
if __name__ == "__main__":
    main()