import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= 設定區 =================
# 1. 基礎路徑設定 (全部改為絕對路徑，確保萬無一失)
BASE_ROOT = "/home/S113065528/Adaptive-RAG"
PREDICT_FILE = os.path.join(BASE_ROOT, "classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40/predict/dict_id_pred_results.json")
DATA_ROOT = os.path.join(BASE_ROOT, "processed_data")
OUTPUT_FILE = os.path.join(BASE_ROOT, "experiment/ppl/cognitive_data_with_triple_ppl.json")

MODEL_NAME = "google/flan-t5-xl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 策略預測檔案路徑 (抓取模型生成的答案) =================
model_name_str = "flan_t5_xl"
base_pred_dir = os.path.join(BASE_ROOT, "predictions/test")

dataName_to_multi_one_zero_file = {
    'nq': {
        'A': os.path.join(base_pred_dir, f'nor_qa_{model_name_str}_nq____prompt_set_1', 'prediction__nq_to_nq__test_subsampled.json'),
        'B': os.path.join(base_pred_dir, f'oner_qa_{model_name_str}_nq____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__nq_to_nq__test_subsampled.json'),
        'C': os.path.join(base_pred_dir, f'ircot_qa_{model_name_str}_nq____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__nq_to_nq__test_subsampled.json')
    },
    'hotpotqa': {
        'A': os.path.join(base_pred_dir, f'nor_qa_{model_name_str}_hotpotqa____prompt_set_1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled.json'),
        'B': os.path.join(base_pred_dir, f'oner_qa_{model_name_str}_hotpotqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled.json'),
        'C': os.path.join(base_pred_dir, f'ircot_qa_{model_name_str}_hotpotqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__hotpotqa_to_hotpotqa__test_subsampled.json')
    },
    '2wikimultihopqa': {
        'A': os.path.join(base_pred_dir, f'nor_qa_{model_name_str}_2wikimultihopqa____prompt_set_1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled.json'),
        'B': os.path.join(base_pred_dir, f'oner_qa_{model_name_str}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled.json'),
        'C': os.path.join(base_pred_dir, f'ircot_qa_{model_name_str}_2wikimultihopqa____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__2wikimultihopqa_to_2wikimultihopqa__test_subsampled.json')
    },
    'musique': {
        'A': os.path.join(base_pred_dir, f'nor_qa_{model_name_str}_musique____prompt_set_1', 'prediction__musique_to_musique__test_subsampled.json'),
        'B': os.path.join(base_pred_dir, f'oner_qa_{model_name_str}_musique____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__musique_to_musique__test_subsampled.json'),
        'C': os.path.join(base_pred_dir, f'ircot_qa_{model_name_str}_musique____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__musique_to_musique__test_subsampled.json')
    },
    'trivia': {
        'A': os.path.join(base_pred_dir, f'nor_qa_{model_name_str}_trivia____prompt_set_1', 'prediction__trivia_to_trivia__test_subsampled.json'),
        'B': os.path.join(base_pred_dir, f'oner_qa_{model_name_str}_trivia____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__trivia_to_trivia__test_subsampled.json'),
        'C': os.path.join(base_pred_dir, f'ircot_qa_{model_name_str}_trivia____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__trivia_to_trivia__test_subsampled.json')
    },
    'squad': {
        'A': os.path.join(base_pred_dir, f'nor_qa_{model_name_str}_squad____prompt_set_1', 'prediction__squad_to_squad__test_subsampled.json'),
        'B': os.path.join(base_pred_dir, f'oner_qa_{model_name_str}_squad____prompt_set_1___bm25_retrieval_count__15___distractor_count__1', 'prediction__squad_to_squad__test_subsampled.json'),
        'C': os.path.join(base_pred_dir, f'ircot_qa_{model_name_str}_squad____prompt_set_1___bm25_retrieval_count__6___distractor_count__1', 'prediction__squad_to_squad__test_subsampled.json')
    }
}
# ==========================================

def load_queries_and_answers(predict_data):
    """抓取原始的題目 (Query) 與 所有標準答案 (Answers)"""
    id_to_qa = {}
    dataset_names = set(info["dataset_name"] for info in predict_data.values())
    
    for d_name in dataset_names:
        file_path = os.path.join(DATA_ROOT, d_name, "test_subsampled.jsonl")
        if not os.path.exists(file_path): continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                qid = data["question_id"]
                if qid in predict_data:
                    answers = []
                    if "answers_objects" in data and len(data["answers_objects"]) > 0:
                        spans = data["answers_objects"][0].get("spans", [])
                        if spans: answers = spans
                    id_to_qa[qid] = {"query": data["question_text"], "answers": answers}
    return id_to_qa

def preload_generated_answers():
    """精準載入模型生成的答案 (基於提供的單純字典格式 {"qid": "answer"})"""
    print("📂 正在載入各策略的生成答案 (Generated Answers)...")
    gen_cache = {}
    missing_files = []
    loaded_answers_count = 0
    
    for d_name, labels_dict in dataName_to_multi_one_zero_file.items():
        gen_cache[d_name] = {}
        for lbl, path in labels_dict.items():
            gen_cache[d_name][lbl] = {}
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 直接迭代讀取 {"qid": "答案字串"} 的格式
                    for qid_key, answer_str in data.items():
                        gen_cache[d_name][lbl][str(qid_key)] = str(answer_str)
                        loaded_answers_count += 1
            else:
                missing_files.append(path)
                
    if missing_files:
        print(f"⚠️ 警告：找不到以下 {len(missing_files)} 個預測檔案 (請確認實驗是否已跑完對應策略):")
        for mf in missing_files[:3]: print(f"  - {mf}")
        if len(missing_files) > 3: print("  ...及其他檔案")
        
    print(f"✅ 成功將 {loaded_answers_count} 筆生成答案載入記憶體。")
    return gen_cache

def calculate_triple_ppl(query, answers_list, generated_answer, model, tokenizer):
    """計算三種 PPL：Query PPL, Ground Truth PPL, Generated Answer PPL"""
    query_inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    
    with torch.no_grad():
        # 1. Query PPL
        outputs_q = model(input_ids=query_inputs.input_ids, labels=query_inputs.input_ids)
        query_ppl = np.exp(outputs_q.loss.item())
        
        # 2. Ground Truth PPL (取所有可能答案中最低的)
        min_gt_ppl = float('inf')
        for ans in answers_list:
            gt_inputs = tokenizer(ans, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            outputs_gt = model(input_ids=query_inputs.input_ids, labels=gt_inputs.input_ids)
            ans_ppl = np.exp(outputs_gt.loss.item())
            if ans_ppl < min_gt_ppl:
                min_gt_ppl = ans_ppl
                
        # 3. Generated Answer PPL (如果真的有抓到生成的答案，且不是空白)
        if generated_answer and generated_answer.strip() != "":
            gen_inputs = tokenizer(generated_answer, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            outputs_gen = model(input_ids=query_inputs.input_ids, labels=gen_inputs.input_ids)
            gen_ppl = np.exp(outputs_gen.loss.item())
        else:
            gen_ppl = float('inf') 
            
    return query_ppl, min_gt_ppl, gen_ppl

def main():
    print("🚀 載入 Router 分類預測資料...")
    with open(PREDICT_FILE, 'r', encoding='utf-8') as f:
        predict_data = json.load(f)
        
    id_to_qa = load_queries_and_answers(predict_data)
    
    # 呼叫強健版預載入函數
    generated_cache = preload_generated_answers()
    
    print(f"🧠 正在載入模型 {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(DEVICE)
    model.eval()

    enhanced_dataset = []
    
    # 統計字典
    stats = {
        "A": {"query_ppl": [], "gt_ppl": [], "gen_ppl": []},
        "B": {"query_ppl": [], "gt_ppl": [], "gen_ppl": []},
        "C": {"query_ppl": [], "gt_ppl": [], "gen_ppl": []}
    }
    
    print("🔍 開始計算三重 PPL (Query / Ground Truth / Generated)...")
    for qid, info in tqdm(list(predict_data.items())):
        label = info["prediction"]
        d_name = info["dataset_name"]
        qa_data = id_to_qa.get(qid)
        
        if qa_data and qa_data.get("answers") and label in stats:
            query = qa_data["query"]
            answers_list = qa_data["answers"]
            
            # 從快取中抓出這題對應策略所生成的最終答案 (強制轉為字串以防萬一)
            generated_answer = generated_cache.get(d_name, {}).get(label, {}).get(str(qid), "")
            
            # 算出三種 PPL
            q_ppl, gt_ppl, gen_ppl = calculate_triple_ppl(query, answers_list, generated_answer, model, tokenizer)
            
            if gt_ppl == float('inf'): continue
            
            stats[label]["query_ppl"].append(q_ppl)
            stats[label]["gt_ppl"].append(gt_ppl)
            if gen_ppl != float('inf'): stats[label]["gen_ppl"].append(gen_ppl)
            
            enhanced_dataset.append({
                "question_id": qid,
                "dataset_name": d_name,
                "predicted_strategy": label,
                "query": query,
                "generated_answer": generated_answer,
                "query_ppl": q_ppl,
                "gt_ppl": gt_ppl,
                "gen_ppl": gen_ppl
            })

    # ================= 儲存資料集 =================
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(enhanced_dataset, f, ensure_ascii=False, indent=4)
    print(f"\n✅ 成功儲存增強資料集至: {OUTPUT_FILE}")

    # ================= 印出統計結果 =================
    print("\n" + "="*70)
    print("🌟 實驗結果：各策略 (Label) 的三重 PPL 平均統計 🌟")
    print("="*70)
    
    for label in ["A", "B", "C"]:
        q_ppls = stats[label]["query_ppl"]
        gt_ppls = stats[label]["gt_ppl"]
        gen_ppls = stats[label]["gen_ppl"]
        
        if q_ppls:
            print(f"🔹 策略標籤 {label} (共 {len(q_ppls)} 題):")
            print(f"   [Query PPL]           平均: {np.mean(q_ppls):.4f} | 中位數: {np.median(q_ppls):.4f}")
            print(f"   [Ground Truth PPL]    平均: {np.mean(gt_ppls):.4f} | 中位數: {np.median(gt_ppls):.4f}")
            
            # 避免 Numpy Runtime Warning 的安全印出法
            if gen_ppls:
                print(f"   [Generated Ans PPL]   平均: {np.mean(gen_ppls):.4f} | 中位數: {np.median(gen_ppls):.4f}")
            else:
                print("   [Generated Ans PPL]   (無有效生成答案可供計算)")
            print("-" * 70)
        else:
            print(f"🔹 策略標籤 {label} : 無資料")

if __name__ == "__main__":
    main()