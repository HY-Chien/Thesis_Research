import json
import nltk
from faker import Faker
from tqdm import tqdm
import re
import os

# ================= 設定區 =================
INPUT_FILE = '/home/S113065528/Adaptive-RAG/classifier/data/musique_hotpot_wiki2_nq_tqa_sqd/predict.json'
OUTPUT_FILE = './attack_dataset/predict_2wikimultihopqa_perturbed_100.json'     # 檔名稍微改一下，區分是測試用的
LOG_FILE = './attack_dataset/2wikimultihopqa_subject_attack_log_100.json'
TARGET_DATASET = '2wikimultihopqa'
MAX_SAMPLES = 100  # ★★★ 設定只改 100 筆 ★★★
# =========================================

fake = Faker()

# 1. 下載資源
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'words']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            try:
                nltk.data.find(f'taggers/{r}')
            except LookupError:
                nltk.download(r, quiet=True)

download_nltk_resources()

# 2. 安全字名單 (不替換的字)
SAFE_WORDS = {
    "who", "what", "where", "when", "how", "why", "which",
    "is", "was", "are", "were", "did", "does", "do", "has", "have", "had",
    "the", "a", "an", "this", "that", "these", "those",
    "in", "on", "at", "by", "for", "with", "from", "to", "of",
    "president", "king", "queen", "prime", "minister", "governor",
    "song", "album", "movie", "book", "novel", "series", "character",
    "year", "date", "time", "day", "month", "century",
    "state", "country", "city", "river", "mountain", "ocean",
    "first", "last", "best", "most", "name", "type", "kind",
    "man", "woman", "person", "people", "child", "children",
    "war", "battle", "treaty", "act", "law", "system",
    "episode", "season", "show", "band", "group", "singer", "actor"
}

def get_fake_name():
    return fake.first_name() if fake.boolean() else fake.city()

def perturb_subject_only(text):
    if not text:
        return text, 0, []

    try:
        tokens = nltk.word_tokenize(text.lower())
        tags = nltk.pos_tag(tokens)
    except:
        return text, 0, []

    candidates = []
    current_chunk = []
    
    for word, tag in tags:
        if tag.startswith('NN') and word.isalnum():
            if word not in SAFE_WORDS:
                current_chunk.append(word)
            else:
                if current_chunk:
                    candidates.append(" ".join(current_chunk))
                    current_chunk = []
        else:
            if current_chunk:
                candidates.append(" ".join(current_chunk))
                current_chunk = []
    
    if current_chunk:
        candidates.append(" ".join(current_chunk))

    candidates = sorted(candidates, key=len, reverse=True)
    
    final_candidates = []
    for c in candidates:
        is_substring = False
        for existing in final_candidates:
            if c in existing:
                is_substring = True
                break
        if not is_substring:
            final_candidates.append(c)

    new_text = text
    replaced_count = 0
    changes_made = []
    MAX_REPLACE = 1 
    
    for target in final_candidates[:MAX_REPLACE]:
        fake_entity = get_fake_name().lower()
        pattern = re.compile(r'\b' + re.escape(target) + r'\b', re.IGNORECASE)
        
        if pattern.search(new_text):
            new_text = pattern.sub(fake_entity, new_text, count=1)
            replaced_count += 1
            changes_made.append({
                "original_word": target,
                "replaced_with": fake_entity,
                "type": "SUBJECT_CANDIDATE"
            })

    return new_text, replaced_count, changes_made

def main():
    print(f"Reading {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤：找不到檔案 {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} samples.")
    print(f"Target: '{TARGET_DATASET}', Limit: {MAX_SAMPLES} samples")
    
    perturbed_data = []
    log_entries = []
    
    modified_count = 0  # 目前成功修改的數量
    
    for entry in tqdm(data, desc="Perturbing"):
        dataset_name = entry.get('dataset_name', '')
        
        # 條件：是 NQ 且 還沒改滿 100 筆
        if dataset_name == TARGET_DATASET and modified_count < MAX_SAMPLES:
            original_q = entry.get('question', '')
            
            # 嘗試替換
            new_q, count, changes = perturb_subject_only(original_q)
            
            new_entry = entry.copy()
            new_entry['question'] = new_q
            
            # 只有當真的有發生替換時，計數器才 +1
            if count > 0:
                modified_count += 1
                log_entries.append({
                    "dataset_name": dataset_name,
                    "id": entry.get('id'),
                    "original_question": original_q,
                    "perturbed_question": new_q,
                    "changes": changes
                })
            
            perturbed_data.append(new_entry)
        else:
            # 不是 NQ，或是已經改滿 100 筆了 -> 保持原樣
            perturbed_data.append(entry)
            
    print(f"\nProcessing Complete.")
    print(f"Target Limit: {MAX_SAMPLES}")
    print(f"Actually Modified: {modified_count}")
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(perturbed_data, f, indent=4)
    print(f"Saved model input to: {OUTPUT_FILE}")
    
    with open(LOG_FILE, 'w') as f:
        json.dump(log_entries, f, indent=4)
    print(f"Saved logs to: {LOG_FILE}")

if __name__ == "__main__":
    main()