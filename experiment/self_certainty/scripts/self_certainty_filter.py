"""
Self-Certainty Filter for Rewritten Queries
============================================
根據 "Scalable Best-of-N Selection via Self-Certainty" (arxiv 2502.18581) 的方法，
計算每筆改寫結果的 Self-Certainty 分數，藉此偵測模型是否在改寫時「瞎掰」。

核心概念：
  - PPL 只看被選中 token 的機率，容易被句子長度影響。
  - Self-Certainty 對每個 token 位置把整個 vocab 的 log prob 加總，
    測量模型輸出分佈的「尖銳程度」（certainty）。
  - 公式：SC = mean_over_tokens( -1/V * Σ_j log P(j|context) - log(V) )
  - 均勻分佈 → SC = 0；模型越確定 → SC 越高
  - SC 低 → 模型不確定 → 改寫可能含幻覺

用法：
  python self_certainty_filter.py [--threshold 0.05]
"""

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import json
import math
import logging
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 設定區 =================
ORIGINAL_FILE  = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/predict_nq_original.json"
REWRITTEN_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/predict_nq_rewritten_expansion_v2.json"
OUTPUT_DIR     = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "self_certainty_scores.json")
LOG_FILE       = os.path.join(OUTPUT_DIR, "self_certainty_filter.log")

HF_TOKEN  = os.environ.get("HF_TOKEN", "hf_..")
MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 4   # 一次處理幾筆（依 GPU 記憶體調整）
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


# ---------- 與改寫腳本相同的 prompt 格式 ----------
SYSTEM_CONTENT = (
    "You are a question paraphraser. Your ONLY job is to make a question longer by adding filler or rephrasing words — "
    "you must NEVER add any new facts, dates, names, locations, or topics that are not already in the original question. "
    "RULES: "
    "1. Keep ALL named entities exactly as they appear in the original — do not correct, expand, or replace them. "
    "2. The rewritten question must ask the EXACT same thing as the original — one question, same answer. "
    "3. Do NOT add a second question or sub-question. One question only. "
    "4. The output must be at least 15 words long. "
    "5. Use only plain lowercase English. No quotation marks, no special symbols. "
    "6. Output ONLY the rewritten question. No explanation, no prefix, no extra text. "
    "EXAMPLES: "
    "Input: who wrote the song bohemian rhapsody | Output: who is the person that wrote the song bohemian rhapsody "
    "Input: when did the movie titanic come out | Output: in what year did the movie titanic first come out "
    "Input: who sang it's just a matter of time | Output: which singer was it that sang the song its just a matter of time "
    "Input: where was it came from outer space filmed | Output: in what location was the movie it came from outer space filmed "
    "Input: who came up with the process of natural selection | Output: who was the person that originally came up with the process of natural selection "
)


def build_prompt_text(tokenizer, original_question: str) -> str:
    """與改寫時相同的 prompt 格式，作為 Self-Certainty 的 context。"""
    messages = [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user",   "content": f"Input: {original_question} | Output:"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def compute_self_certainty(
    model,
    tokenizer,
    prompt_text: str,
    rewritten_text: str,
) -> float:
    """
    計算單筆 (prompt, rewritten) 的 Self-Certainty 分數。

    步驟：
      1. 把 prompt + rewritten 合在一起 tokenize
      2. forward pass 取得 logits
      3. 只保留 rewritten 部分對應的 logits
      4. log_softmax 後對 vocab 維度加總 → logprob_sum (shape: seq_len)
      5. SC = mean( -1/V * logprob_sum - log(V) )

    Returns:
        float: SC 分數（越高代表模型越確定，改寫越可靠）
    """
    V = model.config.vocab_size

    # Tokenize prompt（不加 special tokens 避免重複）
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_len = prompt_ids.shape[1]

    # Tokenize rewritten text
    rewrite_ids = tokenizer(rewritten_text, return_tensors="pt", add_special_tokens=False).input_ids

    # 合併
    full_ids = torch.cat([prompt_ids, rewrite_ids], dim=1).to(model.device)

    # Forward pass
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(full_ids).logits  # (1, total_len, V)

    # 取出 rewritten 部分的 logits（shifted by 1 for next-token prediction）
    # logits[:, prompt_len-1 : prompt_len+rewrite_len-1, :] 對應 rewrite tokens
    rewrite_len = rewrite_ids.shape[1]
    rewrite_logits = logits[:, prompt_len - 1 : prompt_len + rewrite_len - 1, :]  # (1, rewrite_len, V)

    # log_softmax over vocab
    log_probs = F.log_softmax(rewrite_logits.float(), dim=-1)  # (1, rewrite_len, V)

    # 對 vocab 維度加總 → logprob_sum per token position
    logprob_sum = log_probs.sum(dim=-1).squeeze(0)  # (rewrite_len,)

    # Self-Certainty formula
    V_tensor = torch.tensor(V, dtype=torch.float32, device=logprob_sum.device)
    sc_per_token = -1.0 / V_tensor * logprob_sum - torch.log(V_tensor)
    sc = sc_per_token.mean().item()

    return sc


def main(threshold: float):
    logging.info(f"⏳ 載入模型 {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        token=HF_TOKEN,
    )
    model.eval()

    logging.info("✅ 模型載入完成\n" + "=" * 60)

    # 讀入資料
    with open(ORIGINAL_FILE,  encoding="utf-8") as f:
        originals = json.load(f)
    with open(REWRITTEN_FILE, encoding="utf-8") as f:
        rewrittens = json.load(f)

    orig_map = {item["id"]: item["question"] for item in originals}
    rew_map  = {item["id"]: item for item in rewrittens}

    results   = []
    flagged   = []
    sc_values = []

    for item in tqdm(rewrittens, desc="Computing Self-Certainty"):
        qid          = item["id"]
        original_q   = orig_map.get(qid, "")
        rewritten_q  = item.get("question", "")

        if not original_q or not rewritten_q:
            continue

        prompt_text = build_prompt_text(tokenizer, original_q)
        sc = compute_self_certainty(model, tokenizer, prompt_text, rewritten_q)
        sc_values.append(sc)

        entry = {
            "id":           qid,
            "original":     original_q,
            "rewritten":    rewritten_q,
            "self_certainty": round(sc, 6),
            "flagged":      sc < threshold,
        }
        results.append(entry)

        if sc < threshold:
            flagged.append(entry)
            logging.info(f"🚩 FLAGGED  [{qid}]  SC={sc:.4f}")
            logging.info(f"   原句: {original_q}")
            logging.info(f"   改寫: {rewritten_q}")
            logging.info("-" * 50)

    # 儲存完整結果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 統計
    avg_sc = sum(sc_values) / len(sc_values) if sc_values else 0
    logging.info("\n" + "=" * 60)
    logging.info(f"✅ 完成！共 {len(results)} 筆")
    logging.info(f"   平均 Self-Certainty : {avg_sc:.4f}")
    logging.info(f"   Threshold           : {threshold}")
    logging.info(f"   被標記為可疑 (SC<{threshold}): {len(flagged)} 筆")
    logging.info(f"   結果已存至 {OUTPUT_FILE}")

    # 另存 flagged list
    flagged_file = OUTPUT_FILE.replace(".json", "_flagged.json")
    with open(flagged_file, "w", encoding="utf-8") as f:
        json.dump(flagged, f, ensure_ascii=False, indent=2)
    logging.info(f"   可疑清單已存至 {flagged_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="SC 低於此值視為可疑改寫（預設 0.05，建議先跑完再看分佈決定）"
    )
    args = parser.parse_args()
    main(args.threshold)
