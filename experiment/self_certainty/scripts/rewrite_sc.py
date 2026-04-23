"""
rewrite_sc.py — Self-Certainty 版的 rewrite_ppl.py
====================================================
實驗目標：
  改寫題目（句子變長、語意不變）後，Flan-T5-XL 回答這題時的「信心」
  （Self-Certainty）有沒有跟著改變？

為什麼不用 PPL？
  根據 arxiv 2502.18581，PPL 無法預測模型能否正確回答問題，
  且會被句子長度影響。Self-Certainty 改測「整個詞彙表分佈的尖銳程度」，
  與模型正確性的相關性更強。

Self-Certainty 公式（用在 Flan-T5-XL decoder 輸出）：
  對 decoder 每個 timestep t：
    logprob_sum_t = Σ_j  log P(token_j | encoder_output, prev_tokens)   ← 整個 vocab 加總
    SC_t = -1/V * logprob_sum_t - log(V)
  最終 SC = mean(SC_t)

直覺：
  - 模型很確定答案 → 機率集中在少數 token → 分佈尖 → SC 高
  - 模型在亂猜     → 機率分散在整個 vocab → 分佈平 → SC 低
  - 均勻分佈時 SC = 0（理論最低值）

輸出：
  每筆資料有 original_sc / rewritten_sc / sc_diff
  sc_diff > 0 代表改寫後模型更確定；sc_diff < 0 代表更不確定

用法：
  python rewrite_sc.py
"""

import os
import json
import math
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ================= 設定區 =================
BASE_MODEL_PATH = "google/flan-t5-xl"

ORIGINAL_FILE  = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/predict_nq_original.json"
REWRITTEN_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/predict_nq_rewritten_expansion_v2.json"

# 分類器的預測結果（A/B/C），用來和 SC 對照
# 格式預期為 list of {"id": ..., "predicted_label": "A"/"B"/"C"}
# 若無此檔案，程式仍可執行，只是輸出不含 classifier_label
CLASSIFIER_PRED_FILE = "/home/S113065528/Adaptive-RAG/experiment/query_rewrite/predict_nq_original.json"

OUTPUT_DIR  = "/home/S113065528/Adaptive-RAG/experiment/self_certainty"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rewrite_sc_comparison.json")
LOG_FILE    = os.path.join(OUTPUT_DIR, "rewrite_sc.log")

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 50  # 生成答案的最大長度
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)


@torch.no_grad()
def compute_sc(question: str, model, tokenizer) -> float:
    """
    用 Flan-T5-XL 回答 question，計算回答過程的 Self-Certainty。

    Flan-T5-XL 是 encoder-decoder 架構：
      - encoder 讀入問題
      - decoder 逐步生成答案
    SC 是量 decoder 每個生成 timestep 的分佈尖銳程度。

    Args:
        question: 要問的題目（原句或改寫句）
        model: Flan-T5-XL
        tokenizer: 對應的 tokenizer

    Returns:
        float: SC 分數（越高代表模型越確定答案）
    """
    if not question or question.strip() == "":
        return float("nan")

    V = model.config.vocab_size

    # Tokenize 問題作為 encoder 輸入
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(DEVICE)

    # 生成答案，同時取出每個 timestep 的 logits（output_scores=True）
    # output_scores 回傳的是 log_softmax 之前的 raw logits
    generate_output = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,          # greedy，結果穩定可重現
        output_scores=True,       # 取出每個 timestep 的 logit 向量
        return_dict_in_generate=True,
    )

    # generate_output.scores: tuple，長度 = 實際生成的 token 數
    # 每個元素 shape: (1, V) — 該 timestep 對整個 vocab 的 raw logit
    if not generate_output.scores:
        return float("nan")

    sc_per_token = []
    for step_logits in generate_output.scores:
        # step_logits shape: (1, V)
        log_probs   = F.log_softmax(step_logits.float(), dim=-1)  # (1, V)
        logprob_sum = log_probs.sum(dim=-1).item()                # scalar：整個 vocab 加總

        # Self-Certainty 公式
        sc_t = -1.0 / V * logprob_sum - math.log(V)
        sc_per_token.append(sc_t)

    return float(np.mean(sc_per_token))


def main():
    logging.info(f"🧠 載入模型: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    logging.info("✅ 模型載入完成\n" + "=" * 60)

    # 讀入原始題目
    with open(ORIGINAL_FILE, encoding="utf-8") as f:
        originals = json.load(f)
    orig_map = {item["id"]: item["question"] for item in originals}

    # 讀入改寫題目
    with open(REWRITTEN_FILE, encoding="utf-8") as f:
        rewrittens = json.load(f)

    # 讀入分類器預測（選填）
    clf_map = {}
    if os.path.exists(CLASSIFIER_PRED_FILE) and CLASSIFIER_PRED_FILE != ORIGINAL_FILE:
        with open(CLASSIFIER_PRED_FILE, encoding="utf-8") as f:
            clf_data = json.load(f)
        clf_map = {item["id"]: item.get("predicted_label", "") for item in clf_data}

    results      = []
    orig_sc_list = []
    rew_sc_list  = []

    for item in tqdm(rewrittens, desc="Computing Self-Certainty"):
        qid        = item["id"]
        original_q = orig_map.get(qid, "")
        rewritten_q = item.get("question", "")

        if not original_q or not rewritten_q:
            continue

        sc_orig = compute_sc(original_q,  model, tokenizer)
        sc_rew  = compute_sc(rewritten_q, model, tokenizer)

        orig_sc_list.append(sc_orig)
        rew_sc_list.append(sc_rew)

        entry = {
            "id":             qid,
            "original":       original_q,
            "rewritten":      rewritten_q,
            "original_sc":    round(sc_orig, 6),
            "rewritten_sc":   round(sc_rew,  6),
            # sc_diff > 0：改寫後模型更確定；< 0：更不確定
            "sc_diff":        round(sc_rew - sc_orig, 6),
        }

        if clf_map:
            entry["classifier_label"] = clf_map.get(qid, "")

        results.append(entry)

        logging.info(
            f"[{qid}]  orig_SC={sc_orig:.4f}  rew_SC={sc_rew:.4f}  "
            f"diff={sc_rew - sc_orig:+.4f}"
        )

    # 儲存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 統計摘要
    logging.info("\n" + "=" * 60)
    logging.info(f"✅ 完成！共 {len(results)} 筆")
    logging.info(f"   原始題目   平均 SC : {np.mean(orig_sc_list):.4f}")
    logging.info(f"   改寫題目   平均 SC : {np.mean(rew_sc_list):.4f}")
    logging.info(f"   平均 SC 差 (改寫-原) : {np.mean(rew_sc_list) - np.mean(orig_sc_list):+.4f}")

    # 按 sc_diff 正負分組統計
    pos = [r for r in results if r["sc_diff"] > 0]
    neg = [r for r in results if r["sc_diff"] < 0]
    logging.info(f"   改寫後更確定 (sc_diff > 0): {len(pos)} 筆")
    logging.info(f"   改寫後更不確定 (sc_diff < 0): {len(neg)} 筆")
    logging.info(f"   結果已存至 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
