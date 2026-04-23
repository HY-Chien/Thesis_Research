#!/usr/bin/env bash

# ================= 設定區 =================
SYSTEM="nor_qa" 
MODEL="flan-t5-xl"
DATASET="nq"
LLM_PORT_NUM="8010"

# 💡 說明：因為設定了 DATASET="nq"，底層程式會自動去讀取原始檔案：
# /home/S113065528/Adaptive-RAG/processed_data/nq/test_subsampled.jsonl

# 1. 預測結果路徑 (runner.py 預設產生的資料夾)
PRED_DIR="/home/S113065528/Adaptive-RAG/predictions/test/nor_qa_flan_t5_xl_nq____prompt_set_1"
PRED_BACKUP="/home/S113065528/Adaptive-RAG/predictions/test/nor_qa_flan_t5_xl_nq____prompt_set_1_backup"

# 2. 你的專屬實驗輸出目標資料夾
EXP_OUTPUT_DIR="/home/S113065528/Adaptive-RAG/experiment/nq_all_nor_qa/results"
# ==========================================

# 強制切換到專案根目錄
cd /home/S113065528/Adaptive-RAG

echo "=================================================="
echo "🚀 開始執行基準實驗：使用原始 NQ 測試集進行 '裸考 (nor_qa)'"
echo "=================================================="

# 1. 備份舊的預測結果 (避免 runner.py 偵測到有舊資料就偷懶跳過預測)
if [ -d "$PRED_DIR" ]; then
    echo "📦 發現舊的預測結果，正在進行備份..."
    mv "$PRED_DIR" "$PRED_BACKUP"
fi

# 2. 呼叫原本的程式執行 nor_qa (Zero-shot) 預測與評分
echo "🧠 正在呼叫 LLM 進行預測與評分 (這會花一點時間)..."
bash run_retrieval_test.sh $SYSTEM $MODEL $DATASET $LLM_PORT_NUM

# 3. 將新鮮出爐的結果移動到你的實驗資料夾
echo "📂 正在將最新實驗結果複製到專屬資料夾..."
mkdir -p "$EXP_OUTPUT_DIR"
cp -r "$PRED_DIR"/* "$EXP_OUTPUT_DIR"/

# 4. 實驗結束，清理與還原
echo "🧹 實驗完成，正在還原環境..."
rm -rf "$PRED_DIR"

if [ -d "$PRED_BACKUP" ]; then
    mv "$PRED_BACKUP" "$PRED_DIR"
fi

echo "=================================================="
echo "✅ 實驗順利結束！"
echo "🎉 原始 NQ 測試集 (500 題) 的裸考成績已經存放在："
echo "👉 $EXP_OUTPUT_DIR"
echo "=================================================="