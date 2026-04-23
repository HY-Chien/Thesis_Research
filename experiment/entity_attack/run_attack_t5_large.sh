#!/bin/bash

# ================= 設定區 =================
GPU=7

# ★★★ 修正 1：模型路徑建議用絕對路徑，避免找不到 ★★★
# 請確認這個路徑是否正確指向您的模型資料夾
MODEL_PATH="/home/S113065528/Adaptive-RAG/classifier/outputs/musique_hotpot_wiki2_nq_tqa_sqd/model/t5-large/flan_t5_xl/epoch/30/2026_01_08/22_38_40"

# 測試資料集 (因為檔案就在當前目錄，所以用 ./)
TEST_FILE="./attack_dataset/predict_2wikimultihopqa_original.json"

# 輸出路徑
BASE_OUTPUT_DIR="/home/S113065528/Adaptive-RAG/experiment/entity_attack"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/t5_large_original_2wikimultihopqa_result_epoch_30"

mkdir -p ${OUTPUT_DIR}
# =========================================

echo "=================================================="
echo "正在載入模型: ${MODEL_PATH}"
echo "測試資料集: ${TEST_FILE}"
echo "結果將輸出至: ${OUTPUT_DIR}"
echo "=================================================="

# ★★★ 修正 2：指定 run_classifier.py 的完整路徑 ★★★
# 以及設定 PYTHONPATH 確保它能 import 到同一目錄下的其他檔案
export PYTHONPATH=$PYTHONPATH:/home/S113065528/Adaptive-RAG/classifier

CUDA_VISIBLE_DEVICES=${GPU} python /home/S113065528/Adaptive-RAG/classifier/run_classifier.py \
    --model_name_or_path ${MODEL_PATH} \
    --validation_file ${TEST_FILE} \
    --question_column question \
    --answer_column answer \
    --max_seq_length 384 \
    --doc_stride 128 \
    --per_device_eval_batch_size 100 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --val_column 'validation' \
    --do_eval

echo "=================================================="
echo "預測完成！"
echo "請檢查輸出檔案: ${OUTPUT_DIR}/prediction_nq.json"
echo "=================================================="