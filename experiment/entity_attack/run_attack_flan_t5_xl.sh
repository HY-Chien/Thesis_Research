#!/bin/bash

# ================= 設定區 =================
GPU=7

# ★★★ 修改 1：直接換成 Teacher 原生模型 ★★★
# 這會去 HuggingFace 下載，或者讀取你 cache 裡的模型
MODEL_PATH="google/flan-t5-xl"

# 測試資料集 (你的實體替換檔)
TEST_FILE="./attack_dataset/predict_2wikimultihopqa_original.json"

# 輸出路徑 (為了不蓋掉原本的，換個名字)
BASE_OUTPUT_DIR="/home/S113065528/Adaptive-RAG/experiment/entity_attack"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/flan_t5_xl_origin_2wikimultihopqa_result_epoch_30"

mkdir -p ${OUTPUT_DIR}
# =========================================

echo "=================================================="
echo "正在載入模型: ${MODEL_PATH}"
echo "測試資料集: ${TEST_FILE}"
echo "結果將輸出至: ${OUTPUT_DIR}"
echo "=================================================="

export PYTHONPATH=$PYTHONPATH:/home/S113065528/Adaptive-RAG/classifier

# ★★★ 程式碼本體：參數完全沒動 ★★★
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
echo "請檢查輸出檔案: ${OUTPUT_DIR}/prediction_2wikimultihopqa.json"
echo "=================================================="