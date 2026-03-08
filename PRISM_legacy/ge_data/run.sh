#!/bin/bash
#SBATCH -J DATA
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8

SCRIPT=ge_data_all_llama2chat.py

OUTPUT_DIR=llama2-70b

DATA_DIR=/mnt/inaisfs/data/home/liuf_criait/data
DATASET_DIR=${DATA_DIR}/dataset
MODEL_DIR=${DATA_DIR}/model/Llama-2-70b-chat-hf

echo "start time: $(date)"

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATASET_DIR}/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --model_path ${MODEL_DIR} \
    --dataset_name ShareGPT \
    --num_rows 68000

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATASET_DIR}/ultrachat_200k \
    --model_path ${MODEL_DIR} \
    --dataset_name UltraChat \
    --num_rows 463000

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATASET_DIR}/OpenThoughts2-1M \
    --model_path ${MODEL_DIR} \
    --dataset_name OpenThoughts2 \
    --num_rows 269000

echo "end time: $(date)"
