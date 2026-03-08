#!/bin/bash
#SBATCH -J DATA
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8

SCRIPT=ge_data_all_llama3.py

OUTPUT_DIR=llama3-8b
DATA_DIR=/mnt/inaisfs/data/home/liuf_criait/data/dataset
MODEL_DIR=/mnt/inaisfs/data/home/liuf_criait/data/model/Llama-3-8B-Instruct

echo "start time: $(date)"

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATA_DIR}/ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --model_path ${MODEL_DIR} \
    --dataset_name ShareGPT \
    --num_rows 68000 \
    --num_gpus 8

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATA_DIR}/ultrachat_200k \
    --model_path ${MODEL_DIR} \
    --dataset_name UltraChat \
    --num_rows 463000 \
    --num_gpus 8

python allocation.py \
    --script ${SCRIPT} \
    --outdir ${OUTPUT_DIR} \
    --data_path ${DATA_DIR}/OpenThoughts2-1M \
    --model_path ${MODEL_DIR} \
    --dataset_name OpenThoughts2 \
    --num_rows 269000 \
    --num_gpus 8

echo "end time: $(date)"
