#!/bin/bash
#SBATCH -J EAGLE
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8

set -x

nvcc -V
python -V

export PYTHONPATH=$(pwd):${PYTHONPATH}

PART=llama2-70b
PROJECT=LD-${PART}

MODEL=EAGLE2
NAME=${MODEL}-800k

DATA_PATH=/mnt/inaisfs/data/home/liuf_criait/data
BASE_PATH=${DATA_PATH}/model/Llama-2-70b-chat-hf
CONFIG_PATH=train/${PART}/${MODEL}_config.json

echo "start time: $(date)"

accelerate launch train/main_LD.py \
    --project ${PROJECT} \
    --name ${NAME} \
    --basepath ${BASE_PATH} \
    --tmpdir ge_data/${PART} \
    --cpdir checkpoints/${PART}/${NAME} \
    --configpath ${CONFIG_PATH} \
    --epoch 15 \
    --bs 1 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 800000

echo "end time: $(date)"
