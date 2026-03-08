#!/bin/bash
#SBATCH -J HASS
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8

set -x

nvcc -V
python -V

export PYTHONPATH=$(pwd):${PYTHONPATH}
export WANDB_API_KEY=05ac0c7fac19bec004160369c32723326fa8a618

PART=llama3-8b
PROJECT=LD-${PART}

MODEL=HASS
NAME=${MODEL}-100k

DATA_PATH=/mnt/inaisfs/data/home/liuf_criait/data
BASE_PATH=${DATA_PATH}/model/Llama-3-8B-Instruct
CONFIG_PATH=train/${PART}/${MODEL}_config.json

echo "start time: $(date)"

accelerate launch train/main_LD.py \
    --project ${PROJECT} \
    --name ${NAME} \
    --basepath ${BASE_PATH} \
    --tmpdir ge_data/${PART} \
    --cpdir checkpoints/${PART}/${NAME} \
    --configpath ${CONFIG_PATH} \
    --epoch 40 \
    --bs 1 \
    --topk 10 \
    --topk_w 0 \
    --forward_num_total 3 \
    --data_num 100000 \
    --detach

echo "end time: $(date)"
