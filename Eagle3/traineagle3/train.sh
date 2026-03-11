#!/bin/bash
#SBATCH -J Eagle3-400k
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --gres=gpu:8

set -x

nvcc -V
python -V

export PYTHONPATH=$(pwd):${PYTHONPATH}
export WANDB_API_KEY=05ac0c7fac19bec004160369c32723326fa8a618

export HF_HUB_CACHE=cache/hub
export HF_DATASETS_CACHE=cache/datasets

NAME=Eagle3-400k

BASE_PATH=/mnt/inaisfs/data/home/liuf_criait/data/model/Llama-3-8B-Instruct
DATA_PATH=/mnt/inaisfs/data/home/liuf_criait/data/dataset

echo "start time: $(date)"

deepspeed main.py \
    --deepspeed_config ds_config.json \
    --name ${NAME} \
    --basepath ${BASE_PATH} \
    --datapath ${DATA_PATH} \
    --data_num 400000 \
    --num_epochs 25 \
    --savedir checkpoints/llama3-8b/${NAME}

echo "end time: $(date)"
