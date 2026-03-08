#!/bin/bash
#SBATCH -J PRISM
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --gres=gpu:1

export PYTHONPATH=$(pwd):${PYTHONPATH}

PROJECT=llama3-8b

MODEL=PRISM
NAME=${MODEL}-800k

EA_MODEL_DIR=checkpoints/${PROJECT}/${NAME}
EA_CONFIG_PATH=train/llama3-8b/${MODEL}_config.json
BASE_MODEL_PATH=/mnt/inaisfs/data/home/liuf_criait/data/model/Llama-3-8B-Instruct

echo "start time: $(date)"

for iter in {0..0}
do
  echo "iter: ${iter}"

  # EA_MODEL_PATH=${EA_MODEL_DIR}/state_${iter}
  EA_MODEL_PATH=${EA_MODEL_DIR}
  cp ${EA_CONFIG_PATH} ${EA_MODEL_PATH}/config.json

  for bench_name in "mt_bench" "humaneval" "gsm8k" "alpaca" "sum" "qa"
  do
    echo "bench_name: ${bench_name}"

    for temperature in "0.0" "1.0"
    do
      echo "temperature: ${temperature}"

      CUDA_VISIBLE_DEVICES=0 python evaluation/gen_ea_answer_llama3chat.py \
        --ea-model-path ${EA_MODEL_PATH} \
        --base-model-path ${BASE_MODEL_PATH} \
        --model-id ${PROJECT}/${NAME} \
        --bench-name ${bench_name} \
        --total-token 60 \
        --depth 5 \
        --top-k 10 \
        --temperature ${temperature}

#      CUDA_VISIBLE_DEVICES=0 python evaluation/gen_baseline_answer_llama3chat.py \
#        --ea-model-path ${EA_MODEL_PATH} \
#        --base-model-path ${BASE_MODEL_PATH} \
#        --model-id ${PROJECT}/Naive \
#        --bench-name ${bench_name} \
#        --temperature ${temperature}

      python evaluation/summary.py \
        --model_path ${BASE_MODEL_PATH} \
        --baseline_json ${bench_name}/${PROJECT}/Naive-temperature-${temperature}.jsonl \
        --LD_json ${bench_name}/${PROJECT}/${NAME}-temperature-${temperature}.jsonl

      python evaluation/conditional_accept_rate.py \
        --input_file ${bench_name}/${PROJECT}/${NAME}-temperature-${temperature}.jsonl
    done
  done
done

echo "end time: $(date)"
